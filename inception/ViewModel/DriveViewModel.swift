//
//  DriveViewModel.swift
//  iDriveBot
//

import Foundation
import Combine
import SwiftUI
import ARKit
import UIKit
import CoreVideo
import simd

@MainActor
final class DriveViewModel: ObservableObject {

    // MARK: - Services

    let cameraService = ARCameraService()
    let inferenceService = InferenceService()
    let miniMapService = MiniMapService()

    // MARK: - Session (for non-rendering uses like MiniMap)

    var arSession: ARSession { cameraService.session }

    // MARK: - Published UI State

    @Published var cameraPixelBuffer: CVPixelBuffer?
    @Published var trackedObjects: [TrackedObject] = []
    @Published var inferenceMs: Double = 0
    @Published var detectionCount: Int = 0
    @Published var orientation: AppOrientation = .normal
    @Published var imageResolution: CGSize = .zero

    // MARK: - Internal State

    private var subscriptions = Set<AnyCancellable>()
    private var frameCount = 0
    private var isInferring = false
    private var didSetupPipeline = false
    private var latestCameraTimestamp: TimeInterval = 0
    private var latestTrackedObjectsTimestamp: TimeInterval = 0
    private var lastInferenceTime: CFTimeInterval = 0
    /// EWMA of recent inference latency (ms). Seed at 100ms as a safe starting estimate.
    private var ewmaInferenceMs: Double = 100.0
    private let ewmaAlpha: Double = 0.15  // smoothing factor; lower = more stable

    /// Adaptive minimum interval targeting 65% Neural Engine duty cycle.
    /// If inference is fast (70ms) → ~108ms interval (≈9fps).
    /// If throttled (200ms) → ~308ms interval (≈3fps) — backs off automatically.
    private var adaptiveMinInterval: CFTimeInterval {
        let targetMs = ewmaInferenceMs / 0.65
        // Clamp between 6fps (heavy scene) and 15fps (fast device)
        return max(1.0 / 15.0, min(1.0 / 6.0, targetMs / 1000.0))
    }


    // MARK: - Lifecycle

    func start() {
        if !didSetupPipeline {
            setupPipeline()
            didSetupPipeline = true
        }
        cameraService.start(reset: true)
    }

    func stop() {
        cameraService.stop()
        isInferring = false
    }

    // MARK: - Orientation

    func updateDeviceOrientation(_ deviceOrientation: UIDeviceOrientation) {
        orientation = AppOrientation(from: deviceOrientation)
    }

    // MARK: - Pipeline Setup

    private func setupPipeline() {
        let cameraService = self.cameraService
        cameraService.onFrame = { [weak self, cameraService] pixelBuffer, context in
            guard let self else {
                ARCameraService.FrameDebug.shared.recordCallbackNil()
                cameraService.endProcessing()
                return
            }
            self.handleARFrame(pixelBuffer, context: context)
        }

        var isUpdatingMesh = false
        cameraService.meshAnchorPublisher
            .sink { [weak self] anchors in
                guard let self, !isUpdatingMesh else { return }
                isUpdatingMesh = true
                Task { @MainActor in
                    self.miniMapService.updateMeshAnchors(anchors)
                    isUpdatingMesh = false
                }
            }
            .store(in: &subscriptions)
    }

    // MARK: - Frame Handling

    private func handleARFrame(_ pixelBuffer: CVPixelBuffer, context: ARFrameContext) {
        let t0 = CACurrentMediaTime()

        // Keep the preview on the freshest frame available, even while inference is busy.
        Task { @MainActor in
            guard context.timestamp >= self.latestCameraTimestamp else { return }
            self.latestCameraTimestamp = context.timestamp
            self.cameraPixelBuffer = pixelBuffer
            self.imageResolution = context.imageResolution
            self.updateMiniMapCamera(with: context)
        }

        frameCount += 1
        let now = CACurrentMediaTime()
        let shouldInfer = !isInferring && (now - lastInferenceTime >= adaptiveMinInterval)

        if shouldInfer {
            isInferring = true
            lastInferenceTime = now
            // Release the AR frame gate immediately so camera preview stays responsive
            // while inference runs on its private queue.
            cameraService.endProcessing()

            let dispatchTime = CACurrentMediaTime()
            inferenceService.run(pixelBuffer: pixelBuffer) { [weak self] result in
                guard let self else {
                    ARCameraService.FrameDebug.shared.recordCallbackNil()
                    return
                }

                let inferLatency = (CACurrentMediaTime() - dispatchTime) * 1000
                defer {
                    self.isInferring = false
                    print(String(format: "[FrameDebug] INFER: latency=%.1fms thread=%@",
                                 inferLatency, Thread.isMainThread ? "main" : "bg"))
                }

                switch result {
                case .success(let output):
                    // Update EWMA so adaptiveMinInterval adjusts on next frame.
                    self.ewmaInferenceMs = self.ewmaAlpha * output.inferenceMs
                        + (1.0 - self.ewmaAlpha) * self.ewmaInferenceMs

                    // Completion already runs on main thread — compute objects here directly.
                    // makeDisplayObjects is O(detections) with depth lookups, typically <1ms.
                    let objects = Self.makeDisplayObjects(
                        from: output.detections,
                        context: context,
                        timestamp: context.timestamp
                    )
                    guard context.timestamp >= self.latestTrackedObjectsTimestamp else { return }
                    self.latestTrackedObjectsTimestamp = context.timestamp
                    self.inferenceMs = output.inferenceMs
                    self.miniMapService.updateTrackedObjects(objects)
                    self.trackedObjects = objects
                    self.detectionCount = objects.count

                case .failure(let error):
                    print("[FrameDebug] Inference FAILED: \(error)")
                }
            }

        } else {
            cameraService.endProcessing()

            // Only print occasionally to avoid log spam
            if frameCount % 30 == 0 {
                let totalMs = (CACurrentMediaTime() - t0) * 1000
                print(String(format: "[FrameDebug] PREVIEW: total=%.1fms", totalMs))
            }
        }
    }

    private nonisolated static func makeDisplayObjects(
        from detections: [Detection],
        context: ARFrameContext,
        timestamp: TimeInterval
    ) -> [TrackedObject] {
        detections.enumerated().map { index, detection in
            let objectEstimate = estimateWorldPosition(for: detection, context: context)
            return TrackedObject(
                id: index,
                detection: detection,
                velocity: .zero,
                age: 1,
                timeSinceUpdate: 0,
                visibility: .visible,
                depth: objectEstimate.depth,
                worldPosition: objectEstimate.worldPosition,
                lastSeenTimestamp: timestamp
            )
        }
    }

    private nonisolated static func estimateWorldPosition(
        for detection: Detection,
        context: ARFrameContext
    ) -> (worldPosition: simd_float3?, depth: Float?) {
        guard let sceneDepth = context.sceneDepth else {
            return (nil, nil)
        }

        let imageResolution = context.imageResolution
        let depthResolution = sceneDepth.resolution
        guard imageResolution.width > 0,
              imageResolution.height > 0,
              depthResolution.width > 0,
              depthResolution.height > 0 else {
            return (nil, nil)
        }

        let bbox = detection.bbox
        let imagePoint = CGPoint(
            x: min(max(bbox.midX * imageResolution.width, 0), imageResolution.width),
            y: min(max((bbox.maxY - 0.02) * imageResolution.height, 0), imageResolution.height)
        )
        let depthPoint = CGPoint(
            x: imagePoint.x * depthResolution.width / imageResolution.width,
            y: imagePoint.y * depthResolution.height / imageResolution.height
        )

        guard let sampledDepth = sampledDepth(around: depthPoint, sceneDepth: sceneDepth) else {
            return (nil, nil)
        }

        let fx = sceneDepth.intrinsics.columns.0.x
        let fy = sceneDepth.intrinsics.columns.1.y
        let cx = sceneDepth.intrinsics.columns.2.x
        let cy = sceneDepth.intrinsics.columns.2.y
        guard fx > 0, fy > 0 else {
            return (nil, sampledDepth)
        }

        let cameraX = (Float(depthPoint.x) - cx) * sampledDepth / fx
        let cameraY = (Float(depthPoint.y) - cy) * sampledDepth / fy
        let cameraSpacePoint = simd_float4(cameraX, cameraY, -sampledDepth, 1)
        let worldPoint = context.cameraTransform * cameraSpacePoint

        return (simd_float3(worldPoint.x, worldPoint.y, worldPoint.z), sampledDepth)
    }

    private nonisolated static func sampledDepth(
        around point: CGPoint,
        sceneDepth: ARFrameContext.SceneDepthData
    ) -> Float? {
        let centerX = Int(point.x.rounded())
        let centerY = Int(point.y.rounded())
        let width = Int(sceneDepth.resolution.width)
        let height = Int(sceneDepth.resolution.height)

        CVPixelBufferLockBaseAddress(sceneDepth.depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(sceneDepth.depthMap, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(sceneDepth.depthMap) else { return nil }
        let floatsPerRow = CVPixelBufferGetBytesPerRow(sceneDepth.depthMap) / MemoryLayout<Float32>.stride
        let ptr = base.assumingMemoryBound(to: Float32.self)

        var candidates: [Float] = []
        candidates.reserveCapacity(25)

        for offsetY in -2...2 {
            for offsetX in -2...2 {
                let sampleX = centerX + offsetX
                let sampleY = centerY + offsetY
                guard sampleX >= 0, sampleX < width, sampleY >= 0, sampleY < height else { continue }
                let depth = ptr[sampleY * floatsPerRow + sampleX]
                guard depth.isFinite, depth > 0 else { continue }
                candidates.append(depth)
            }
        }

        guard !candidates.isEmpty else { return nil }
        candidates.sort()
        return candidates[candidates.count / 2]
    }

    private func updateMiniMapCamera(with context: ARFrameContext) {
        miniMapService.updateCamera(
            transform: context.cameraTransform,
            orientation: orientation
        )
    }

    func setMiniMapPresentationMode(_ mode: MiniMapService.PresentationMode) {
        miniMapService.setPresentationMode(mode)
    }

    func setMiniMapZoomScale(_ scale: Double) {
        miniMapService.setExpandedZoomScale(scale)
    }

    func panExpandedMiniMap(by translation: CGPoint, viewportSize: CGSize) {
        miniMapService.panExpandedMap(byScreenTranslation: translation, viewportSize: viewportSize)
    }
}
