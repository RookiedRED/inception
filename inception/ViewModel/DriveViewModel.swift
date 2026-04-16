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
        }

        frameCount += 1
        let shouldInfer = !isInferring

        if shouldInfer {
            isInferring = true
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
                    let objects = self.makeDisplayObjects(from: output.detections, timestamp: context.timestamp)
                    Task { @MainActor in
                        self.updateMiniMapCamera(with: context)
                        self.inferenceMs = output.inferenceMs
                        self.miniMapService.updateTrackedObjects(objects)
                        self.trackedObjects = objects
                        self.detectionCount = objects.count
                    }

                case .failure(let error):
                    print("[FrameDebug] Inference FAILED: \(error)")
                }
            }

        } else {
            Task { @MainActor in
                self.updateMiniMapCamera(with: context)
            }

            cameraService.endProcessing()

            // Only print occasionally to avoid log spam
            if frameCount % 30 == 0 {
                let totalMs = (CACurrentMediaTime() - t0) * 1000
                print(String(format: "[FrameDebug] PREVIEW: total=%.1fms", totalMs))
            }
        }
    }

    private func makeDisplayObjects(from detections: [Detection], timestamp: TimeInterval) -> [TrackedObject] {
        detections.enumerated().map { index, detection in
            TrackedObject(
                id: index,
                detection: detection,
                velocity: .zero,
                age: 1,
                timeSinceUpdate: 0,
                visibility: .visible,
                depth: nil,
                worldPosition: nil,
                lastSeenTimestamp: timestamp
            )
        }
    }

    private func updateMiniMapCamera(with context: ARFrameContext) {
        miniMapService.updateCamera(
            transform: context.cameraTransform,
            orientation: orientation
        )
    }
}
