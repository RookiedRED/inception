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
    let trackingService = TrackingService()
    let miniMapService = MiniMapService()

    // MARK: - Session (for non-rendering uses like MiniMap)

    var arSession: ARSession { cameraService.session }

    // MARK: - Published UI State

    @Published var cameraImage: UIImage?                     // camera feed (replaces ARSCNView)
    @Published var overlay: UIImage?
    @Published var trackedObjects: [TrackedObject] = []
    @Published var inferenceMs: Double = 0
    @Published var detectionCount: Int = 0
    @Published var orientation: AppOrientation = .normal

    // MARK: - Internal State

    private var subscriptions = Set<AnyCancellable>()
    private var frameCount = 0
    private let inferenceInterval = 2
    private var isInferring = false
    private var didSetupPipeline = false

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
        let dbg = ARCameraService.FrameDebug.shared
        let t0 = CACurrentMediaTime()

        // Generate camera preview image (GPU-accelerated, ~1ms)
        let cameraImg = ARCameraService.imageFromBuffer(pixelBuffer)
        let tImg = CACurrentMediaTime()
        dbg.recordImageTime((tImg - t0) * 1000)

        frameCount += 1
        let shouldInfer = (frameCount % inferenceInterval == 0) && !isInferring

        if shouldInfer {
            isInferring = true
            let dispatchTime = CACurrentMediaTime()

            let cameraService = self.cameraService
            inferenceService.run(pixelBuffer: pixelBuffer) { [weak self, cameraService] result in
                guard let self else {
                    print("[FrameDebug] WARNING: self is nil in inference callback, unlocking gate")
                    ARCameraService.FrameDebug.shared.recordCallbackNil()
                    cameraService.endProcessing()
                    return
                }

                let callbackTime = CACurrentMediaTime()
                let inferLatency = (callbackTime - dispatchTime) * 1000

                defer {
                    self.isInferring = false
                    self.cameraService.endProcessing()
                    let totalMs = (CACurrentMediaTime() - dispatchTime) * 1000
                    print(String(format: "[FrameDebug] INFER: latency=%.1fms total=%.1fms thread=%@",
                                 inferLatency, totalMs,
                                 Thread.isMainThread ? "main" : "bg"))
                }

                switch result {
                case .success(let output):
                    let objects = self.trackingService.update(
                        detections: output.detections,
                        context: context
                    )

                    let detections = objects.map(\.detection)
                    let overlay = self.inferenceService.renderOverlay(detections)

                    Task { @MainActor in
                        self.miniMapService.updateCamera(
                            transform: context.cameraTransform,
                            orientation: self.orientation
                        )
                        self.miniMapService.updateTrackedObjects(objects)

                        self.cameraImage = cameraImg
                        self.overlay = overlay
                        self.trackedObjects = objects
                        self.inferenceMs = output.inferenceMs
                        self.detectionCount = objects.count
                    }

                case .failure(let error):
                    print("[FrameDebug] Inference FAILED: \(error)")
                }
            }

        } else {
            // Tracking-only frame
            let tTrack0 = CACurrentMediaTime()
            let objects = trackingService.predict(context: context)
            let tTrack1 = CACurrentMediaTime()

            let detections = objects.map(\.detection)
            let overlay = inferenceService.renderOverlay(detections)
            let tRender = CACurrentMediaTime()

            Task { @MainActor in
                self.miniMapService.updateCamera(
                    transform: context.cameraTransform,
                    orientation: self.orientation
                )
                self.miniMapService.updateTrackedObjects(objects)

                self.cameraImage = cameraImg
                self.overlay = overlay
                self.trackedObjects = objects
                self.detectionCount = objects.count
            }

            cameraService.endProcessing()

            let totalMs = (CACurrentMediaTime() - t0) * 1000
            let trackMs = (tTrack1 - tTrack0) * 1000
            let renderMs = (tRender - tTrack1) * 1000
            // Only print occasionally to avoid log spam
            if frameCount % 30 == 0 {
                print(String(format: "[FrameDebug] TRACK: img=%.1fms track=%.1fms render=%.1fms total=%.1fms",
                             (tImg - t0) * 1000, trackMs, renderMs, totalMs))
            }
        }
    }
}
