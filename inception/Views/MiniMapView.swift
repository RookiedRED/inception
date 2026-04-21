//
//  MiniMapView.swift
//  iDriveBot
//
//  SceneKit wrapper for minimap rendering
//

import SwiftUI
import SceneKit

/// SwiftUI wrapper around the SceneKit-based minimap renderer.
struct MiniMapView: UIViewRepresentable {
    let scene: SCNScene
    let isZoomEnabled: Bool
    let onZoomChanged: (Double) -> Void
    let onPanChanged: (CGPoint, CGSize) -> Void
    let onToggleRequested: () -> Void
    /// Called with the landmark UUID when the user taps a landmark node,
    /// or nil when tapping empty space.
    let onLandmarkTapped: (UUID?) -> Void
    /// Returns the landmark UUID at `point` in `view`, or nil if none.
    let landmarkHitTest: (CGPoint, SCNView) -> UUID?

    func makeCoordinator() -> Coordinator {
        Coordinator(
            onZoomChanged: onZoomChanged,
            onPanChanged: onPanChanged,
            onToggleRequested: onToggleRequested,
            onLandmarkTapped: onLandmarkTapped,
            landmarkHitTest: landmarkHitTest
        )
    }

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = scene

        // Visual style
        view.backgroundColor = UIColor(white: 0.94, alpha: 0.96)
        view.autoenablesDefaultLighting = false
        view.allowsCameraControl = false
        view.rendersContinuously = true
        view.isPlaying = true
        view.antialiasingMode = .none
        view.preferredFramesPerSecond = 60

        let tapGesture = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleTap(_:))
        )
        tapGesture.cancelsTouchesInView = false
        tapGesture.numberOfTouchesRequired = 1
        view.addGestureRecognizer(tapGesture)

        let pinchGesture = UIPinchGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePinch(_:))
        )
        pinchGesture.cancelsTouchesInView = false
        pinchGesture.delegate = context.coordinator
        view.addGestureRecognizer(pinchGesture)

        let panGesture = UIPanGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePan(_:))
        )
        panGesture.cancelsTouchesInView = false
        panGesture.minimumNumberOfTouches = 2
        panGesture.maximumNumberOfTouches = 2
        panGesture.delegate = context.coordinator
        view.addGestureRecognizer(panGesture)

        // Performance / debug
        view.showsStatistics = false
        view.debugOptions = []
        view.isJitteringEnabled = false

        context.coordinator.attach(to: view)
        context.coordinator.setZoomEnabled(isZoomEnabled)
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        if uiView.scene !== scene {
            uiView.scene = scene
        }
        context.coordinator.setZoomEnabled(isZoomEnabled)
        uiView.isUserInteractionEnabled = true
        uiView.isPlaying = true
    }
}

extension MiniMapView {
    /// Coordinates SceneKit gestures and forwards them to the SwiftUI view model layer.
    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        private let onZoomChanged: (Double) -> Void
        private let onPanChanged: (CGPoint, CGSize) -> Void
        private let onToggleRequested: () -> Void
        private let onLandmarkTapped: (UUID?) -> Void
        private let landmarkHitTest: (CGPoint, SCNView) -> UUID?
        private weak var view: SCNView?
        private var currentZoomScale: Double = 1.0
        private var gestureStartZoomScale: Double = 1.0
        private var isZoomEnabled = false

        init(
            onZoomChanged: @escaping (Double) -> Void,
            onPanChanged: @escaping (CGPoint, CGSize) -> Void,
            onToggleRequested: @escaping () -> Void,
            onLandmarkTapped: @escaping (UUID?) -> Void,
            landmarkHitTest: @escaping (CGPoint, SCNView) -> UUID?
        ) {
            self.onZoomChanged = onZoomChanged
            self.onPanChanged = onPanChanged
            self.onToggleRequested = onToggleRequested
            self.onLandmarkTapped = onLandmarkTapped
            self.landmarkHitTest = landmarkHitTest
        }

        func attach(to view: SCNView) {
            self.view = view
        }

        func setZoomEnabled(_ isEnabled: Bool) {
            isZoomEnabled = isEnabled
            if !isEnabled {
                gestureStartZoomScale = currentZoomScale
            }
        }

        @objc
        func handleTap(_ gesture: UITapGestureRecognizer) {
            guard gesture.state == .ended, let view else { return }
            let point = gesture.location(in: view)
            if let id = landmarkHitTest(point, view) {
                onLandmarkTapped(id)
            } else {
                onToggleRequested()
            }
        }

        @objc
        func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            guard isZoomEnabled else { return }

            switch gesture.state {
            case .began:
                gestureStartZoomScale = currentZoomScale
            case .changed, .ended:
                let nextScale = gestureStartZoomScale * Double(gesture.scale)
                let clampedScale = min(max(nextScale, 0.55), 2.4)
                currentZoomScale = clampedScale
                onZoomChanged(clampedScale)
            default:
                break
            }
        }

        @objc
        func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard isZoomEnabled, let view else { return }

            let translation = gesture.translation(in: view)
            guard translation != .zero else { return }

            onPanChanged(translation, view.bounds.size)
            gesture.setTranslation(.zero, in: view)
        }

        func gestureRecognizer(
            _ gestureRecognizer: UIGestureRecognizer,
            shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer
        ) -> Bool {
            let recognizers = [gestureRecognizer, otherGestureRecognizer]
            let containsPinch = recognizers.contains { $0 is UIPinchGestureRecognizer }
            let containsPan = recognizers.contains { $0 is UIPanGestureRecognizer }
            return containsPinch && containsPan
        }
    }
}
