//
//  MiniMapView.swift
//  iDriveBot
//
//  SceneKit wrapper for minimap rendering
//

import SwiftUI
import SceneKit

struct MiniMapView: UIViewRepresentable {
    let scene: SCNScene
    let isZoomEnabled: Bool
    let onZoomChanged: (Double) -> Void
    let onPanChanged: (CGPoint, CGSize) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onZoomChanged: onZoomChanged, onPanChanged: onPanChanged)
    }

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = scene

        // Visual style
        view.backgroundColor = UIColor(white: 0.94, alpha: 0.96)
        view.autoenablesDefaultLighting = false
        view.allowsCameraControl = false
        view.rendersContinuously = false
        view.isPlaying = false
        view.antialiasingMode = .none
        view.preferredFramesPerSecond = 30

        let pinchGesture = UIPinchGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePinch(_:))
        )
        pinchGesture.cancelsTouchesInView = false
        view.addGestureRecognizer(pinchGesture)

        let panGesture = UIPanGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePan(_:))
        )
        panGesture.cancelsTouchesInView = false
        panGesture.maximumNumberOfTouches = 2
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
        uiView.isUserInteractionEnabled = isZoomEnabled
    }
}
extension MiniMapView {
    final class Coordinator: NSObject {
        private let onZoomChanged: (Double) -> Void
        private let onPanChanged: (CGPoint, CGSize) -> Void
        private weak var view: SCNView?
        private var currentZoomScale: Double = 1.0
        private var gestureStartZoomScale: Double = 1.0
        private var isZoomEnabled = false

        init(
            onZoomChanged: @escaping (Double) -> Void,
            onPanChanged: @escaping (CGPoint, CGSize) -> Void
        ) {
            self.onZoomChanged = onZoomChanged
            self.onPanChanged = onPanChanged
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
    }
}
