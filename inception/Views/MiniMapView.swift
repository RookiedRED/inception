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

        // Interaction off - this is a minimap, not a free camera viewport
        view.isUserInteractionEnabled = false

        // Performance / debug
        view.showsStatistics = false
        view.debugOptions = []
        view.isJitteringEnabled = false

        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        if uiView.scene !== scene {
            uiView.scene = scene
        }
    }
}
