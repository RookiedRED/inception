//
//  RootView.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/8/26.
//

import SwiftUI

struct RootView: View {
    @StateObject private var viewModel = DriveViewModel()

    var body: some View {
        ZStack {
            DetectionOverlayView(
                image: viewModel.overlay,
                trackedObjects: viewModel.trackedObjects,
                orientation: viewModel.orientation
            )
            .ignoresSafeArea()

            HUDView(
                inferenceMs: viewModel.inferenceMs,
                detectionCount: viewModel.detectionCount,
                orientation: viewModel.orientation
            )
            .frame(
                maxWidth: .infinity,
                maxHeight: .infinity,
                alignment: viewModel.orientation.isFlipped ? .bottom : .top
            )
            .padding(.horizontal, 16)
            .padding(.top, 16)
            .padding(.bottom, 16)

            MiniMapView(scene: viewModel.miniMapService.scene)
                .frame(width: 180, height: 180)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.white.opacity(0.8), lineWidth: 1)
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
                .padding(.trailing, 16)
                .padding(.bottom, 24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background {
            CameraPreviewView(image: viewModel.cameraImage)
        }
        .onAppear { viewModel.start() }
        .onDisappear { viewModel.stop() }
    }
}
