//
//  RootView.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/8/26.
//

import SwiftUI

struct RootView: View {
    @StateObject private var viewModel = DriveViewModel()
    @State private var isMiniMapExpanded = false

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                CameraPreviewView(pixelBuffer: viewModel.cameraPixelBuffer)
                    .ignoresSafeArea()

                DetectionOverlayView(
                    trackedObjects: viewModel.trackedObjects,
                    orientation: viewModel.orientation,
                    imageResolution: viewModel.imageResolution
                )
                .ignoresSafeArea()

                miniMap()
                
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
                .padding(.all, 16)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .ignoresSafeArea()   // <- 加這裡
        }
        .onAppear { viewModel.start() }
        .onDisappear { viewModel.stop() }
        .animation(.spring(response: 0.32, dampingFraction: 0.88), value: isMiniMapExpanded)
    }

    private func miniMap() -> some View {
        MiniMapView(
            scene: viewModel.miniMapService.scene,
            isZoomEnabled: isMiniMapExpanded,
            onZoomChanged: { scale in
                viewModel.setMiniMapZoomScale(scale)
            },
            onPanChanged: { translation, viewportSize in
                viewModel.panExpandedMiniMap(by: translation, viewportSize: viewportSize)
            }
        )
            .if(isMiniMapExpanded) { view in
                view
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .ignoresSafeArea()
            }
            .if(!isMiniMapExpanded) { view in
                view
                    .frame(width: 180, height: 180)
                    .aspectRatio(1, contentMode: .fit)
            }
            .clipShape(RoundedRectangle(cornerRadius: isMiniMapExpanded ? 0 : 12))
            .overlay {
                RoundedRectangle(cornerRadius: isMiniMapExpanded ? 0 : 12)
                    .stroke(Color.white.opacity(0.85), lineWidth: isMiniMapExpanded ? 0 : 1)
            }
            .shadow(color: .black.opacity(0.22), radius: isMiniMapExpanded ? 0 : 18, y: 8)
            .frame(
                maxWidth: .infinity,
                maxHeight: .infinity,
                alignment: isMiniMapExpanded ? .center : .bottomTrailing
            )
            .padding(.trailing, isMiniMapExpanded ? 0 : 16)
            .padding(.bottom, isMiniMapExpanded ? 0 : 24)
            .contentShape(Rectangle())
            .onTapGesture {
                if !isMiniMapExpanded {
                    setMiniMapExpanded(true)
                }
            }
    }

    private func setMiniMapExpanded(_ isExpanded: Bool) {
        isMiniMapExpanded = isExpanded
        viewModel.setMiniMapPresentationMode(isExpanded ? .expanded : .compact)
    }
}

private extension View {
    @ViewBuilder
    func `if`<Content: View>(_ condition: Bool, transform: (Self) -> Content) -> some View {
        if condition {
            transform(self)
        } else {
            self
        }
    }
}
