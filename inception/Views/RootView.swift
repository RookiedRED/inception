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
    @State private var showsNavigationPrompt = true

    var body: some View {
        GeometryReader { proxy in
            let safeInsets = proxy.safeAreaInsets

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
                .padding(.top, viewModel.orientation.isFlipped ? 0 : safeInsets.top + 16)
                .padding(.bottom, viewModel.orientation.isFlipped ? safeInsets.bottom + 16 : 0)
                .padding(.horizontal, 16)

                if showsNavigationPrompt, viewModel.isNavigating, let dist = viewModel.navigationDistance {
                    NavigationSidePromptView(
                        distanceMeters: dist,
                        isUpdating: viewModel.isCalculatingNavigation,
                        onCancel: { viewModel.cancelNavigation() }
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
                    .padding(.leading, 16)
                    .padding(.top, safeInsets.top + 76)
                    .padding(.bottom, safeInsets.bottom + 16)
                    .transition(.move(edge: .leading).combined(with: .opacity))
                }

                if let landmark = viewModel.selectedLandmark, !viewModel.isNavigating {
                    LandmarkCalloutView(
                        landmark: landmark,
                        distanceMeters: viewModel.distanceToLandmark(landmark),
                        isLoading: viewModel.isCalculatingNavigation,
                        onConfirm: { viewModel.confirmSelectedLandmarkNavigation() },
                        onDismiss: { viewModel.selectLandmark(id: nil) }
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .trailing)
                    .padding(.trailing, 16)
                    .padding(.top, safeInsets.top + 92)
                    .padding(.bottom, max(safeInsets.bottom + 20, isMiniMapExpanded ? safeInsets.bottom + 20 : 116))
                    .transition(.move(edge: .trailing).combined(with: .opacity))
                }

                if viewModel.navigationArrivedMessage {
                    ArrivedBannerView()
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                        .padding(.top, safeInsets.top + 18)
                        .padding(.horizontal, 20)
                        .transition(.scale(scale: 0.85).combined(with: .opacity))
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .ignoresSafeArea()
        .onAppear { viewModel.start() }
        .onDisappear { viewModel.stop() }
        .onChange(of: viewModel.isNavigating) { _, isNavigating in
            if isNavigating {
                showsNavigationPrompt = true
            } else {
                showsNavigationPrompt = false
            }
        }
        .onChange(of: viewModel.selectedLandmark?.id) { _, selectedID in
            if selectedID != nil {
                showsNavigationPrompt = false
            } else if viewModel.isNavigating {
                showsNavigationPrompt = true
            }
        }
        .onChange(of: viewModel.navigationArrivedMessage) { _, isVisible in
            if isVisible {
                showsNavigationPrompt = false
            } else if viewModel.isNavigating {
                showsNavigationPrompt = true
            }
        }
        .animation(.spring(response: 0.32, dampingFraction: 0.88), value: isMiniMapExpanded)
        .animation(.spring(response: 0.3, dampingFraction: 0.85), value: viewModel.isNavigating)
        .animation(.spring(response: 0.3, dampingFraction: 0.85), value: viewModel.selectedLandmark?.id)
        .animation(.spring(response: 0.25, dampingFraction: 0.84), value: viewModel.isCalculatingNavigation)
        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: viewModel.navigationArrivedMessage)
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
            },
            onToggleRequested: {
                if isMiniMapExpanded, dismissVisiblePopupIfNeeded() {
                    return
                }
                if isMiniMapExpanded {
                    viewModel.selectLandmark(id: nil)
                }
                setMiniMapExpanded(!isMiniMapExpanded)
            },
            onLandmarkTapped: { id in
                // Expand minimap if compact, then select landmark
                if !isMiniMapExpanded {
                    setMiniMapExpanded(true)
                }
                viewModel.selectLandmark(id: id)
            },
            landmarkHitTest: { point, scnView in
                viewModel.miniMapService.landmarkID(at: point, in: scnView)
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
    }

    private func setMiniMapExpanded(_ isExpanded: Bool) {
        isMiniMapExpanded = isExpanded
        viewModel.setMiniMapPresentationMode(isExpanded ? .expanded : .compact)
    }

    private func dismissVisiblePopupIfNeeded() -> Bool {
        if viewModel.selectedLandmark != nil {
            viewModel.selectLandmark(id: nil)
            return true
        }

        if viewModel.navigationArrivedMessage {
            viewModel.navigationArrivedMessage = false
            return true
        }

        if showsNavigationPrompt {
            showsNavigationPrompt = false
            return true
        }

        return false
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
