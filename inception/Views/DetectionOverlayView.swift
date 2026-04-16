//
//  DetectionOverlayView.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/8/26.
//

import SwiftUI

struct DetectionOverlayView: View {
    let trackedObjects: [TrackedObject]
    let orientation: AppOrientation
    let imageResolution: CGSize

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                ForEach(trackedObjects) { obj in
                    let box = displayRect(
                        for: obj.detection.bbox,
                        viewSize: geo.size,
                        imageResolution: imageResolution
                    )

                    if !box.isNull && !box.isEmpty {
                        ZStack(alignment: .topLeading) {
                            Rectangle()
                                .stroke(Color(obj.detection.color), lineWidth: 2)
                                .background(
                                    Rectangle()
                                        .fill(Color(obj.detection.color).opacity(0.12))
                                )
                                .frame(width: box.width, height: box.height)
                                .position(x: box.midX, y: box.midY)

                            Text(obj.trackLabel)
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                                .foregroundColor(.white)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(Color(obj.detection.color).opacity(
                                    obj.visibility == .visible ? 0.75 : 0.4
                                ))
                                .rotationEffect(.degrees(orientation.isFlipped ? 180 : 0))
                                .position(
                                    x: min(box.minX + 52, geo.size.width - 52),
                                    y: max(box.minY + 10, 10)
                                )
                        }
                    }
                }
            }
        }
        .allowsHitTesting(false)
    }

    private func displayRect(for bbox: CGRect, viewSize: CGSize, imageResolution: CGSize) -> CGRect {
        guard imageResolution.width > 0, imageResolution.height > 0 else { return .null }

        // Mirror the preview's aspect-fill math so the vector overlay stays registered.
        let scale = max(
            viewSize.width / imageResolution.width,
            viewSize.height / imageResolution.height
        )
        let displayedWidth = imageResolution.width * scale
        let displayedHeight = imageResolution.height * scale
        let offsetX = (viewSize.width - displayedWidth) * 0.5
        let offsetY = (viewSize.height - displayedHeight) * 0.5

        return CGRect(
            x: offsetX + bbox.minX * displayedWidth,
            y: offsetY + bbox.minY * displayedHeight,
            width: bbox.width * displayedWidth,
            height: bbox.height * displayedHeight
        ).intersection(CGRect(origin: .zero, size: viewSize))
    }
}
