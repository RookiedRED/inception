//
//  DetectionOverlayView.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/8/26.
//

import SwiftUI

struct DetectionOverlayView: View {
    let image: UIImage?
    let trackedObjects: [TrackedObject]
    let orientation: AppOrientation

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                // 框 + 填色 overlay（拉伸對齊相機畫面）
                if let image {
                    Image(uiImage: image)
                        .resizable()
                }

                // 文字標籤（SwiftUI 渲染，不會被拉伸）
                ForEach(trackedObjects) { obj in
                    let x = obj.detection.bbox.minX * geo.size.width
                    let y = obj.detection.bbox.minY * geo.size.height

                    Text(obj.trackLabel)
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundColor(.white)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Color(obj.detection.color).opacity(
                            obj.visibility == .visible ? 0.75 : 0.4
                        ))
                        .rotationEffect(.degrees(orientation.isFlipped ? 180 : 0))
                        .position(x: x + 40, y: y + 10)
                }
            }
        }
        .allowsHitTesting(false)
    }
}
