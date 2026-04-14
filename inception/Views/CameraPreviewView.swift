import SwiftUI

/// Lightweight camera preview — just displays a UIImage.
/// No ARSCNView, no SceneKit, no internal ARFrame retention.
struct CameraPreviewView: View {
    let image: UIImage?

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                Color.black

                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: proxy.size.width, height: proxy.size.height)
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
            .clipped()
        }
        .ignoresSafeArea()
    }
}
