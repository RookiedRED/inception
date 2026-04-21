import SwiftUI
import CoreImage
import CoreVideo
import MetalKit

/// SwiftUI wrapper around the Metal-based live camera preview surface.
struct CameraPreviewView: UIViewRepresentable {
    let source: CameraPreviewSource

    func makeUIView(context: Context) -> PreviewMetalView {
        let view = PreviewMetalView()
        view.backgroundColor = .black
        source.attach(view)
        return view
    }

    func updateUIView(_ uiView: PreviewMetalView, context: Context) {
        source.attach(uiView)
    }
}

@MainActor
/// Bridges the latest camera frame from the view model into the preview view.
final class CameraPreviewSource {
    private weak var view: PreviewMetalView?
    private var latestPixelBuffer: CVPixelBuffer?

    func attach(_ view: PreviewMetalView) {
        self.view = view
        if let latestPixelBuffer {
            view.setPixelBuffer(latestPixelBuffer)
        }
    }

    func publish(_ pixelBuffer: CVPixelBuffer) {
        latestPixelBuffer = pixelBuffer
        view?.setPixelBuffer(pixelBuffer)
    }
}

/// Renders camera frames with Core Image directly into a Metal drawable.
final class PreviewMetalView: MTKView {
    private var pixelBuffer: CVPixelBuffer?

    private let ciContext: CIContext?
    private let colorSpace = CGColorSpaceCreateDeviceRGB()

    init() {
        let device = MTLCreateSystemDefaultDevice()
        self.ciContext = device.map { CIContext(mtlDevice: $0) }
        super.init(frame: .zero, device: device)

        framebufferOnly = false
        isOpaque = true
        enableSetNeedsDisplay = false
        isPaused = false
        autoResizeDrawable = true
        contentMode = .scaleToFill
        preferredFramesPerSecond = 60
    }

    @available(*, unavailable)
    required init(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        let scale = window?.windowScene?.screen.scale ?? traitCollection.displayScale
        drawableSize = CGSize(width: bounds.width * scale, height: bounds.height * scale)
    }

    override func draw(_ rect: CGRect) {
        guard let pixelBuffer,
              let ciContext,
              let currentDrawable else { return }

        // Mirror the overlay's aspect-fill transform so 2D boxes stay registered.
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let sourceRect = image.extent
        let targetRect = CGRect(origin: .zero, size: drawableSize)
        let scale = max(
            targetRect.width / sourceRect.width,
            targetRect.height / sourceRect.height
        )
        let scaledWidth = sourceRect.width * scale
        let scaledHeight = sourceRect.height * scale
        let x = (targetRect.width - scaledWidth) * 0.5
        let y = (targetRect.height - scaledHeight) * 0.5
        let transformed = image.transformed(
            by: CGAffineTransform(scaleX: scale, y: scale)
                .concatenating(CGAffineTransform(translationX: x, y: y))
        )

        ciContext.render(
            transformed,
            to: currentDrawable.texture,
            commandBuffer: nil,
            bounds: targetRect,
            colorSpace: colorSpace
        )

        currentDrawable.present()
    }

    func setPixelBuffer(_ pixelBuffer: CVPixelBuffer) {
        self.pixelBuffer = pixelBuffer
    }
}
