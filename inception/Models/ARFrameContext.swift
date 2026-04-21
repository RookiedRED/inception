//
//  ARFrameContext.swift
//  iDriveBot
//
//  Clean version for lightweight AR frame metadata
//

import Foundation
import simd
import CoreGraphics
import CoreVideo

/// Immutable frame metadata extracted from `ARFrame`.
/// The struct intentionally avoids storing `ARFrame` itself so frame lifetime stays short.
struct ARFrameContext {
    /// Retained scene-depth payload used for world-position reconstruction.
    struct SceneDepthData: @unchecked Sendable {
        /// Direct reference to ARKit's depth `CVPixelBuffer`.
        /// Access sites are responsible for lock/unlock synchronization.
        let depthMap: CVPixelBuffer
        let resolution: CGSize
        let intrinsics: simd_float3x3

        /// Reads a single depth sample in depth-map coordinates.
        func depthAt(x: Int, y: Int) -> Float32? {
            let width = Int(resolution.width)
            let height = Int(resolution.height)
            guard width > 0, height > 0, x >= 0, x < width, y >= 0, y < height else { return nil }
            CVPixelBufferLockBaseAddress(depthMap, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
            let floatsPerRow = CVPixelBufferGetBytesPerRow(depthMap) / MemoryLayout<Float32>.stride
            let depth = base.assumingMemoryBound(to: Float32.self)[y * floatsPerRow + x]
            guard depth.isFinite, depth > 0 else { return nil }
            return depth
        }
    }

    /// Camera-to-world transform
    let cameraTransform: simd_float4x4

    /// Camera intrinsics (fx, fy, cx, cy)
    let intrinsics: simd_float3x3

    /// Pixel resolution of the captured image
    let imageResolution: CGSize

    /// Optional copied scene depth for metric world reconstruction.
    let sceneDepth: SceneDepthData?

    /// Frame timestamp from ARKit
    let timestamp: TimeInterval
}
