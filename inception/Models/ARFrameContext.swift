//
//  ARFrameContext.swift
//  iDriveBot
//
//  Clean version for lightweight AR frame metadata
//

import Foundation
import simd
import CoreGraphics

/// Lightweight metadata extracted from an ARFrame.
/// Important:
/// - Do NOT store ARFrame itself here
struct ARFrameContext {
    struct SceneDepthData {
        let values: [Float32]
        let resolution: CGSize
        let intrinsics: simd_float3x3

        func depthAt(x: Int, y: Int) -> Float32? {
            let width = Int(resolution.width)
            let height = Int(resolution.height)
            guard width > 0, height > 0 else { return nil }
            guard x >= 0, x < width, y >= 0, y < height else { return nil }

            let depth = values[y * width + x]
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
