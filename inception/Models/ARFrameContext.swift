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
/// - Do NOT store depthMap here
///   (to avoid retaining heavy frame-backed resources)
struct ARFrameContext {
    /// Camera-to-world transform
    let cameraTransform: simd_float4x4

    /// Camera intrinsics (fx, fy, cx, cy)
    let intrinsics: simd_float3x3

    /// Pixel resolution of the captured image
    let imageResolution: CGSize

    /// Frame timestamp from ARKit
    let timestamp: TimeInterval
}
