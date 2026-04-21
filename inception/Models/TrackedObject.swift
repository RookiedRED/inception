//
//  TrackedObject.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/9/26.
//

import Foundation
import simd

/// Lightweight display model used by the overlay and minimap.
/// In the current detection-only pipeline, IDs are recreated every inference pass.
struct TrackedObject: Identifiable {
    let id: Int
    var detection: Detection
    var velocity: SIMD2<Float>
    var age: Int
    var timeSinceUpdate: Int
    var visibility: Visibility
    var depth: Float?
    var worldPosition: simd_float3?
    var lastSeenTimestamp: TimeInterval

    enum Visibility: String {
        case visible
        case occluded
        case lost
    }
}

extension TrackedObject {
    /// Overlay/minimap label prioritizing depth when scene depth is available.
    var trackLabel: String {
        if let d = depth {
            return "#\(id) \(detection.className) \(String(format: "%.1fm", d))"
        }
        return "#\(id) \(detection.className) \(Int(detection.confidence * 100))%"
    }
}
