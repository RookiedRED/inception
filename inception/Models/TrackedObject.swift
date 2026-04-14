//
//  TrackedObject.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/9/26.
//

import Foundation
import simd

/// A detection with a stable track ID maintained across frames.
struct TrackedObject: Identifiable {
    let id: Int                         // stable across frames (survives ReID)
    var detection: Detection            // latest bbox + class info
    var velocity: SIMD2<Float>          // normalized units per frame
    var age: Int                        // total frames since created
    var timeSinceUpdate: Int            // frames since last matched
    var visibility: Visibility
    var depth: Float?                   // meters from camera (LiDAR)
    var worldPosition: simd_float3?     // 3D world coordinates
    var lastSeenTimestamp: TimeInterval

    enum Visibility: String {
        case visible
        case occluded
        case lost
    }
}

extension TrackedObject {
    var trackLabel: String {
        if let d = depth {
            return "#\(id) \(detection.className) \(String(format: "%.1fm", d))"
        }
        return "#\(id) \(detection.className) \(Int(detection.confidence * 100))%"
    }
}
