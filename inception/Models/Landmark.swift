//
//  Landmark.swift
//  inception
//
//  A persistent marker for a static object remembered across frames.
//

import Foundation
import simd

/// Persistent world-space marker created from repeated observations of a static object.
struct Landmark: Identifiable {
    let id: UUID
    let className: String
    /// World-space position used by the minimap and route-planning features.
    var worldPosition: simd_float3
    var confidence: Float
    var lastSeenTimestamp: TimeInterval
    /// Number of supporting detections accumulated for this landmark.
    var observationCount: Int
}
