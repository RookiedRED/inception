//
//  Landmark.swift
//  inception
//
//  A persistent marker for a static object remembered across frames.
//

import Foundation
import simd

struct Landmark: Identifiable {
    let id: UUID
    let className: String
    /// World-space XZ position (Y is ignored for minimap rendering).
    var worldPosition: simd_float3
    var confidence: Float
    var lastSeenTimestamp: TimeInterval
    /// How many times this landmark has been observed.
    /// Only landmarks above a minimum threshold are shown on the map.
    var observationCount: Int
}
