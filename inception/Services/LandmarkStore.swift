//
//  LandmarkStore.swift
//  inception
//
//  Remembers static objects across frames.
//  Dynamic objects (people, vehicles, animals) are intentionally excluded.
//

import Foundation
import simd

@MainActor
final class LandmarkStore {

    // MARK: - Config

    /// Classes that can move autonomously — never saved as landmarks.
    static let dynamicClasses: Set<String> = [
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat",
        "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe"
    ]

    /// Two detections of the same class within this radius (metres, XZ only)
    /// are considered the same landmark and merged.
    /// Set to 1.5m because LiDAR depth error at 3-5m is typically ±0.5-1m,
    /// so the same object seen from different angles can project 1-1.5m apart.
    private let mergeRadius: Float = 1.5

    /// Minimum detection confidence required to save or update a landmark.
    private let minimumConfidence: Float = 0.35

    /// A landmark must be observed at least this many times before appearing on the map.
    /// Filters out spurious single-frame detections.
    private let minimumObservations: Int = 2

    /// Hard cap to prevent unbounded memory growth.
    private let maximumLandmarks: Int = 120

    // MARK: - Storage

    private var landmarks: [UUID: Landmark] = [:]

    /// Only returns landmarks that have been confirmed by enough observations.
    var all: [Landmark] {
        landmarks.values.filter { $0.observationCount >= minimumObservations }
    }

    // MARK: - Public API

    /// Call after every inference pass with the current frame's tracked objects.
    func update(with objects: [TrackedObject]) {
        for obj in objects {
            guard let pos = obj.worldPosition else { continue }
            guard obj.detection.confidence >= minimumConfidence else { continue }
            guard !Self.dynamicClasses.contains(obj.detection.className) else { continue }

            if let existingID = closestLandmarkID(to: pos, className: obj.detection.className) {
                // Merge: EWMA-nudge position toward the new observation.
                guard var existing = landmarks[existingID] else { continue }
                let alpha: Float = 0.15
                existing.worldPosition = simd_mix(existing.worldPosition, pos, SIMD3<Float>(repeating: alpha))
                existing.confidence = max(existing.confidence, obj.detection.confidence)
                existing.lastSeenTimestamp = obj.lastSeenTimestamp
                existing.observationCount += 1
                landmarks[existingID] = existing
            } else {
                guard landmarks.count < maximumLandmarks else { continue }
                let landmark = Landmark(
                    id: UUID(),
                    className: obj.detection.className,
                    worldPosition: pos,
                    confidence: obj.detection.confidence,
                    lastSeenTimestamp: obj.lastSeenTimestamp,
                    observationCount: 1
                )
                landmarks[landmark.id] = landmark
            }
        }

        // After processing the frame, merge any landmarks that drifted into each other.
        consolidate()
    }

    func clear() {
        landmarks.removeAll()
    }

    // MARK: - Nearest-neighbour lookup

    /// Returns the ID of the CLOSEST existing landmark of the same class within mergeRadius.
    /// Using the closest (not arbitrary first) reduces false non-merges when multiple
    /// candidates exist at different distances.
    private func closestLandmarkID(to position: simd_float3, className: String) -> UUID? {
        let flat = SIMD2<Float>(position.x, position.z)
        var bestID: UUID?
        var bestDist: Float = mergeRadius  // only accept within radius

        for (id, lm) in landmarks {
            guard lm.className == className else { continue }
            let lmFlat = SIMD2<Float>(lm.worldPosition.x, lm.worldPosition.z)
            let dist = simd_distance(flat, lmFlat)
            if dist < bestDist {
                bestDist = dist
                bestID = id
            }
        }
        return bestID
    }

    // MARK: - Consolidation

    /// After each update batch, sweep all same-class landmark pairs and merge any that
    /// have drifted within mergeRadius of each other.
    /// This handles the edge case where two separately-created landmarks later converge
    /// as their EWMA positions are updated toward the true object location.
    private func consolidate() {
        var ids = Array(landmarks.keys)
        var i = 0
        while i < ids.count {
            guard let a = landmarks[ids[i]] else { i += 1; continue }
            let aFlat = SIMD2<Float>(a.worldPosition.x, a.worldPosition.z)
            var j = i + 1
            while j < ids.count {
                guard let b = landmarks[ids[j]] else { j += 1; continue }
                guard a.className == b.className else { j += 1; continue }

                let bFlat = SIMD2<Float>(b.worldPosition.x, b.worldPosition.z)
                if simd_distance(aFlat, bFlat) < mergeRadius {
                    // Merge b into a: average position, take best confidence and count
                    var merged = a
                    merged.worldPosition = (a.worldPosition + b.worldPosition) * 0.5
                    merged.confidence = max(a.confidence, b.confidence)
                    merged.lastSeenTimestamp = max(a.lastSeenTimestamp, b.lastSeenTimestamp)
                    merged.observationCount = a.observationCount + b.observationCount
                    landmarks[ids[i]] = merged

                    landmarks.removeValue(forKey: ids[j])
                    ids.remove(at: j)
                    // Don't increment j — next element shifted into this index
                } else {
                    j += 1
                }
            }
            i += 1
        }
    }
}
