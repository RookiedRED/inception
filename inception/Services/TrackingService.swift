//
//  TrackingService.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/9/26.
//

@preconcurrency import Vision
import UIKit
import simd
import CoreVideo

/// Persistent-ID oriented tracker for AR scanning.
/// Goal:
/// - keep same object ID across temporary disappearance / revisits
/// - only render currently visible objects
/// - store long-lived object memory in world space
final class TrackingService {

    // MARK: - Configuration

    private let highConfThreshold: Float = 0.5
    private let iouMatchThreshold: CGFloat = 0.25
    private let lowIoUMatchThreshold: CGFloat = 0.45

    /// How many predict-only frames an active track may survive
    private let maxAge = 6

    /// Tracks with timeSinceUpdate == 0 are visible.
    /// Others are kept internally for short-term reassociation only.
    private let minHitsToShow = 2
    private let maxPredictedFramesToRender = 1

    /// Slightly relaxed bounds for deciding whether a predicted track is still plausible
    private let visibleBounds = CGRect(x: -0.12, y: -0.12, width: 1.24, height: 1.24)

    // MARK: - State

    private var tracks: [Track] = []
    private let persistentMemory = PersistentObjectMemory()

    /// Serialize all mutable state access to avoid race conditions
    private let queue = DispatchQueue(label: "tracking.service.queue")

    private struct Track {
        var id: Int                          // persistent object ID
        var kalman: KalmanBoxTracker
        var detection: Detection
        var age: Int = 0
        var hits: Int = 1
        var timeSinceUpdate: Int = 0

        var depth: Float?
        var worldPosition: simd_float3?
        var lastSeenTimestamp: TimeInterval = 0

        /// Final box for display
        var displayBBox: CGRect?
    }

    // MARK: - Public API

    func update(detections: [Detection], context: ARFrameContext) -> [TrackedObject] {
        queue.sync {
            _update(detections: detections, context: context)
        }
    }

    func predict(context: ARFrameContext) -> [TrackedObject] {
        queue.sync {
            _predict(context: context)
        }
    }

    // MARK: - Internal update

    private func _update(detections: [Detection], context: ARFrameContext) -> [TrackedObject] {
        // 1) Predict all existing tracks forward
        for i in tracks.indices {
            tracks[i].kalman.predict()
            tracks[i].age += 1
            tracks[i].timeSinceUpdate += 1

            let filteredBox = tracks[i].kalman.predictedBBox
            let displayBox = smoothDisplayBox(previous: tracks[i].displayBBox, current: filteredBox, alpha: 0.18)
            tracks[i].displayBBox = displayBox

            tracks[i].detection = Detection(
                bbox: displayBox,
                classId: tracks[i].detection.classId,
                className: tracks[i].detection.className,
                confidence: tracks[i].detection.confidence,
                maskCoeffs: tracks[i].detection.maskCoeffs
            )
        }

        // 2) Split detections by confidence
        let highDets = detections.filter { $0.confidence >= highConfThreshold }
        let lowDets = detections.filter { $0.confidence < highConfThreshold }

        // 3) High-confidence association
        let allTrackIndices = Array(tracks.indices)
        let (matched1, unmatchedTrackIdx1, unmatchedDetIdx1) =
            hungarianMatch(
                trackIndices: allTrackIndices,
                detections: highDets,
                iouThreshold: iouMatchThreshold
            )

        for (ti, di) in matched1 {
            applyMatch(trackIndex: ti, detection: highDets[di], context: context)
        }

        // 4) Low-confidence association for remaining tracks
        let (matched2, unmatchedTrackIdx2, _) =
            hungarianMatch(
                trackIndices: unmatchedTrackIdx1,
                detections: lowDets,
                iouThreshold: lowIoUMatchThreshold
            )

        for (ti, di) in matched2 {
            applyMatch(trackIndex: ti, detection: lowDets[di], context: context)
        }

        // 5) Reassociate unmatched high-confidence detections with recently unmatched tracks
        let reassoc = reassociateRecentlyUnmatchedTracks(
            unmatchedTrackIdx: unmatchedTrackIdx2,
            unmatchedDetIdx: unmatchedDetIdx1,
            detections: highDets,
            context: context
        )

        for (ti, di) in reassoc.matched {
            applyMatch(trackIndex: ti, detection: highDets[di], context: context)
        }

        // 6) Remaining unmatched high-confidence detections become / update persistent tracks
        for di in reassoc.remainingDetIdx {
            createTrack(from: highDets[di], context: context)
        }

        // 7) Prune dead or implausible tracks
        pruneDeadTracks()

        // 8) Update persistent memory visibility
        syncPersistentVisibility(timestamp: context.timestamp)

        return buildOutput()
    }

    // MARK: - Internal predict

    private func _predict(context: ARFrameContext) -> [TrackedObject] {
        // No Vision bbox propagation here.
        // Keep predict-only path simple and stable to avoid old boxes drifting visibly.
        for i in tracks.indices {
            tracks[i].kalman.predict()
            tracks[i].age += 1
            tracks[i].timeSinceUpdate += 1

            let filteredBox = tracks[i].kalman.predictedBBox
            let displayBox = smoothDisplayBox(previous: tracks[i].displayBBox, current: filteredBox, alpha: 0.18)
            tracks[i].displayBBox = displayBox

            tracks[i].detection = Detection(
                bbox: displayBox,
                classId: tracks[i].detection.classId,
                className: tracks[i].detection.className,
                confidence: tracks[i].detection.confidence,
                maskCoeffs: tracks[i].detection.maskCoeffs
            )
        }

        pruneDeadTracks()
        syncPersistentVisibility(timestamp: context.timestamp)

        return buildOutput()
    }

    // MARK: - Matching

    private func hungarianMatch(
        trackIndices: [Int],
        detections: [Detection],
        iouThreshold: CGFloat
    ) -> (matched: [(trackIdx: Int, detIdx: Int)], unmatchedTracks: [Int], unmatchedDets: [Int]) {

        guard !trackIndices.isEmpty, !detections.isEmpty else {
            return ([], trackIndices, Array(detections.indices))
        }

        let gateCost: Float = 1.0 + 1e-5
        var costMatrix = [[Float]](
            repeating: [Float](repeating: gateCost, count: detections.count),
            count: trackIndices.count
        )

        for (ri, ti) in trackIndices.enumerated() {
            let track = tracks[ti]
            let trackBox = track.kalman.predictedBBox

            for (ci, det) in detections.enumerated() {
                // Class mismatch -> do not match
                guard track.detection.className == det.className else { continue }

                let score = iou(trackBox, det.bbox)
                if score > iouThreshold {
                    costMatrix[ri][ci] = 1.0 - Float(score)
                }
            }
        }

        let assignments = HungarianSolver.solve(costs: costMatrix)

        var matched: [(Int, Int)] = []
        var matchedTrackRows = Set<Int>()
        var matchedDetCols = Set<Int>()

        for (row, col) in assignments {
            if costMatrix[row][col] < gateCost {
                matched.append((trackIndices[row], col))
                matchedTrackRows.insert(row)
                matchedDetCols.insert(col)
            }
        }

        let unmatchedTracks = trackIndices.enumerated()
            .filter { !matchedTrackRows.contains($0.offset) }
            .map(\.element)

        let unmatchedDets = detections.indices.filter { !matchedDetCols.contains($0) }

        return (matched, unmatchedTracks, unmatchedDets)
    }

    private func reassociateRecentlyUnmatchedTracks(
        unmatchedTrackIdx: [Int],
        unmatchedDetIdx: [Int],
        detections: [Detection],
        context: ARFrameContext
    ) -> (matched: [(trackIdx: Int, detIdx: Int)], remainingTrackIdx: [Int], remainingDetIdx: [Int]) {

        guard !unmatchedTrackIdx.isEmpty, !unmatchedDetIdx.isEmpty else {
            return ([], unmatchedTrackIdx, unmatchedDetIdx)
        }

        var matched: [(trackIdx: Int, detIdx: Int)] = []
        var usedTracks = Set<Int>()
        var usedDets = Set<Int>()

        for di in unmatchedDetIdx {
            let det = detections[di]
            let detWorld = estimateWorldPosition(for: det, context: context)

            var bestTrack: Int?
            var bestScore: CGFloat = .greatestFiniteMagnitude

            for ti in unmatchedTrackIdx {
                if usedTracks.contains(ti) { continue }

                let track = tracks[ti]

                // Recently unmatched only
                guard track.timeSinceUpdate > 0, track.timeSinceUpdate <= 2 else { continue }

                // Same semantic class
                guard track.detection.className == det.className else { continue }

                let predicted = track.kalman.predictedBBox
                let iouScore = iou(predicted, det.bbox)
                guard iouScore > 0.08 else { continue }

                let detArea = max(det.bbox.width * det.bbox.height, 0.0001)
                let trackArea = max(predicted.width * predicted.height, 0.0001)
                let areaRatio = max(detArea / trackArea, trackArea / detArea)
                guard areaRatio < 2.0 else { continue }

                var worldPenalty: CGFloat = 0
                if let p0 = detWorld, let p1 = track.worldPosition {
                    let dist = CGFloat(simd_distance(p0, p1))
                    guard dist < 1.2 else { continue }
                    worldPenalty = dist * 0.4
                }

                let score = (1.0 - iouScore) + worldPenalty + CGFloat(track.timeSinceUpdate) * 0.1
                if score < bestScore {
                    bestScore = score
                    bestTrack = ti
                }
            }

            if let ti = bestTrack {
                matched.append((ti, di))
                usedTracks.insert(ti)
                usedDets.insert(di)
            }
        }

        let remainingTracks = unmatchedTrackIdx.filter { !usedTracks.contains($0) }
        let remainingDets = unmatchedDetIdx.filter { !usedDets.contains($0) }

        return (matched, remainingTracks, remainingDets)
    }

    // MARK: - Track lifecycle

    private func applyMatch(trackIndex ti: Int, detection: Detection, context: ARFrameContext) {
        let stabilized = stabilizedBox(
            measured: detection.bbox,
            reference: tracks[ti].displayBBox ?? tracks[ti].kalman.predictedBBox
        )

        tracks[ti].kalman.update(bbox: stabilized)

        let filteredBox = tracks[ti].kalman.predictedBBox
        let displayBox = smoothDisplayBox(previous: tracks[ti].displayBBox, current: filteredBox, alpha: 0.18)
        tracks[ti].displayBBox = displayBox

        tracks[ti].detection = Detection(
            bbox: displayBox,
            classId: detection.classId,
            className: detection.className,
            confidence: detection.confidence,
            maskCoeffs: detection.maskCoeffs
        )

        tracks[ti].timeSinceUpdate = 0
        tracks[ti].hits += 1

        let depth = sampleDepth(for: tracks[ti].detection, context: context)
        let worldPosition = estimateWorldPosition(for: tracks[ti].detection, context: context)

        tracks[ti].depth = depth
        tracks[ti].worldPosition = worldPosition
        tracks[ti].lastSeenTimestamp = context.timestamp

        if let worldPosition {
            persistentMemory.updateVisible(
                id: tracks[ti].id,
                worldPosition: worldPosition,
                bbox: displayBox,
                depth: depth,
                timestamp: context.timestamp
            )
        }
    }

    private func createTrack(from detection: Detection, context: ARFrameContext) {
        guard let worldPosition = estimateWorldPosition(for: detection, context: context) else { return }

        let depth = sampleDepth(for: detection, context: context)

        let persistentId = persistentMemory.associate(
            className: detection.className,
            worldPosition: worldPosition,
            bbox: detection.bbox,
            depth: depth,
            timestamp: context.timestamp
        )

        // If same persistent object already has an active track, update it instead of creating a duplicate
        if let existingIndex = tracks.firstIndex(where: { $0.id == persistentId }) {
            applyMatch(trackIndex: existingIndex, detection: detection, context: context)
            return
        }

        var track = Track(
            id: persistentId,
            kalman: KalmanBoxTracker(bbox: detection.bbox),
            detection: detection
        )

        let filteredBox = track.kalman.predictedBBox
        let displayBox = smoothDisplayBox(previous: nil, current: filteredBox, alpha: 0.18)

        track.displayBBox = displayBox
        track.detection = Detection(
            bbox: displayBox,
            classId: detection.classId,
            className: detection.className,
            confidence: detection.confidence,
            maskCoeffs: detection.maskCoeffs
        )
        track.depth = depth
        track.worldPosition = worldPosition
        track.lastSeenTimestamp = context.timestamp

        tracks.append(track)

        persistentMemory.updateVisible(
            id: persistentId,
            worldPosition: worldPosition,
            bbox: displayBox,
            depth: depth,
            timestamp: context.timestamp
        )
    }

    private func pruneDeadTracks() {
        tracks.removeAll { track in
            if track.timeSinceUpdate > maxAge { return true }

            let box = track.kalman.predictedBBox
            let center = CGPoint(x: box.midX, y: box.midY)

            // Remove quickly once it leaves the plausible view region
            if !visibleBounds.contains(center) && track.timeSinceUpdate > 1 {
                return true
            }

            // Remove degenerate / exploded boxes
            if box.width < 0.01 || box.height < 0.01 || box.width > 1.2 || box.height > 1.2 {
                return true
            }

            return false
        }
    }

    private func syncPersistentVisibility(timestamp: TimeInterval) {
        let visibleIds = Set(tracks.filter { $0.timeSinceUpdate <= maxPredictedFramesToRender }.map(\.id))
        let allIds = Set(persistentMemory.records.keys)
        let invisibleIds = allIds.subtracting(visibleIds)
        persistentMemory.markInvisible(idsNotSeen: invisibleIds, timestamp: timestamp)
    }

    // MARK: - Output

    private func buildOutput() -> [TrackedObject] {
        tracks.compactMap { track in
            guard track.hits >= minHitsToShow else { return nil }
            guard track.timeSinceUpdate <= maxPredictedFramesToRender else { return nil }

            let visibility: TrackedObject.Visibility =
                track.timeSinceUpdate == 0 ? .visible : .occluded

            return TrackedObject(
                id: track.id,
                detection: track.detection,
                velocity: track.kalman.velocity,
                age: track.age,
                timeSinceUpdate: track.timeSinceUpdate,
                visibility: visibility,
                depth: track.depth,
                worldPosition: track.worldPosition,
                lastSeenTimestamp: track.lastSeenTimestamp
            )
        }
    }

    // MARK: - Helpers

    private func stabilizedBox(measured: CGRect, reference: CGRect) -> CGRect {
        let refW = max(reference.width, 0.0001)
        let refH = max(reference.height, 0.0001)

        let minW = refW * 0.85
        let maxW = refW * 1.15
        let minH = refH * 0.85
        let maxH = refH * 1.15

        let clampedW = min(max(measured.width, minW), maxW)
        let clampedH = min(max(measured.height, minH), maxH)

        let cx = measured.midX
        let cy = measured.midY

        return CGRect(
            x: cx - clampedW / 2,
            y: cy - clampedH / 2,
            width: clampedW,
            height: clampedH
        )
    }

    private func smoothDisplayBox(previous: CGRect?, current: CGRect, alpha: CGFloat) -> CGRect {
        guard let previous else { return current }

        let cx = previous.midX * (1 - alpha) + current.midX * alpha
        let cy = previous.midY * (1 - alpha) + current.midY * alpha
        let w  = previous.width * (1 - alpha) + current.width * alpha
        let h  = previous.height * (1 - alpha) + current.height * alpha

        return CGRect(
            x: cx - w / 2,
            y: cy - h / 2,
            width: w,
            height: h
        )
    }

    private func estimateWorldPosition(for detection: Detection, context: ARFrameContext) -> simd_float3? {
        let center = CGPoint(x: detection.bbox.midX, y: detection.bbox.midY)

        // If no true depth is available in the current pipeline, use a fixed forward distance.
        // This still gives a coarse world anchor across the same AR session.
        let depth = sampleDepth(for: detection, context: context) ?? 2.0

        let fx = context.intrinsics[0][0]
        let fy = context.intrinsics[1][1]
        let cx = context.intrinsics[2][0]
        let cy = context.intrinsics[2][1]

        let px = Float(center.x) * Float(context.imageResolution.width)
        let py = Float(center.y) * Float(context.imageResolution.height)

        let camX = (px - cx) / fx * depth
        let camY = (py - cy) / fy * depth
        let camZ = -depth

        let camPoint = simd_float4(camX, camY, camZ, 1)
        let worldPoint = context.cameraTransform * camPoint

        return simd_float3(worldPoint.x, worldPoint.y, worldPoint.z)
    }

    private func sampleDepth(for detection: Detection, context: ARFrameContext) -> Float? {
        // This version intentionally does not depend on context.depthMap
        // so it can compile with the simplified ARFrameContext.
        return nil
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return unionArea > 0 ? interArea / unionArea : 0
    }
}
