//
//  PersistentObjectMemory.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/11/26.
//

import Foundation
import simd
import CoreGraphics

final class PersistentObjectMemory {

    struct Record: Identifiable {
        let id: Int
        var className: String
        var worldPosition: simd_float3
        var lastSeenTimestamp: TimeInterval
        var observationCount: Int
        var lastBBox: CGRect
        var lastDepth: Float?
        var isVisible: Bool
    }

    private(set) var records: [Int: Record] = [:]
    private var nextId: Int = 1

    // 同一個靜態物件（像電視、沙發）在同一 session 裡的匹配半徑
    private let sameObjectRadius: Float = 0.8
    private let maxDepthDiff: Float = 1.0
    private let maxAreaRatio: CGFloat = 2.5
    private let staleAge: TimeInterval = 600   // 10 分鐘，可依需求調

    func associate(
        className: String,
        worldPosition: simd_float3,
        bbox: CGRect,
        depth: Float?,
        timestamp: TimeInterval
    ) -> Int {
        pruneStale(now: timestamp)

        let bboxArea = max(bbox.width * bbox.height, 1e-4)

        var bestId: Int?
        var bestScore: Float = .greatestFiniteMagnitude

        for (id, record) in records {
            guard record.className == className else { continue }

            let dist = simd_distance(worldPosition, record.worldPosition)
            guard dist < sameObjectRadius else { continue }

            if let d0 = depth, let d1 = record.lastDepth {
                guard abs(d0 - d1) < maxDepthDiff else { continue }
            }

            let oldArea = max(record.lastBBox.width * record.lastBBox.height, 1e-4)
            let ratio = max(bboxArea / oldArea, oldArea / bboxArea)
            guard ratio < maxAreaRatio else { continue }

            let depthPenalty: Float
            if let d0 = depth, let d1 = record.lastDepth {
                depthPenalty = abs(d0 - d1) * 0.2
            } else {
                depthPenalty = 0
            }

            let sizePenalty = Float(min(ratio - 1.0, 2.0)) * 0.15
            let score = dist + depthPenalty + sizePenalty

            if score < bestScore {
                bestScore = score
                bestId = id
            }
        }

        if let id = bestId {
            updateRecord(
                id: id,
                className: className,
                worldPosition: worldPosition,
                bbox: bbox,
                depth: depth,
                timestamp: timestamp,
                visible: true
            )
            return id
        } else {
            let id = nextId
            nextId += 1
            records[id] = Record(
                id: id,
                className: className,
                worldPosition: worldPosition,
                lastSeenTimestamp: timestamp,
                observationCount: 1,
                lastBBox: bbox,
                lastDepth: depth,
                isVisible: true
            )
            return id
        }
    }

    func markInvisible(idsNotSeen: Set<Int>, timestamp: TimeInterval) {
        for id in idsNotSeen {
            guard var record = records[id] else { continue }
            record.isVisible = false
            record.lastSeenTimestamp = timestamp
            records[id] = record
        }
    }

    func updateVisible(
        id: Int,
        worldPosition: simd_float3,
        bbox: CGRect,
        depth: Float?,
        timestamp: TimeInterval
    ) {
        guard let existing = records[id] else { return }
        updateRecord(
            id: id,
            className: existing.className,
            worldPosition: worldPosition,
            bbox: bbox,
            depth: depth,
            timestamp: timestamp,
            visible: true
        )
    }

    func visibleRecords() -> [Record] {
        records.values.filter { $0.isVisible }
    }

    private func updateRecord(
        id: Int,
        className: String,
        worldPosition: simd_float3,
        bbox: CGRect,
        depth: Float?,
        timestamp: TimeInterval,
        visible: Bool
    ) {
        guard var record = records[id] else { return }

        // 用 EMA 讓世界座標更穩
        let alpha: Float = 0.25
        record.worldPosition = simd_mix(record.worldPosition, worldPosition, simd_float3(repeating: alpha))
        record.lastBBox = smoothBBox(old: record.lastBBox, new: bbox, alpha: 0.25)
        record.lastDepth = depth ?? record.lastDepth
        record.lastSeenTimestamp = timestamp
        record.observationCount += 1
        record.isVisible = visible

        records[id] = record
    }

    private func smoothBBox(old: CGRect, new: CGRect, alpha: CGFloat) -> CGRect {
        let cx = old.midX * (1 - alpha) + new.midX * alpha
        let cy = old.midY * (1 - alpha) + new.midY * alpha
        let w = old.width * (1 - alpha) + new.width * alpha
        let h = old.height * (1 - alpha) + new.height * alpha

        return CGRect(
            x: cx - w / 2,
            y: cy - h / 2,
            width: w,
            height: h
        )
    }

    private func pruneStale(now: TimeInterval) {
        records = records.filter { _, record in
            now - record.lastSeenTimestamp < staleAge
        }
    }
}
