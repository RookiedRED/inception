//
//  NavigationService.swift
//  inception
//
//  2D occupancy map + A* pathfinder for AR navigation.
//  Builds a sparse grid from AR mesh vertical faces and finds
//  navigable paths between two world-space XZ positions.
//

import Foundation
import ARKit
import simd

final class NavigationService {

    // MARK: - Config

    /// World-space metres per grid cell.
    private let cellSize: Float = 0.25
    /// Cells of clearance around each obstacle (~0.5 m).
    private let dilationCells: Int = 2
    /// Safety cap to keep pathfinding interactive.
    private let maxAStarIterations = 3000
    /// Sample only 1-in-N faces per anchor to keep rebuild fast.
    private let faceStepDivisor = 800

    // MARK: - State

    private var dilatedCells: Set<GridCell> = []
    private let routeQueue = DispatchQueue(label: "inception.navigation.route", qos: .userInitiated)

    // MARK: - Types

    fileprivate struct GridCell: Hashable {
        let x: Int
        let z: Int
    }

    // MARK: - Public API

    /// Rebuild the occupancy map from AR scene mesh anchors.
    /// Only vertical faces (walls, furniture sides) are treated as obstacles.
    /// Safe to call on the main thread; uses stride sampling to stay fast.
    func rebuildOccupancy(from anchors: [ARMeshAnchor]) {
        var raw: Set<GridCell> = []

        for anchor in anchors {
            let geo   = anchor.geometry
            let verts = geo.vertices
            let faces = geo.faces
            let faceCount = faces.count
            let bpi = faces.bytesPerIndex          // bytes per index (2 or 4)
            let ipf = faces.indexCountPerPrimitive // should be 3
            guard ipf == 3, faceCount > 0 else { continue }

            let vBuf = verts.buffer.contents()
            let fBuf = faces.buffer.contents()
            let faceStep = max(1, faceCount / faceStepDivisor)

            for fi in stride(from: 0, to: faceCount, by: faceStep) {
                let base = fBuf.advanced(by: fi * ipf * bpi)

                func readIdx(_ offset: Int) -> UInt32 {
                    let ptr = base.advanced(by: offset * bpi)
                    return bpi == 2
                        ? UInt32(ptr.assumingMemoryBound(to: UInt16.self).pointee)
                        : ptr.assumingMemoryBound(to: UInt32.self).pointee
                }

                func worldVertex(_ i: UInt32) -> simd_float3 {
                    let lp = vBuf
                        .advanced(by: Int(i) * verts.stride)
                        .assumingMemoryBound(to: SIMD3<Float>.self).pointee
                    let w = anchor.transform * simd_float4(lp.x, lp.y, lp.z, 1)
                    return simd_float3(w.x, w.y, w.z)
                }

                let v0 = worldVertex(readIdx(0))
                let v1 = worldVertex(readIdx(1))
                let v2 = worldVertex(readIdx(2))

                // Compute face normal; skip degenerate and horizontal faces
                let cross = simd_cross(v1 - v0, v2 - v0)
                let area  = simd_length(cross)
                guard area > 1e-6 else { continue }
                let normal = cross / area
                // abs(dot(n, up)) < 0.5 → mostly vertical → treat as wall
                guard abs(simd_dot(normal, SIMD3<Float>(0, 1, 0))) < 0.5 else { continue }

                raw.insert(worldToCell((v0 + v1 + v2) / 3))
            }
        }

        // Dilate obstacles for safe clearance
        var dilated = raw
        for c in raw {
            for dx in -dilationCells...dilationCells {
                for dz in -dilationCells...dilationCells {
                    dilated.insert(GridCell(x: c.x + dx, z: c.z + dz))
                }
            }
        }
        dilatedCells = dilated
    }

    /// Find a smoothed path from `start` to `goal` through free space.
    /// Returns world-space XZ waypoints (Y held constant at `start.y`).
    /// Falls back to a two-point straight line if A* cannot find a path.
    func findPath(from start: simd_float3, to goal: simd_float3) -> [simd_float3] {
        let sc = worldToCell(start)
        let gc = worldToCell(goal)
        guard sc != gc else { return [goal] }

        let raw = astar(from: sc, to: gc)
        guard !raw.isEmpty else { return [start, goal] }

        let worldPath = raw.map { cellToWorld($0, y: start.y) }
        return smoothPath(worldPath)
    }

    func calculateRoute(
        from start: simd_float3,
        to goal: simd_float3,
        using anchors: [ARMeshAnchor],
        completion: @escaping ([simd_float3]) -> Void
    ) {
        routeQueue.async { [weak self] in
            guard let self else { return }
            self.rebuildOccupancy(from: anchors)
            let route = self.findPath(from: start, to: goal)
            DispatchQueue.main.async {
                completion(route)
            }
        }
    }

    // MARK: - A*

    private struct AStarNode: Comparable {
        let cell: GridCell
        let f: Float
        static func < (a: Self, b: Self) -> Bool { a.f < b.f }
    }

    private func astar(from start: GridCell, to goal: GridCell) -> [GridCell] {
        var open     = MinHeap<AStarNode>()
        var cameFrom = [GridCell: GridCell]()
        var gScore   = [GridCell: Float]()
        var closed   = Set<GridCell>()

        gScore[start] = 0
        open.insert(AStarNode(cell: start, f: octile(start, goal)))

        var iterations = 0
        while let cur = open.extractMin(), iterations < maxAStarIterations {
            iterations += 1

            if cur.cell == goal {
                return reconstructPath(cameFrom, end: goal)
            }
            if closed.contains(cur.cell) { continue }
            closed.insert(cur.cell)

            for (nb, moveCost) in neighborCosts(of: cur.cell) {
                guard !closed.contains(nb), !dilatedCells.contains(nb) else { continue }
                let tentG = (gScore[cur.cell] ?? .infinity) + moveCost
                if tentG < (gScore[nb] ?? .infinity) {
                    cameFrom[nb] = cur.cell
                    gScore[nb]   = tentG
                    open.insert(AStarNode(cell: nb, f: tentG + octile(nb, goal)))
                }
            }
        }
        return []   // no path found
    }

    private func reconstructPath(_ cameFrom: [GridCell: GridCell], end: GridCell) -> [GridCell] {
        var path = [end]
        var cur  = end
        while let prev = cameFrom[cur] { path.append(prev); cur = prev }
        return path.reversed()
    }

    // MARK: - Path Smoothing (greedy line-of-sight)

    private func smoothPath(_ path: [simd_float3]) -> [simd_float3] {
        guard path.count > 2 else { return path }
        var result = [path[0]]
        var i = 0
        while i < path.count - 1 {
            var farthest = i + 1
            for j in stride(from: path.count - 1, through: i + 2, by: -1) {
                if lineOfSight(from: path[i], to: path[j]) { farthest = j; break }
            }
            result.append(path[farthest])
            i = farthest
        }
        return result
    }

    /// Bresenham rasterisation of the line between two world positions.
    private func lineOfSight(from a: simd_float3, to b: simd_float3) -> Bool {
        var p   = worldToCell(a)
        let end = worldToCell(b)
        let dx = abs(end.x - p.x), dz = abs(end.z - p.z)
        let sx = end.x > p.x ? 1 : -1
        let sz = end.z > p.z ? 1 : -1
        var err = dx - dz
        for _ in 0...(dx + dz + 1) {
            if dilatedCells.contains(p) { return false }
            if p == end { break }
            let e2 = 2 * err
            if e2 > -dz { err -= dz; p = GridCell(x: p.x + sx, z: p.z) }
            if e2 <  dx { err += dx; p = GridCell(x: p.x,       z: p.z + sz) }
        }
        return true
    }

    // MARK: - Grid Helpers

    private func worldToCell(_ pos: simd_float3) -> GridCell {
        GridCell(x: Int(floor(pos.x / cellSize)), z: Int(floor(pos.z / cellSize)))
    }

    private func cellToWorld(_ c: GridCell, y: Float) -> simd_float3 {
        simd_float3(
            Float(c.x) * cellSize + cellSize * 0.5,
            y,
            Float(c.z) * cellSize + cellSize * 0.5
        )
    }

    /// Octile distance heuristic for 8-directional A*.
    private func octile(_ a: GridCell, _ b: GridCell) -> Float {
        let dx = Float(abs(a.x - b.x))
        let dz = Float(abs(a.z - b.z))
        return 1.414 * min(dx, dz) + abs(dx - dz)
    }

    private func neighborCosts(of c: GridCell) -> [(GridCell, Float)] {
        var result = [(GridCell, Float)]()
        result.reserveCapacity(8)
        for dx in -1...1 {
            for dz in -1...1 {
                guard dx != 0 || dz != 0 else { continue }
                let cost: Float = (dx != 0 && dz != 0) ? 1.414 : 1.0
                result.append((GridCell(x: c.x + dx, z: c.z + dz), cost))
            }
        }
        return result
    }
}

// MARK: - Min-Heap (binary heap for A* open set)

private struct MinHeap<T: Comparable> {
    private var data: [T] = []
    var isEmpty: Bool { data.isEmpty }

    mutating func insert(_ value: T) {
        data.append(value)
        siftUp(data.count - 1)
    }

    mutating func extractMin() -> T? {
        guard !data.isEmpty else { return nil }
        if data.count == 1 { return data.removeLast() }
        let top = data[0]
        data[0] = data.removeLast()
        siftDown(0)
        return top
    }

    private mutating func siftUp(_ i: Int) {
        var i = i
        while i > 0 {
            let p = (i - 1) / 2
            guard data[i] < data[p] else { break }
            data.swapAt(i, p)
            i = p
        }
    }

    private mutating func siftDown(_ i: Int) {
        var i = i
        let n = data.count
        while true {
            let l = 2 * i + 1, r = 2 * i + 2
            var s = i
            if l < n && data[l] < data[s] { s = l }
            if r < n && data[r] < data[s] { s = r }
            guard s != i else { break }
            data.swapAt(i, s)
            i = s
        }
    }
}
