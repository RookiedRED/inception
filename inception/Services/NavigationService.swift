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

/// Builds a sparse occupancy grid from AR meshes and computes walkable minimap routes.
final class NavigationService {

    // MARK: - Grid Configuration

    /// World-space metres per grid cell.
    private let cellSize: Float = 0.25
    /// Extra cells of clearance added around each obstacle cell.
    private let dilationCells: Int = 0
    /// Safety cap to keep pathfinding interactive.
    private let maxAStarIterations = 8_000
    /// Sample only 1-in-N faces per anchor to keep rebuild fast.
    private let faceStepDivisor = 800

    // MARK: - Occupancy State

    /// Cells confirmed as obstacle-free floor (upward-facing AR mesh faces).
    /// A* only routes through these — unexplored areas are impassable.
    private var exploredCells: Set<GridCell> = []
    /// Cells marked as obstacles (walls / furniture) including dilation buffer.
    private var dilatedCells: Set<GridCell> = []
    /// Low-QoS queue: pathfinding runs at utility priority so it doesn't
    /// compete with the SceneKit render loop on the main thread.
    private let routeQueue = DispatchQueue(label: "inception.navigation.route", qos: .utility)

    // MARK: - Internal Types

    fileprivate struct GridCell: Hashable {
        let x: Int
        let z: Int
    }

    /// Immutable snapshot of one mesh anchor captured before background processing begins.
    private struct AnchorSnapshot {
        let transform: simd_float4x4
        let vertexData: Data
        let vertexStride: Int
        let faceData: Data
        let faceCount: Int
        let bytesPerIndex: Int
        let indicesPerFace: Int
    }

    // MARK: - Public API

    /// A snapshot of the current occupancy grid for debug visualisation.
    struct OccupancySnapshot {
        /// Cells A* can actually route through: explored floor AND not currently
        /// blocked by an obstacle.  Shown as green in the minimap overlay.
        let exploredCells: [(x: Int, z: Int)]
        /// Currently detected obstacle cells (walls).  Shown as red.
        let obstacleCells: [(x: Int, z: Int)]
        /// World metres per cell.
        let cellSize: Float
    }

    /// Captures the current A* grid state for debug visualisation.
    /// Must be called after a `calculateRoute` completion fires.
    func occupancySnapshot() -> OccupancySnapshot {
        // Only return cells that A* can genuinely route through (explored AND obstacle-free).
        let passable = exploredCells
            .filter { !dilatedCells.contains($0) }
            .map    { (x: $0.x, z: $0.z) }
        return OccupancySnapshot(
            exploredCells: passable,
            obstacleCells: dilatedCells.map { (x: $0.x, z: $0.z) },
            cellSize: cellSize
        )
    }

    /// Copies live ARKit mesh buffers into immutable `Data` blobs for safe background use.
    private func snapshotAnchors(_ anchors: [ARMeshAnchor]) -> [AnchorSnapshot] {
        anchors.compactMap { anchor in
            let geo  = anchor.geometry
            let verts = geo.vertices
            let faces = geo.faces
            let ipf   = faces.indexCountPerPrimitive
            guard ipf == 3, faces.count > 0 else { return nil }
            let vData = Data(bytes: verts.buffer.contents(), count: verts.buffer.length)
            let fData = Data(bytes: faces.buffer.contents(), count: faces.buffer.length)
            return AnchorSnapshot(
                transform:      anchor.transform,
                vertexData:     vData,
                vertexStride:   verts.stride,
                faceData:       fData,
                faceCount:      faces.count,
                bytesPerIndex:  faces.bytesPerIndex,
                indicesPerFace: ipf
            )
        }
    }

    /// Rebuilds floor and obstacle cells from a batch of mesh-anchor snapshots.
    private func rebuildOccupancy(from snapshots: [AnchorSnapshot]) {
        var rawObstacles: Set<GridCell> = []
        var rawFloor:     Set<GridCell> = []

        for snap in snapshots {
            let faceCount = snap.faceCount
            let bpi       = snap.bytesPerIndex
            let ipf       = snap.indicesPerFace
            let faceStep  = max(1, faceCount / faceStepDivisor)

            snap.vertexData.withUnsafeBytes { vRaw in
                snap.faceData.withUnsafeBytes { fRaw in
                    let vBuf = vRaw.baseAddress!
                    let fBuf = fRaw.baseAddress!

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
                                .advanced(by: Int(i) * snap.vertexStride)
                                .assumingMemoryBound(to: SIMD3<Float>.self).pointee
                            let w = snap.transform * simd_float4(lp.x, lp.y, lp.z, 1)
                            return simd_float3(w.x, w.y, w.z)
                        }

                        let v0 = worldVertex(readIdx(0))
                        let v1 = worldVertex(readIdx(1))
                        let v2 = worldVertex(readIdx(2))

                        let cross = simd_cross(v1 - v0, v2 - v0)
                        let area  = simd_length(cross)
                        guard area > 1e-6 else { continue }
                        let normal = cross / area

                        let normalDotUp = simd_dot(normal, SIMD3<Float>(0, 1, 0))
                        let centroid    = (v0 + v1 + v2) / 3

                        if abs(normalDotUp) < 0.5 {
                            rawObstacles.insert(worldToCell(centroid))
                        } else if normalDotUp > 0.7 {
                            rawFloor.insert(worldToCell(centroid))
                        }
                    }
                }
            }
        }

        // Preserve previously seen walls, but clear cells the user has already walked through.
        let passedAndNowClear = exploredCells.subtracting(rawObstacles)
        dilatedCells.formUnion(rawObstacles)
        dilatedCells.subtract(passedAndNowClear)

        // Expand floor coverage slightly to bridge gaps in sparse AR floor scans.
        var expandedFloor = rawFloor
        for c in rawFloor {
            for dx in -4...4 {
                for dz in -4...4 {
                    expandedFloor.insert(GridCell(x: c.x + dx, z: c.z + dz))
                }
            }
        }

        exploredCells.formUnion(expandedFloor)
    }

    /// Clears accumulated occupancy data.
    /// Call this when the AR session restarts or the user requests a fresh scan.
    func resetOccupancy() {
        exploredCells = []
        dilatedCells  = []
    }

    /// Find a smoothed path from `start` to `goal` through free space.
    /// Returns world-space XZ waypoints (Y held constant at `start.y`).
    /// Uses a two-pass strategy: confirmed floor first, obstacle-only fallback second.
    func findPath(from start: simd_float3, to goal: simd_float3) -> [simd_float3] {
        let startCell = worldToCell(start)
        let goalCell  = worldToCell(goal)

        if !exploredCells.isEmpty {
            let sc = nearestFreeCell(to: startCell)
            let gc = nearestFreeCell(to: goalCell)
            if sc != gc {
                let raw = astar(from: sc, to: gc, exploredOnly: true)
                if !raw.isEmpty {
                    return smoothPath(raw.map { cellToWorld($0, y: start.y) })
                }
            }
        }

        let sc2 = nearestFreeCellObstacleOnly(to: startCell)
        let gc2 = nearestFreeCellObstacleOnly(to: goalCell)
        guard sc2 != gc2 else { return [goal] }

        let raw2 = astar(from: sc2, to: gc2, exploredOnly: false)
        guard !raw2.isEmpty else { return [] }
        return smoothPath(raw2.map { cellToWorld($0, y: start.y) })
    }

    /// Finds the nearest obstacle-free cell, ignoring floor-confirmation requirements.
    private func nearestFreeCellObstacleOnly(to cell: GridCell, maxRadius: Int = 8) -> GridCell {
        guard dilatedCells.contains(cell) else { return cell }
        for r in 1...maxRadius {
            for dx in -r...r {
                for dz in -r...r {
                    guard abs(dx) == r || abs(dz) == r else { continue }
                    let c = GridCell(x: cell.x + dx, z: cell.z + dz)
                    if !dilatedCells.contains(c) { return c }
                }
            }
        }
        return cell
    }

    /// Finds the nearest cell that is both obstacle-free and passable for the current routing mode.
    private func nearestFreeCell(to cell: GridCell, maxRadius: Int = 8) -> GridCell {
        let isFree: (GridCell) -> Bool = { [self] c in
            !dilatedCells.contains(c) &&
            (exploredCells.isEmpty || exploredCells.contains(c))
        }
        guard !isFree(cell) else { return cell }
        for r in 1...maxRadius {
            for dx in -r...r {
                for dz in -r...r {
                    guard abs(dx) == r || abs(dz) == r else { continue }
                    let candidate = GridCell(x: cell.x + dx, z: cell.z + dz)
                    if isFree(candidate) { return candidate }
                }
            }
        }
        return cell
    }

    func calculateRoute(
        from start: simd_float3,
        to goal: simd_float3,
        using anchors: [ARMeshAnchor],
        completion: @escaping ([simd_float3]) -> Void
    ) {
        let snapshots = snapshotAnchors(anchors)
        routeQueue.async { [weak self] in
            guard let self else { return }
            self.rebuildOccupancy(from: snapshots)
            let route = self.findPath(from: start, to: goal)
            DispatchQueue.main.async {
                completion(route)
            }
        }
    }

    // MARK: - A* Search

    private struct AStarNode: Comparable {
        let cell: GridCell
        let f: Float
        static func < (a: Self, b: Self) -> Bool { a.f < b.f }
    }

    /// When `exploredOnly` is true, the search is constrained to confirmed floor cells.
    private func astar(from start: GridCell, to goal: GridCell, exploredOnly: Bool) -> [GridCell] {
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
                if exploredOnly && !exploredCells.isEmpty && !exploredCells.contains(nb) { continue }
                let tentG = (gScore[cur.cell] ?? .infinity) + moveCost
                if tentG < (gScore[nb] ?? .infinity) {
                    cameFrom[nb] = cur.cell
                    gScore[nb]   = tentG
                    open.insert(AStarNode(cell: nb, f: tentG + octile(nb, goal)))
                }
            }
        }
        return []
    }

    private func reconstructPath(_ cameFrom: [GridCell: GridCell], end: GridCell) -> [GridCell] {
        var path = [end]
        var cur  = end
        while let prev = cameFrom[cur] { path.append(prev); cur = prev }
        return path.reversed()
    }

    // MARK: - Path Smoothing

    /// Removes unnecessary intermediate points when a later waypoint is directly reachable.
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

    /// Uses Bresenham rasterization to verify obstacle-free line of sight between waypoints.
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

    /// Octile-distance heuristic for 8-directional grid search.
    private func octile(_ a: GridCell, _ b: GridCell) -> Float {
        let dx = Float(abs(a.x - b.x))
        let dz = Float(abs(a.z - b.z))
        return 1.414 * min(dx, dz) + abs(dx - dz)
    }

    /// Returns the eight neighboring cells and their movement costs.
    private func neighborCosts(of c: GridCell) -> [(GridCell, Float)] {
        var result = [(GridCell, Float)]()
        result.reserveCapacity(8)
        for dx in -1...1 {
            for dz in -1...1 {
                guard dx != 0 || dz != 0 else { continue }
                if dx != 0 && dz != 0 {
                    if dilatedCells.contains(GridCell(x: c.x + dx, z: c.z)) ||
                       dilatedCells.contains(GridCell(x: c.x, z: c.z + dz)) {
                        continue
                    }
                }
                let cost: Float = (dx != 0 && dz != 0) ? 1.414 : 1.0
                result.append((GridCell(x: c.x + dx, z: c.z + dz), cost))
            }
        }
        return result
    }
}

// MARK: - Min-Heap

/// Minimal binary heap used as the A* open set.
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
