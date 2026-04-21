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
    /// Extra cells of clearance added around each obstacle cell.
    /// Set to 0 — wall cells are exact, no buffer expansion.
    private let dilationCells: Int = 0
    /// Safety cap to keep pathfinding interactive.
    private let maxAStarIterations = 8_000
    /// Sample only 1-in-N faces per anchor to keep rebuild fast.
    private let faceStepDivisor = 800

    // MARK: - State

    /// Cells confirmed as obstacle-free floor (upward-facing AR mesh faces).
    /// A* only routes through these — unexplored areas are impassable.
    private var exploredCells: Set<GridCell> = []
    /// Cells marked as obstacles (walls / furniture) including dilation buffer.
    private var dilatedCells: Set<GridCell> = []
    /// Low-QoS queue: pathfinding runs at utility priority so it doesn't
    /// compete with the SceneKit render loop on the main thread.
    private let routeQueue = DispatchQueue(label: "inception.navigation.route", qos: .utility)

    // MARK: - Types

    fileprivate struct GridCell: Hashable {
        let x: Int
        let z: Int
    }

    /// Thread-safe snapshot of one ARMeshAnchor's geometry.
    /// MTLBuffer contents are copied into `Data` on the calling (main) thread
    /// so the background queue never touches live ARKit-owned memory.
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

    /// Copies MTLBuffer bytes from ARMeshAnchors into thread-safe `Data` objects.
    /// **Must be called on the main thread** before dispatching to the background queue.
    private func snapshotAnchors(_ anchors: [ARMeshAnchor]) -> [AnchorSnapshot] {
        anchors.compactMap { anchor in
            let geo  = anchor.geometry
            let verts = geo.vertices
            let faces = geo.faces
            let ipf   = faces.indexCountPerPrimitive
            guard ipf == 3, faces.count > 0 else { return nil }
            // Data(bytes:count:) performs a memcpy — safe even if ARKit later mutates the buffer.
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

    /// Rebuild the occupancy map from pre-snapshotted anchor geometry.
    ///
    /// Face classification (world-space normals, gravity-aligned Y axis):
    ///   • |dot(n, up)| < 0.5  → vertical → wall / obstacle
    ///   •  dot(n, up) > 0.7   → upward-facing → navigable floor
    ///   • everything else (ceiling, slanted) → ignored
    ///
    /// Only cells confirmed as floor are passable; unexplored areas are blocked.
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
                            // Vertical face → wall / obstacle
                            rawObstacles.insert(worldToCell(centroid))
                        } else if normalDotUp > 0.7 {
                            // Upward-facing horizontal face → navigable floor
                            rawFloor.insert(worldToCell(centroid))
                        }
                        // Downward (ceiling) and slanted faces are discarded
                    }
                }
            }
        }

        // ── Obstacle persistence with pass-through correction ─────────────────
        // Historical obstacles accumulate (walls persist across scans).
        // Exception: a cell the user has PREVIOUSLY walked through (already in
        // exploredCells) and that the CURRENT scan does NOT see as a wall gets
        // cleared — the user physically passing through is proof it isn't blocked.
        // Cells the user hasn't visited yet keep their historical obstacle status.
        let passedAndNowClear = exploredCells.subtracting(rawObstacles)
        dilatedCells.formUnion(rawObstacles)    // accumulate new walls
        dilatedCells.subtract(passedAndNowClear) // correct false positives on walked paths

        // Dilate floor by 4 cells (~1 m) to bridge gaps in sparse AR floor scans.
        var expandedFloor = rawFloor
        for c in rawFloor {
            for dx in -4...4 {
                for dz in -4...4 {
                    expandedFloor.insert(GridCell(x: c.x + dx, z: c.z + dz))
                }
            }
        }

        // exploredCells only grows — cells are never removed.
        // A* checks (exploredCells.contains && !dilatedCells.contains) at query time,
        // so current obstacles naturally block routing without permanently erasing
        // the floor memory.
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
    /// Returns an empty array when no navigable path exists — never returns a
    /// straight line that cuts through walls.
    ///
    /// Two-pass strategy:
    ///   1. Prefer a path through confirmed floor only (unexplored = impassable).
    ///   2. If that fails (sparse scan) fall back to obstacle-only mode so the
    ///      user still sees *some* route rather than nothing.
    func findPath(from start: simd_float3, to goal: simd_float3) -> [simd_float3] {
        let startCell = worldToCell(start)
        let goalCell  = worldToCell(goal)

        // ── Pass 1: explored-floor constraint ──────────────────────────────
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

        // ── Pass 2: obstacle-only fallback ─────────────────────────────────
        // Floor data exists but no connected path through scanned floor yet
        // (user hasn't scanned the full route).  Route through non-obstacle
        // space so navigation is still usable.
        let sc2 = nearestFreeCellObstacleOnly(to: startCell)
        let gc2 = nearestFreeCellObstacleOnly(to: goalCell)
        guard sc2 != gc2 else { return [goal] }

        let raw2 = astar(from: sc2, to: gc2, exploredOnly: false)
        guard !raw2.isEmpty else { return [] }
        return smoothPath(raw2.map { cellToWorld($0, y: start.y) })
    }

    /// Nearest cell that is obstacle-free (ignores exploredCells constraint).
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

    /// Returns the nearest passable grid cell to `cell`.
    /// A cell is passable when it is not an obstacle AND lies in a scanned
    /// floor area (or no floor data exists yet, in which case only the
    /// obstacle check applies).
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
        return cell   // fallback: original cell
    }

    func calculateRoute(
        from start: simd_float3,
        to goal: simd_float3,
        using anchors: [ARMeshAnchor],
        completion: @escaping ([simd_float3]) -> Void
    ) {
        // Copy MTLBuffer contents into Data on the calling (main) thread BEFORE
        // dispatching to the background queue.  Accessing an ARKit MTLBuffer's
        // raw bytes from a non-AR thread causes EXC_BAD_ACCESS because ARKit
        // may reallocate or update the buffer concurrently.
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

    // MARK: - A*

    private struct AStarNode: Comparable {
        let cell: GridCell
        let f: Float
        static func < (a: Self, b: Self) -> Bool { a.f < b.f }
    }

    /// `exploredOnly`: when true, only routes through cells in `exploredCells`.
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
                // In explored-only mode, skip cells that haven't been scanned as floor.
                if exploredOnly && !exploredCells.isEmpty && !exploredCells.contains(nb) { continue }
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
                // Prevent diagonal moves from cutting through wall corners.
                // A diagonal step (dx,dz) is only legal when both intermediate
                // cardinal cells are also free.
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
