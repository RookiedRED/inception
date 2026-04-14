//
//  HungarianSolver.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/9/26.
//

import Foundation

/// O(n³) Hungarian (Munkres) algorithm for optimal linear assignment.
/// Solves: minimize total cost of assigning rows to columns.
enum HungarianSolver {

    /// Solve the assignment problem on an n×m cost matrix.
    /// - Parameter costs: Row-major cost matrix `costs[row][col]`.
    /// - Returns: Array of `(row, col)` pairs representing the optimal assignment.
    ///   Only includes assignments where both row < n and col < m (ignores padding).
    static func solve(costs: [[Float]]) -> [(row: Int, col: Int)] {
        let n = costs.count
        guard n > 0 else { return [] }
        let m = costs[0].count
        guard m > 0 else { return [] }

        // Pad to square matrix (size = max(n, m))
        let sz = max(n, m)

        // Build padded cost matrix (1-indexed internally: rows 1..sz, cols 1..sz)
        // Padding entries get cost 0 (they won't appear in final output)
        var c = [[Float]](repeating: [Float](repeating: 0, count: sz + 1), count: sz + 1)
        for i in 0..<n {
            for j in 0..<m {
                c[i + 1][j + 1] = costs[i][j]
            }
        }

        // Potentials
        var u = [Float](repeating: 0, count: sz + 1)
        var v = [Float](repeating: 0, count: sz + 1)

        // p[j] = row assigned to column j (1-indexed, 0 = unassigned)
        var p = [Int](repeating: 0, count: sz + 1)
        var way = [Int](repeating: 0, count: sz + 1)

        for i in 1...sz {
            // Start augmenting path from row i
            p[0] = i
            var j0 = 0
            var minv = [Float](repeating: .greatestFiniteMagnitude, count: sz + 1)
            var used = [Bool](repeating: false, count: sz + 1)

            repeat {
                used[j0] = true
                let i0 = p[j0]
                var delta: Float = .greatestFiniteMagnitude
                var j1 = 0

                for j in 1...sz {
                    if used[j] { continue }
                    let cur = c[i0][j] - u[i0] - v[j]
                    if cur < minv[j] {
                        minv[j] = cur
                        way[j] = j0
                    }
                    if minv[j] < delta {
                        delta = minv[j]
                        j1 = j
                    }
                }

                // Update potentials
                for j in 0...sz {
                    if used[j] {
                        u[p[j]] += delta
                        v[j] -= delta
                    } else {
                        minv[j] -= delta
                    }
                }

                j0 = j1
            } while p[j0] != 0

            // Trace back augmenting path
            repeat {
                let j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while j0 != 0
        }

        // Extract assignments (convert back to 0-indexed)
        var result: [(row: Int, col: Int)] = []
        for j in 1...sz {
            let row = p[j] - 1
            let col = j - 1
            if row >= 0, row < n, col < m {
                result.append((row, col))
            }
        }

        return result
    }
}
