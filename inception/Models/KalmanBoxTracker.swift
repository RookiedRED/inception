//
//  KalmanBoxTracker.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/9/26.
//

import Foundation
import CoreGraphics
import simd

/// 6-state Kalman filter for bounding box tracking.
/// State: [cx, cy, w, h, vx, vy]
/// Measurement: [cx, cy, w, h]
struct KalmanBoxTracker {

    // State vector (8)
    private(set) var state: (
        cx: Float, cy: Float,
        w: Float, h: Float,
        vx: Float, vy: Float,
        vw: Float, vh: Float
    )

    // Covariance matrix P (8x8), stored as flat array row-major
    private var P: [Float]

    // Process noise diagonal values
    private static let qDiag: [Float] = [0.01, 0.01, 0.008, 0.008, 0.04, 0.04, 0.02, 0.02]
    // Measurement noise diagonal values
    private static let rDiag: [Float] = [0.02, 0.02, 0.08, 0.08]

    var predictedBBox: CGRect {
        CGRect(
            x: CGFloat(state.cx - state.w / 2),
            y: CGFloat(state.cy - state.h / 2),
            width: CGFloat(state.w),
            height: CGFloat(state.h)
        )
    }

    var velocity: SIMD2<Float> { SIMD2(state.vx, state.vy) }

    init(bbox: CGRect) {
        state = (
            cx: Float(bbox.midX),
            cy: Float(bbox.midY),
            w: Float(bbox.width),
            h: Float(bbox.height),
            vx: 0, vy: 0,
            vw: 0, vh: 0
        )
        // Initial covariance: high uncertainty for velocity
        P = Self.makeDiagonal([0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 0.5, 0.5])
    }

    // MARK: - Predict

    /// Advance state by one time step. Returns predicted bbox.
    @discardableResult
    mutating func predict() -> CGRect {
        
        assert(P.count == 64, "Kalman P must be 8x8, but got \(P.count)")
        // x = F * x
        state.cx += state.vx
        state.cy += state.vy
        state.w  += state.vw
        state.h  += state.vh

        // Clamp to valid size
        state.w = max(state.w, 0.01)
        state.h = max(state.h, 0.01)

        let n = 8

        // F = identity with:
        // F[0,4] = 1, F[1,5] = 1, F[2,6] = 1, F[3,7] = 1
        // P = FPF^T + Q

        // Row updates for F * P
        for j in 0..<n {
            P[0 * n + j] += P[4 * n + j]
            P[1 * n + j] += P[5 * n + j]
            P[2 * n + j] += P[6 * n + j]
            P[3 * n + j] += P[7 * n + j]
        }

        // Column updates for (F * P) * F^T
        for i in 0..<n {
            P[i * n + 0] += P[i * n + 4]
            P[i * n + 1] += P[i * n + 5]
            P[i * n + 2] += P[i * n + 6]
            P[i * n + 3] += P[i * n + 7]
        }

        // Add process noise
        for i in 0..<n {
            P[i * n + i] += Self.qDiag[i]
        }

        return predictedBBox
    }

    // MARK: - Update

    /// Incorporate a measurement (detected bbox).
    mutating func update(bbox: CGRect) {
        assert(P.count == 64, "Kalman P must be 8x8, but got \(P.count)")
        
        let z = SIMD4<Float>(
            Float(bbox.midX),
            Float(bbox.midY),
            Float(bbox.width),
            Float(bbox.height)
        )

        let predicted = SIMD4<Float>(state.cx, state.cy, state.w, state.h)
        let y = z - predicted

        // S = HPH^T + R -> top-left 4x4 of P + R
        var S = simd_float4x4(0)
        for i in 0..<4 {
            for j in 0..<4 {
                S[j][i] = P[i * 8 + j]
            }
            S[i][i] += Self.rDiag[i]
        }

        let Sinv = S.inverse

        // K = P H^T S^-1
        // H selects first 4 states, so P H^T = first 4 columns of P
        var K = [[Float]](repeating: [Float](repeating: 0, count: 4), count: 8)
        for i in 0..<8 {
            for j in 0..<4 {
                var sum: Float = 0
                for k in 0..<4 {
                    sum += P[i * 8 + k] * Sinv[j][k]
                }
                K[i][j] = sum
            }
        }

        // x = x + Ky
        state.cx += K[0][0] * y[0] + K[0][1] * y[1] + K[0][2] * y[2] + K[0][3] * y[3]
        state.cy += K[1][0] * y[0] + K[1][1] * y[1] + K[1][2] * y[2] + K[1][3] * y[3]
        state.w  += K[2][0] * y[0] + K[2][1] * y[1] + K[2][2] * y[2] + K[2][3] * y[3]
        state.h  += K[3][0] * y[0] + K[3][1] * y[1] + K[3][2] * y[2] + K[3][3] * y[3]
        state.vx += K[4][0] * y[0] + K[4][1] * y[1] + K[4][2] * y[2] + K[4][3] * y[3]
        state.vy += K[5][0] * y[0] + K[5][1] * y[1] + K[5][2] * y[2] + K[5][3] * y[3]
        state.vw += K[6][0] * y[0] + K[6][1] * y[1] + K[6][2] * y[2] + K[6][3] * y[3]
        state.vh += K[7][0] * y[0] + K[7][1] * y[1] + K[7][2] * y[2] + K[7][3] * y[3]

        state.w = max(state.w, 0.01)
        state.h = max(state.h, 0.01)

        // P = (I - KH)P
        var newP = [Float](repeating: 0, count: 64)
        for i in 0..<8 {
            for j in 0..<8 {
                var sum: Float = 0
                for l in 0..<8 {
                    let ikh: Float
                    if l < 4 {
                        ikh = (i == l ? 1.0 : 0.0) - K[i][l]
                    } else {
                        ikh = (i == l ? 1.0 : 0.0)
                    }
                    sum += ikh * P[l * 8 + j]
                }
                newP[i * 8 + j] = sum
            }
        }
        P = newP
    }

    // MARK: - Helpers

    private static func makeDiagonal(_ values: [Float]) -> [Float] {
        let n = values.count
        var m = [Float](repeating: 0, count: n * n)
        for i in 0..<n { m[i * n + i] = values[i] }
        return m
    }
    
    
    mutating func smoothSize(toward bbox: CGRect, alpha: Float = 0.15) {
        let mw = Float(bbox.width)
        let mh = Float(bbox.height)
        state.w = state.w * (1 - alpha) + mw * alpha
        state.h = state.h * (1 - alpha) + mh * alpha
        state.w = max(state.w, 0.01)
        state.h = max(state.h, 0.01)
    }
}
