//
//  InferenceService.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/8/26.
//

@preconcurrency import Vision
import CoreML
import UIKit
import QuartzCore
import Accelerate

final class InferenceService {

    private let confThreshold: Float = 0.25
    private let iouThreshold: Float = 0.45
    private let inputSize: CGFloat = 640
    private let overlaySize: CGFloat = 320

    private var request: VNCoreMLRequest?
    private let queue = DispatchQueue(label: "inference.service.queue", qos: .userInitiated)

    init() {
        setupModel()
    }

    private func setupModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine

            let model = try yolov8n_seg(configuration: config)
            let vnModel = try VNCoreMLModel(for: model.model)

            let req = VNCoreMLRequest(model: vnModel)
            req.imageCropAndScaleOption = .scaleFill
            self.request = req

            print("✅ 模型載入成功")
        } catch {
            print("❌ 模型載入失敗：\(error)")
        }
    }

    // 執行推論，回傳結果
    func run(
        pixelBuffer: CVPixelBuffer,
        completion: @escaping (Result<InferenceResult, Error>) -> Void
    ) {
        queue.async { [weak self] in
            guard let self, let request else { return }

            let t0 = CACurrentMediaTime()

            // ARKit buffer is landscape (width > height). .up = buffer top is scene top.
            let handler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer,
                orientation: .up
            )

            do {
                try handler.perform([request])
                let t1 = CACurrentMediaTime()

                guard let results = request.results else {
                    completion(.success(InferenceResult(detections: [], overlay: nil, inferenceMs: 0)))
                    return
                }

                let detections = self.extractDetections(from: results)
                let t2 = CACurrentMediaTime()

                let overlay = self.renderOverlay(detections)
                let t3 = CACurrentMediaTime()

                print(String(format: "推論: %.1fms | 解析: %.1fms | 渲染: %.1fms",
                    (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000))

                let result = InferenceResult(
                    detections: detections,
                    overlay: overlay,
                    inferenceMs: (t3 - t0) * 1000
                )
                completion(.success(result))

            } catch {
                completion(.failure(error))
            }
        }
    }

    // MARK: - 解析 tensor
    private func extractDetections(from results: [VNObservation]) -> [Detection] {
        for obs in results {
            guard
                let feature = obs as? VNCoreMLFeatureValueObservation,
                feature.featureName == "var_1012",
                let array = feature.featureValue.multiArrayValue
            else { continue }

            return parseDetections(array)
        }
        return []
    }

    private func parseDetections(_ array: MLMultiArray) -> [Detection] {
        let numAnchors = 8400
        let numClasses = 80
        var detections: [Detection] = []
        detections.reserveCapacity(64)

        // 直接用 pointer 存取，比 NSNumber 快 10 倍
        let ptr = array.dataPointer.bindMemory(
            to: Float32.self,
            capacity: array.count
        )
        let stride = numAnchors

        for i in 0..<numAnchors {
            let cx = ptr[0 * stride + i]
            let cy = ptr[1 * stride + i]
            let w  = ptr[2 * stride + i]
            let h  = ptr[3 * stride + i]

            // 用 vDSP 找最大 class，取代 80 次迴圈
            var maxConf: Float = 0
            var maxClassU: vDSP_Length = 0
            vDSP_maxvi(
                ptr.advanced(by: 4 * stride + i),
                stride,
                &maxConf,
                &maxClassU,
                vDSP_Length(numClasses)
            )
            let maxClass = Int(maxClassU) / stride

            guard maxConf >= confThreshold else { continue }

            var coeffs = [Float]()
            coeffs.reserveCapacity(32)
            for m in 0..<32 {
                coeffs.append(ptr[(84 + m) * stride + i])
            }

            let x  = (cx - w / 2) / Float(inputSize)
            let y  = (cy - h / 2) / Float(inputSize)
            let nw = w / Float(inputSize)
            let nh = h / Float(inputSize)

            detections.append(Detection(
                bbox: CGRect(x: CGFloat(x), y: CGFloat(y),
                             width: CGFloat(nw), height: CGFloat(nh)),
                classId: maxClass,
                className: maxClass < COCO_CLASSES.count ? COCO_CLASSES[maxClass] : "unknown",
                confidence: maxConf,
                maskCoeffs: coeffs
            ))
        }

        return nonMaxSuppression(detections)
    }

    // MARK: - NMS
    private func nonMaxSuppression(_ dets: [Detection]) -> [Detection] {
        let sorted = dets.sorted { $0.confidence > $1.confidence }
        var suppressed = Array(repeating: false, count: sorted.count)
        var kept: [Detection] = []

        for i in 0..<sorted.count {
            guard !suppressed[i] else { continue }
            kept.append(sorted[i])
            for j in (i+1)..<sorted.count {
                guard !suppressed[j] else { continue }
                if iou(sorted[i].bbox, sorted[j].bbox) > CGFloat(iouThreshold) {
                    suppressed[j] = true
                }
            }
        }
        return kept
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return unionArea > 0 ? interArea / unionArea : 0
    }

    // MARK: - 渲染（只畫框和填色，文字由 SwiftUI 處理）
    func renderOverlay(_ detections: [Detection]) -> UIImage? {
        guard !detections.isEmpty else { return nil }

        let size = CGSize(width: overlaySize, height: overlaySize)
        let renderer = UIGraphicsImageRenderer(size: size)

        return renderer.image { ctx in
            let cg = ctx.cgContext
            let bounds = CGRect(origin: .zero, size: size)

            for det in detections {
                let box = CGRect(
                    x: det.bbox.minX * size.width,
                    y: det.bbox.minY * size.height,
                    width: det.bbox.width * size.width,
                    height: det.bbox.height * size.height
                ).intersection(bounds)

                guard !box.isNull, !box.isEmpty else { continue }

                cg.setStrokeColor(det.color.cgColor)
                cg.setLineWidth(1.5)
                cg.stroke(box)

                cg.setFillColor(det.color.withAlphaComponent(0.15).cgColor)
                cg.fill(box)
            }
        }
    }
}

// MARK: - 推論結果
struct InferenceResult {
    let detections: [Detection]
    let overlay: UIImage?
    let inferenceMs: Double
}
