//
//  InferenceService.swift
//  inception
//
//  Optimized version
//

import UIKit
import QuartzCore
import Accelerate
import CoreImage
import CoreVideo
import OnnxRuntimeBindings

final class InferenceService {

    private struct PreprocessMapping {
        let sourceWidth: CGFloat
        let sourceHeight: CGFloat
        let scale: CGFloat
        let offsetX: CGFloat
        let offsetY: CGFloat
    }

    // MARK: - Config

    private let confThreshold: Float = 0.25
    private let iouThreshold: Float = 0.45
    private let inputSize: Int = 640

    // 若你的模型 input/output 名稱不同，改這兩個
    private let inputName = "images"
    private let outputName = "output0"

    // MARK: - ORT

    private var session: ORTSession?
    private var ortEnv: ORTEnv?

    // MARK: - Queues / Context

    private let queue = DispatchQueue(label: "inference.service.queue", qos: .userInitiated)
    private let setupQueue = DispatchQueue(label: "inference.service.setup", qos: .userInitiated)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - State

    private var hasLoggedShape = false
    private var transpositionBuffer = [Float32]()

    // MARK: - Reusable buffers

    private var resizedPixelBuffer: CVPixelBuffer?

    private var rBytes = [UInt8]()
    private var gBytes = [UInt8]()
    private var bBytes = [UInt8]()

    private var inputFloats = [Float32]()
    private var inputMutableData = NSMutableData()

    // MARK: - Init

    init() {
        prepareReusableBuffers()
        setupQueue.async { [weak self] in
            self?.setupModel()
        }
    }

    // MARK: - Public

    func run(
        pixelBuffer: CVPixelBuffer,
        completion: @escaping (Result<InferenceResult, Error>) -> Void
    ) {
        queue.async { [weak self] in
            guard let self else {
                DispatchQueue.main.async {
                    completion(.failure(InferenceError.serviceDeallocated))
                }
                return
            }

            guard let session = self.session else {
                DispatchQueue.main.async {
                    completion(.failure(InferenceError.sessionNotReady))
                }
                return
            }

            autoreleasepool {
                let t0 = CACurrentMediaTime()

                guard let mapping = self.preprocessIntoReusableBuffer(pixelBuffer) else {
                    DispatchQueue.main.async {
                        completion(.success(InferenceResult(detections: [], inferenceMs: 0)))
                    }
                    return
                }

                do {
                    let inputShape: [NSNumber] = [
                        1, 3,
                        NSNumber(value: self.inputSize),
                        NSNumber(value: self.inputSize)
                    ]

                    let inputTensor = try ORTValue(
                        tensorData: self.inputMutableData,
                        elementType: .float,
                        shape: inputShape
                    )

                    let outputs = try session.run(
                        withInputs: [self.inputName: inputTensor],
                        outputNames: Set([self.outputName]),
                        runOptions: nil
                    )

                    let t1 = CACurrentMediaTime()

                    guard let outputValue = outputs[self.outputName] else {
                        DispatchQueue.main.async {
                            completion(.success(InferenceResult(detections: [], inferenceMs: 0)))
                        }
                        return
                    }

                    let shapeInfo = try outputValue.tensorTypeAndShapeInfo()
                    let shape = shapeInfo.shape.map { $0.intValue }

                    if !self.hasLoggedShape {
                        self.hasLoggedShape = true
                        print("📐 \(self.outputName) shape: \(shape)")
                    }

                    let outputTensorData = try outputValue.tensorData()
                    let detections = self.parseDetections(
                        fromTensorBytes: outputTensorData as Data,
                        shape: shape,
                        mapping: mapping
                    )

                    let t2 = CACurrentMediaTime()

                    print(String(
                        format: "推論: %.1fms | 解析: %.1fms | 總計: %.1fms",
                        (t1 - t0) * 1000,
                        (t2 - t1) * 1000,
                        (t2 - t0) * 1000
                    ))

                    DispatchQueue.main.async {
                        completion(.success(InferenceResult(
                            detections: detections,
                            inferenceMs: (t2 - t0) * 1000
                        )))
                    }

                } catch {
                    DispatchQueue.main.async {
                        completion(.failure(error))
                    }
                }
            }
        }
    }

    // MARK: - Setup

    private func prepareReusableBuffers() {
        let planeSize = inputSize * inputSize

        rBytes = [UInt8](repeating: 0, count: planeSize)
        gBytes = [UInt8](repeating: 0, count: planeSize)
        bBytes = [UInt8](repeating: 0, count: planeSize)

        inputFloats = [Float32](repeating: 0, count: 3 * planeSize)
        inputMutableData = NSMutableData(length: inputFloats.count * MemoryLayout<Float32>.stride) ?? NSMutableData()

        resizedPixelBuffer = Self.makePixelBuffer(width: inputSize, height: inputSize)
    }

    private func setupModel() {
        do {
            let env = try ORTEnv(loggingLevel: .warning)
            ortEnv = env

            guard let modelPath = Bundle.main.path(forResource: "yolo26n", ofType: "ort") else {
                print("❌ 找不到 yolo26n.ort（請確認已加入 Target Membership）")
                return
            }

            let options = try ORTSessionOptions()

            // CoreML 下不一定 thread 越多越好，先從 2 開始測
            try options.setIntraOpNumThreads(2)

            // 使用 CoreML EP，若裝置/模型不適合會有 fallback
            try options.appendExecutionProvider("CoreML", providerOptions: [:])

            let s = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
            session = s

            if let inputNames = try? s.inputNames(),
               let outputNames = try? s.outputNames() {
                print("📥 Input names: \(inputNames)")
                print("📤 Output names: \(outputNames)")
            }

            print("✅ ORT 模型載入成功（CoreML EP 已啟用）")
        } catch {
            print("❌ ORT 模型載入失敗：\(error)")
        }
    }

    private static func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]

        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &buffer
        )

        if status != kCVReturnSuccess {
            return nil
        }

        return buffer
    }

    // MARK: - Preprocessing
    // CVPixelBuffer -> Float32 NCHW [1, 3, inputSize, inputSize]

    private func preprocessIntoReusableBuffer(_ pixelBuffer: CVPixelBuffer) -> PreprocessMapping? {
        guard let outputBuffer = resizedPixelBuffer else { return nil }

        let size = inputSize
        let planeSize = size * size

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let sourceWidth = ciImage.extent.width
        let sourceHeight = ciImage.extent.height
        let scale = min(CGFloat(size) / sourceWidth, CGFloat(size) / sourceHeight)
        let scaledWidth = sourceWidth * scale
        let scaledHeight = sourceHeight * scale
        let offsetX = (CGFloat(size) - scaledWidth) * 0.5
        let offsetY = (CGFloat(size) - scaledHeight) * 0.5

        let transformedImage = ciImage.transformed(
            by: CGAffineTransform(scaleX: scale, y: scale)
                .concatenating(CGAffineTransform(translationX: offsetX, y: offsetY))
        )
        let background = CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: size, height: size))
        let letterboxed = transformedImage.composited(over: background)

        ciContext.render(letterboxed, to: outputBuffer)

        CVPixelBufferLockBaseAddress(outputBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(outputBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(outputBuffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(outputBuffer)

        var srcVBuf = vImage_Buffer(
            data: baseAddress,
            height: vImagePixelCount(size),
            width: vImagePixelCount(size),
            rowBytes: bytesPerRow
        )

        rBytes.withUnsafeMutableBufferPointer { rPtr in
            gBytes.withUnsafeMutableBufferPointer { gPtr in
                bBytes.withUnsafeMutableBufferPointer { bPtr in
                    var rBuf = vImage_Buffer(
                        data: rPtr.baseAddress!,
                        height: vImagePixelCount(size),
                        width: vImagePixelCount(size),
                        rowBytes: size
                    )
                    var gBuf = vImage_Buffer(
                        data: gPtr.baseAddress!,
                        height: vImagePixelCount(size),
                        width: vImagePixelCount(size),
                        rowBytes: size
                    )
                    var bBuf = vImage_Buffer(
                        data: bPtr.baseAddress!,
                        height: vImagePixelCount(size),
                        width: vImagePixelCount(size),
                        rowBytes: size
                    )

                    // BGRA: B=0, G=1, R=2, A=3
                    vImageExtractChannel_ARGB8888(&srcVBuf, &rBuf, 2, vImage_Flags(kvImageNoFlags))
                    vImageExtractChannel_ARGB8888(&srcVBuf, &gBuf, 1, vImage_Flags(kvImageNoFlags))
                    vImageExtractChannel_ARGB8888(&srcVBuf, &bBuf, 0, vImage_Flags(kvImageNoFlags))
                }
            }
        }

        var normalizationScale: Float = 1.0 / 255.0

        inputFloats.withUnsafeMutableBufferPointer { ptr in
            let dst = ptr.baseAddress!

            vDSP_vfltu8(rBytes, 1, dst + 0 * planeSize, 1, vDSP_Length(planeSize))
            vDSP_vsmul(dst + 0 * planeSize, 1, &normalizationScale, dst + 0 * planeSize, 1, vDSP_Length(planeSize))

            vDSP_vfltu8(gBytes, 1, dst + 1 * planeSize, 1, vDSP_Length(planeSize))
            vDSP_vsmul(dst + 1 * planeSize, 1, &normalizationScale, dst + 1 * planeSize, 1, vDSP_Length(planeSize))

            vDSP_vfltu8(bBytes, 1, dst + 2 * planeSize, 1, vDSP_Length(planeSize))
            vDSP_vsmul(dst + 2 * planeSize, 1, &normalizationScale, dst + 2 * planeSize, 1, vDSP_Length(planeSize))
        }

        let byteCount = inputFloats.count * MemoryLayout<Float32>.stride
        inputMutableData.length = byteCount
        memcpy(inputMutableData.mutableBytes, inputFloats, byteCount)

        return PreprocessMapping(
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            scale: scale,
            offsetX: offsetX,
            offsetY: offsetY
        )
    }

    // MARK: - Parsing

    private func parseDetections(
        fromTensorBytes tensorBytes: Data,
        shape: [Int],
        mapping: PreprocessMapping
    ) -> [Detection] {
        return tensorBytes.withUnsafeBytes { raw in
            guard let ptr = raw.baseAddress?.assumingMemoryBound(to: Float32.self) else {
                return []
            }
            return parseDetections(from: ptr, shape: shape, mapping: mapping)
        }
    }

    // shape 末兩維支援 [C, N] 與 [N, C]
    private func parseDetections(
        from ptr: UnsafePointer<Float32>,
        shape: [Int],
        mapping: PreprocessMapping
    ) -> [Detection] {
        guard shape.count >= 2 else { return [] }

        let dim1 = shape[shape.count - 2]
        let dim2 = shape[shape.count - 1]

        let channelsFirst = dim1 < dim2
        let numChannels = channelsFirst ? dim1 : dim2
        let numAnchors  = channelsFirst ? dim2 : dim1

        guard numChannels > 4, numAnchors > 0 else { return [] }

        let hasMask = numChannels > 84
        let numClasses = hasMask ? numChannels - 4 - 32 : numChannels - 4
        let maskStart = 4 + numClasses

        guard numClasses > 0 else { return [] }

        var detections: [Detection] = []
        detections.reserveCapacity(64)

        if channelsFirst {
            let needed = numAnchors * numChannels
            if transpositionBuffer.count < needed {
                transpositionBuffer = [Float32](repeating: 0, count: needed)
            }

            transpositionBuffer.withUnsafeMutableBufferPointer { dstBuf in
                vDSP_mtrans(
                    ptr,
                    1,
                    dstBuf.baseAddress!,
                    1,
                    vDSP_Length(numAnchors),
                    vDSP_Length(numChannels)
                )
            }

            transpositionBuffer.withUnsafeBufferPointer { buf in
                guard let basePtr = buf.baseAddress else { return }
                parseDetectionsContiguous(
                    from: basePtr,
                    numChannels: numChannels,
                    numAnchors: numAnchors,
                    numClasses: numClasses,
                    hasMask: hasMask,
                    maskStart: maskStart,
                    mapping: mapping,
                    into: &detections
                )
            }
        } else {
            parseDetectionsContiguous(
                from: ptr,
                numChannels: numChannels,
                numAnchors: numAnchors,
                numClasses: numClasses,
                hasMask: hasMask,
                maskStart: maskStart,
                mapping: mapping,
                into: &detections
            )
        }

        return nonMaxSuppression(detections)
    }

    private func parseDetectionsContiguous(
        from tPtr: UnsafePointer<Float32>,
        numChannels: Int,
        numAnchors: Int,
        numClasses: Int,
        hasMask: Bool,
        maskStart: Int,
        mapping: PreprocessMapping,
        into detections: inout [Detection]
    ) {
        for i in 0..<numAnchors {
            let base = i * numChannels

            let cx = tPtr[base + 0]
            let cy = tPtr[base + 1]
            let w  = tPtr[base + 2]
            let h  = tPtr[base + 3]

            var maxConf: Float = 0
            var maxClassU: vDSP_Length = 0
            vDSP_maxvi(tPtr + base + 4, 1, &maxConf, &maxClassU, vDSP_Length(numClasses))

            guard maxConf >= confThreshold else { continue }

            let maxClass = Int(maxClassU)

            var coeffs: [Float] = []
            if hasMask {
                coeffs.reserveCapacity(32)
                for m in 0..<32 {
                    coeffs.append(tPtr[base + maskStart + m])
                }
            }

            appendDetection(cx, cy, w, h, maxClass, maxConf, coeffs, mapping: mapping, to: &detections)
        }
    }

    private func appendDetection(
        _ cx: Float,
        _ cy: Float,
        _ w: Float,
        _ h: Float,
        _ classId: Int,
        _ conf: Float,
        _ coeffs: [Float],
        mapping: PreprocessMapping,
        to detections: inout [Detection]
    ) {
        let modelMinX = CGFloat(cx - w / 2)
        let modelMinY = CGFloat(cy - h / 2)
        let modelWidth = CGFloat(w)
        let modelHeight = CGFloat(h)

        let sourceMinX = (modelMinX - mapping.offsetX) / mapping.scale
        let sourceMinY = (modelMinY - mapping.offsetY) / mapping.scale
        let sourceWidth = modelWidth / mapping.scale
        let sourceHeight = modelHeight / mapping.scale

        let x = max(0, min(1, sourceMinX / mapping.sourceWidth))
        let y = max(0, min(1, sourceMinY / mapping.sourceHeight))
        let maxX = max(0, min(1, (sourceMinX + sourceWidth) / mapping.sourceWidth))
        let maxY = max(0, min(1, (sourceMinY + sourceHeight) / mapping.sourceHeight))

        guard maxX > x, maxY > y else { return }

        detections.append(
            Detection(
                bbox: CGRect(
                    x: CGFloat(x),
                    y: CGFloat(y),
                    width: maxX - x,
                    height: maxY - y
                ),
                classId: classId,
                className: classId < COCO_CLASSES.count ? COCO_CLASSES[classId] : "unknown",
                confidence: conf,
                maskCoeffs: coeffs
            )
        )
    }

    // MARK: - NMS

    private func nonMaxSuppression(_ dets: [Detection]) -> [Detection] {
        guard !dets.isEmpty else { return [] }

        let sorted = dets.sorted { $0.confidence > $1.confidence }
        var suppressed = Array(repeating: false, count: sorted.count)
        var kept: [Detection] = []
        kept.reserveCapacity(min(sorted.count, 64))

        for i in 0..<sorted.count {
            guard !suppressed[i] else { continue }
            kept.append(sorted[i])

            for j in (i + 1)..<sorted.count {
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

}

// MARK: - Result

struct InferenceResult {
    let detections: [Detection]
    let inferenceMs: Double
}

// MARK: - Error

enum InferenceError: LocalizedError {
    case sessionNotReady
    case serviceDeallocated

    var errorDescription: String? {
        switch self {
        case .sessionNotReady:
            return "Inference session 尚未初始化完成"
        case .serviceDeallocated:
            return "InferenceService 已被釋放"
        }
    }
}
