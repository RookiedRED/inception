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
import CoreML
import CoreVideo
import OnnxRuntimeBindings

/// Handles image preprocessing, ONNX Runtime execution, and model-output decoding.
final class InferenceService {

    struct Configuration {
        enum Backend: CustomStringConvertible {
            case ort
            case coreMLPackage

            var description: String {
                switch self {
                case .ort:
                    return "ort"
                case .coreMLPackage:
                    return "coreMLPackage"
                }
            }
        }

        enum ExecutionProfile {
            case cpuOnly
            case cpuAndNeuralEngine
            case all

            var coreMLComputeUnitsValue: String? {
                switch self {
                case .cpuOnly:
                    return nil
                case .cpuAndNeuralEngine:
                    return "CPUAndNeuralEngine"
                case .all:
                    return "ALL"
                }
            }

            var intraOpThreadCount: Int32 {
                switch self {
                case .cpuOnly:
                    return 4
                case .cpuAndNeuralEngine, .all:
                    return 1
                }
            }
        }

        let inputSize: Int
        let backend: Backend
        let executionProfile: ExecutionProfile
        let ortModelResourceName: String
        let coreMLModelResourceName: String

        static let realtimeARORT = Configuration(
            inputSize: 640,
            backend: .ort,
            executionProfile: .cpuAndNeuralEngine,
            ortModelResourceName: "yolo26n",
            coreMLModelResourceName: "yolo26n"
        )

        static let realtimeARCoreML = Configuration(
            inputSize: 640,
            backend: .coreMLPackage,
            executionProfile: .cpuAndNeuralEngine,
            ortModelResourceName: "yolo26n",
            coreMLModelResourceName: "yolo26n"
        )
    }

    /// Stores the letterbox transform required to map detections back into source-image space.
    private struct PreprocessMapping {
        let sourceWidth: CGFloat
        let sourceHeight: CGFloat
        let scale: CGFloat
        let offsetX: CGFloat
        let offsetY: CGFloat
    }

    // MARK: - Model Configuration

    private let confThreshold: Float = 0.25
    private let iouThreshold: Float = 0.45
    private let configuration: Configuration
    private var inputSize: Int { configuration.inputSize }

    /// Input/output tensor names expected by the bundled model.
    private let inputName = "images"
    private let outputName = "output0"

    // MARK: - ONNX Runtime State

    private var session: ORTSession?
    private var ortEnv: ORTEnv?
    private var coreMLModel: MLModel?
    private var coreMLInputName: String?
    private var coreMLOutputName: String?
    private var coreMLInputIsImage = false
    private var coreMLInputMultiArray: MLMultiArray?

    // MARK: - Queues and Render Context

    private let queue = DispatchQueue(label: "inference.service.queue", qos: .userInitiated)
    private let setupQueue = DispatchQueue(label: "inference.service.setup", qos: .userInitiated)
    // Device RGB skips sRGB↔linear conversion — color accuracy is irrelevant for ML preprocessing.
    private let ciContext = CIContext(options: [
        .useSoftwareRenderer: false,
        .workingColorSpace: CGColorSpaceCreateDeviceRGB(),
        .outputColorSpace: CGColorSpaceCreateDeviceRGB()
    ])
    private var letterboxBackgroundImage: CIImage?

    // MARK: - Mutable State

    private var hasLoggedShape = false
    private var hasLoggedCoreMLPredictionLayout = false
    private var transpositionBuffer = [Float32]()
    private var outputConversionBuffer = [Float32]()

    // MARK: - Reusable Buffers

    private var resizedPixelBuffer: CVPixelBuffer?

    private var rBytes = [UInt8]()
    private var gBytes = [UInt8]()
    private var bBytes = [UInt8]()

    /// Shared input tensor backing storage reused across inference calls.
    private var inputMutableData = NSMutableData()

    /// Pre-allocated ORT tensor that points at `inputMutableData`.
    private var inputTensor: ORTValue?

    /// Cached letterbox transform, recomputed only when source resolution changes.
    private var cachedSourceSize: CGSize = .zero
    private var cachedMapping: PreprocessMapping?

    // MARK: - Init

    init(configuration: Configuration = .realtimeARORT) {
        self.configuration = configuration
        prepareReusableBuffers()
        setupQueue.async { [weak self] in
            self?.setupModel()
        }
    }

    // MARK: - Public API

    /// Runs preprocessing, model execution, and postprocessing on the inference queue.
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

            autoreleasepool {
                let t0 = CACurrentMediaTime()

                guard let preparedInput = self.preprocess(pixelBuffer) else {
                    DispatchQueue.main.async {
                        completion(.success(InferenceResult(detections: [], inferenceMs: 0)))
                    }
                    return
                }

                do {
                    let detections: [Detection]
                    switch self.configuration.backend {
                    case .ort:
                        detections = try self.runORTInference(
                            using: preparedInput.pixelBuffer,
                            mapping: preparedInput.mapping
                        )
                    case .coreMLPackage:
                        detections = try self.runCoreMLInference(
                            using: preparedInput.pixelBuffer,
                            mapping: preparedInput.mapping
                        )
                    }

                    let t2 = CACurrentMediaTime()

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

    /// Allocates all reusable CPU-side buffers used during preprocessing.
    private func prepareReusableBuffers() {
        let planeSize = inputSize * inputSize

        rBytes = [UInt8](repeating: 0, count: planeSize)
        gBytes = [UInt8](repeating: 0, count: planeSize)
        bBytes = [UInt8](repeating: 0, count: planeSize)

        inputMutableData = NSMutableData(length: 3 * planeSize * MemoryLayout<Float32>.stride) ?? NSMutableData()

        // ORT holds a raw pointer to `inputMutableData.mutableBytes`, so the tensor can be reused.
        let inputShape: [NSNumber] = [1, 3, NSNumber(value: inputSize), NSNumber(value: inputSize)]
        inputTensor = try? ORTValue(tensorData: inputMutableData, elementType: .float, shape: inputShape)

        resizedPixelBuffer = Self.makePixelBuffer(width: inputSize, height: inputSize)
    }

    /// Loads and configures the ONNX Runtime session.
    private func setupModel() {
        switch configuration.backend {
        case .ort:
            setupORTModel()
        case .coreMLPackage:
            setupCoreMLModel()
        }
    }

    private func setupORTModel() {
        do {
            let env = try ORTEnv(loggingLevel: .warning)
            ortEnv = env

            guard let modelPath = Bundle.main.path(
                forResource: configuration.ortModelResourceName,
                ofType: "ort"
            ) else {
                print("❌ 找不到 \(configuration.ortModelResourceName).ort（請確認已加入 Target Membership）")
                return
            }

            let options = try ORTSessionOptions()

            try options.setIntraOpNumThreads(configuration.executionProfile.intraOpThreadCount)

            if let computeUnits = configuration.executionProfile.coreMLComputeUnitsValue {
                // For live AR, prefer ANE-backed execution and keep CPU headroom for camera/depth work.
                try options.appendCoreMLExecutionProvider(withOptionsV2: [
                    "MLComputeUnits": computeUnits,
                    "ModelFormat": "MLProgram"
                ])
            }

            let s = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
            session = s

            if let inputNames = try? s.inputNames(),
               let outputNames = try? s.outputNames() {
                print("📥 Input names: \(inputNames)")
                print("📤 Output names: \(outputNames)")
            }

            print("✅ ORT 模型載入成功（input: \(inputSize), profile: \(configuration.executionProfile)）")
        } catch {
            print("❌ ORT 模型載入失敗：\(error)")
        }
    }

    private func setupCoreMLModel() {
        do {
            guard let modelURL = Bundle.main.url(
                forResource: configuration.coreMLModelResourceName,
                withExtension: "mlmodelc"
            ) else {
                print("❌ 找不到 \(configuration.coreMLModelResourceName).mlmodelc（請確認 mlpackage 已加入 Target Membership）")
                return
            }

            let modelConfig = MLModelConfiguration()
            modelConfig.computeUnits = configuration.executionProfile.mlComputeUnits

            let model = try MLModel(contentsOf: modelURL, configuration: modelConfig)
            coreMLModel = model

            let inputDescriptions = model.modelDescription.inputDescriptionsByName
            if let imageInput = inputDescriptions.first(where: { $0.value.type == .image }) {
                coreMLInputName = imageInput.key
                coreMLInputIsImage = true
            } else if let multiArrayInput = inputDescriptions.first(where: { $0.value.type == .multiArray }) {
                coreMLInputName = multiArrayInput.key
                coreMLInputIsImage = false
                let shape = multiArrayInput.value.multiArrayConstraint?.shape ?? [1, 3, inputSize as NSNumber, inputSize as NSNumber]
                coreMLInputMultiArray = try? MLMultiArray(shape: shape, dataType: .float32)
            }

            coreMLOutputName = selectCoreMLOutputName(from: model.modelDescription.outputDescriptionsByName)

            print("✅ Core ML 模型載入成功（backend: \(configuration.backend), computeUnits: \(configuration.executionProfile.mlComputeUnits)）")
            print("📥 Core ML inputs: \(Array(inputDescriptions.keys).sorted())")
            print("📤 Core ML outputs: \(Array(model.modelDescription.outputDescriptionsByName.keys).sorted())")
            for (name, description) in model.modelDescription.outputDescriptionsByName.sorted(by: { $0.key < $1.key }) {
                if let shape = description.multiArrayConstraint?.shape {
                    print("   output \(name): multiArray shape=\(shape)")
                } else {
                    print("   output \(name): type=\(description.type.rawValue)")
                }
            }
            if let coreMLOutputName {
                print("🎯 Selected Core ML output: \(coreMLOutputName)")
            }
        } catch {
            print("❌ Core ML 模型載入失敗：\(error)")
        }
    }

    private func selectCoreMLOutputName(from outputs: [String: MLFeatureDescription]) -> String? {
        if outputs.keys.contains(outputName) {
            return outputName
        }

        let sortedCandidates = outputs
            .filter { $0.value.type == .multiArray }
            .sorted { lhs, rhs in
                let lhsScore = coreMLOutputSelectionScore(for: lhs.value)
                let rhsScore = coreMLOutputSelectionScore(for: rhs.value)
                if lhsScore == rhsScore {
                    return lhs.key < rhs.key
                }
                return lhsScore > rhsScore
            }

        return sortedCandidates.first?.key
    }

    private func coreMLOutputSelectionScore(for description: MLFeatureDescription) -> Int {
        guard let shape = description.multiArrayConstraint?.shape.map({ $0.intValue }), shape.count >= 2 else {
            return 0
        }

        let trailingA = shape[shape.count - 2]
        let trailingB = shape[shape.count - 1]
        let maxDim = max(trailingA, trailingB)
        let minDim = min(trailingA, trailingB)

        if maxDim >= 80, minDim >= 4 {
            return 3
        }
        if maxDim >= 80 {
            return 2
        }
        if minDim >= 4 {
            return 1
        }
        return 0
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

    private struct PreparedInput {
        let pixelBuffer: CVPixelBuffer
        let mapping: PreprocessMapping
    }

    /// Renders the incoming frame into the fixed-size model input buffer and returns the letterbox mapping.
    private func preprocess(_ pixelBuffer: CVPixelBuffer) -> PreparedInput? {
        guard let outputBuffer = resizedPixelBuffer else { return nil }

        let size = inputSize

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let sourceWidth = ciImage.extent.width
        let sourceHeight = ciImage.extent.height

        // Recompute the letterbox mapping only when the source resolution changes.
        let sourceSize = CGSize(width: sourceWidth, height: sourceHeight)
        let mapping: PreprocessMapping
        if sourceSize == cachedSourceSize, let cached = cachedMapping {
            mapping = cached
        } else {
            let scale = min(CGFloat(size) / sourceWidth, CGFloat(size) / sourceHeight)
            let scaledWidth = sourceWidth * scale
            let scaledHeight = sourceHeight * scale
            let offsetX = (CGFloat(size) - scaledWidth) * 0.5
            let offsetY = (CGFloat(size) - scaledHeight) * 0.5
            let newMapping = PreprocessMapping(sourceWidth: sourceWidth, sourceHeight: sourceHeight,
                                               scale: scale, offsetX: offsetX, offsetY: offsetY)
            cachedMapping = newMapping
            cachedSourceSize = sourceSize
            mapping = newMapping
        }

        let transformedImage = ciImage.transformed(
            by: CGAffineTransform(scaleX: mapping.scale, y: mapping.scale)
                .concatenating(CGAffineTransform(translationX: mapping.offsetX, y: mapping.offsetY))
        )
        let background: CIImage
        if let letterboxBackgroundImage {
            background = letterboxBackgroundImage
        } else {
            let image = CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: size, height: size))
            letterboxBackgroundImage = image
            background = image
        }
        let letterboxed = transformedImage.composited(over: background)

        ciContext.render(letterboxed, to: outputBuffer)

        return PreparedInput(pixelBuffer: outputBuffer, mapping: mapping)
    }

    private func fillORTInputTensor(from pixelBuffer: CVPixelBuffer) throws {
        let size = inputSize
        let planeSize = size * size

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw InferenceError.invalidPreprocessedBuffer
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        var srcVBuf = vImage_Buffer(
            data: baseAddress,
            height: vImagePixelCount(size),
            width: vImagePixelCount(size),
            rowBytes: bytesPerRow
        )

        rBytes.withUnsafeMutableBufferPointer { rPtr in
            gBytes.withUnsafeMutableBufferPointer { gPtr in
                bBytes.withUnsafeMutableBufferPointer { bPtr in
                    var rBuf = vImage_Buffer(data: rPtr.baseAddress!, height: vImagePixelCount(size), width: vImagePixelCount(size), rowBytes: size)
                    var gBuf = vImage_Buffer(data: gPtr.baseAddress!, height: vImagePixelCount(size), width: vImagePixelCount(size), rowBytes: size)
                    var bBuf = vImage_Buffer(data: bPtr.baseAddress!, height: vImagePixelCount(size), width: vImagePixelCount(size), rowBytes: size)

                    vImageExtractChannel_ARGB8888(&srcVBuf, &rBuf, 2, vImage_Flags(kvImageNoFlags))
                    vImageExtractChannel_ARGB8888(&srcVBuf, &gBuf, 1, vImage_Flags(kvImageNoFlags))
                    vImageExtractChannel_ARGB8888(&srcVBuf, &bBuf, 0, vImage_Flags(kvImageNoFlags))
                }
            }
        }

        var normalizationScale: Float = 1.0 / 255.0
        let dst = inputMutableData.mutableBytes.assumingMemoryBound(to: Float32.self)

        vDSP_vfltu8(rBytes, 1, dst + 0 * planeSize, 1, vDSP_Length(planeSize))
        vDSP_vsmul(dst + 0 * planeSize, 1, &normalizationScale, dst + 0 * planeSize, 1, vDSP_Length(planeSize))
        vDSP_vfltu8(gBytes, 1, dst + 1 * planeSize, 1, vDSP_Length(planeSize))
        vDSP_vsmul(dst + 1 * planeSize, 1, &normalizationScale, dst + 1 * planeSize, 1, vDSP_Length(planeSize))
        vDSP_vfltu8(bBytes, 1, dst + 2 * planeSize, 1, vDSP_Length(planeSize))
        vDSP_vsmul(dst + 2 * planeSize, 1, &normalizationScale, dst + 2 * planeSize, 1, vDSP_Length(planeSize))
    }

    private func populateCoreMLInputMultiArray(from pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        guard let inputMultiArray = coreMLInputMultiArray else {
            throw InferenceError.unsupportedCoreMLInput
        }

        try fillORTInputTensor(from: pixelBuffer)

        let floatCount = inputSize * inputSize * 3
        let src = inputMutableData.mutableBytes.assumingMemoryBound(to: Float32.self)
        let dst = inputMultiArray.dataPointer.assumingMemoryBound(to: Float32.self)
        dst.update(from: src, count: floatCount)
        return inputMultiArray
    }

    private func runORTInference(using pixelBuffer: CVPixelBuffer, mapping: PreprocessMapping) throws -> [Detection] {
        guard let session else {
            throw InferenceError.sessionNotReady
        }
        guard let inputTensor else {
            throw InferenceError.invalidORTInputTensor
        }

        try fillORTInputTensor(from: pixelBuffer)

        let outputs = try session.run(
            withInputs: [inputName: inputTensor],
            outputNames: Set([outputName]),
            runOptions: nil
        )

        guard let outputValue = outputs[outputName] else { return [] }
        let shapeInfo = try outputValue.tensorTypeAndShapeInfo()
        let shape = shapeInfo.shape.map { $0.intValue }

        if !hasLoggedShape {
            hasLoggedShape = true
            print("📐 \(outputName) shape: \(shape)")
        }

        let outputTensorData = try outputValue.tensorData()
        return parseDetections(fromTensorData: outputTensorData, shape: shape, mapping: mapping)
    }

    private func runCoreMLInference(using pixelBuffer: CVPixelBuffer, mapping: PreprocessMapping) throws -> [Detection] {
        guard let coreMLModel, let inputName = coreMLInputName, let outputName = coreMLOutputName else {
            throw InferenceError.sessionNotReady
        }

        let inputValue: MLFeatureValue
        if coreMLInputIsImage {
            inputValue = MLFeatureValue(pixelBuffer: pixelBuffer)
        } else {
            let inputMultiArray = try populateCoreMLInputMultiArray(from: pixelBuffer)
            inputValue = MLFeatureValue(multiArray: inputMultiArray)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: inputValue])
        let prediction = try coreMLModel.prediction(from: provider)
        if !hasLoggedCoreMLPredictionLayout {
            hasLoggedCoreMLPredictionLayout = true
            logCoreMLPredictionLayout(prediction, selectedOutputName: outputName)
        }
        guard let outputValue = prediction.featureValue(for: outputName),
              let outputMultiArray = outputValue.multiArrayValue else {
            throw InferenceError.unsupportedCoreMLOutput
        }

        return parseDetections(from: outputMultiArray, mapping: mapping)
    }

    private func logCoreMLPredictionLayout(_ prediction: MLFeatureProvider, selectedOutputName: String) {
        let sortedNames = prediction.featureNames.sorted()
        print("🧾 Core ML prediction features: \(sortedNames)")
        for name in sortedNames {
            guard let value = prediction.featureValue(for: name) else { continue }
            if let multiArray = value.multiArrayValue {
                let shape = multiArray.shape.map { $0.intValue }
                print("   feature \(name): multiArray shape=\(shape) type=\(multiArray.dataType.rawValue)")
            } else if value.type == .dictionary {
                print("   feature \(name): dictionary")
            } else {
                print("   feature \(name): type=\(value.type.rawValue)")
            }
        }
        print("🎯 Using Core ML prediction feature: \(selectedOutputName)")
    }

    // MARK: - Output Parsing

    /// Entrypoint for decoding raw tensor bytes into UI-facing detections.
    private func parseDetections(
        fromTensorData tensorData: NSMutableData,
        shape: [Int],
        mapping: PreprocessMapping
    ) -> [Detection] {
        guard tensorData.length > 0 else { return [] }
        let ptr = tensorData.bytes.assumingMemoryBound(to: Float32.self)
        return parseDetections(from: ptr, shape: shape, mapping: mapping)
    }

    private func parseDetections(
        from multiArray: MLMultiArray,
        mapping: PreprocessMapping
    ) -> [Detection] {
        let shape = multiArray.shape.map { $0.intValue }
        if let detections = parseNMSDetectionsIfAvailable(from: multiArray, shape: shape, mapping: mapping) {
            return detections
        }

        let scalarCount = shape.reduce(1, *)

        switch multiArray.dataType {
        case .float32:
            let ptr = multiArray.dataPointer.assumingMemoryBound(to: Float32.self)
            return parseDetections(from: ptr, shape: shape, mapping: mapping)
        case .double:
            if outputConversionBuffer.count < scalarCount {
                outputConversionBuffer = [Float32](repeating: 0, count: scalarCount)
            }
            let ptr = multiArray.dataPointer.assumingMemoryBound(to: Double.self)
            for index in 0..<scalarCount {
                outputConversionBuffer[index] = Float32(ptr[index])
            }
            return outputConversionBuffer.withUnsafeBufferPointer { buffer in
                guard let baseAddress = buffer.baseAddress else { return [] }
                return parseDetections(from: baseAddress, shape: shape, mapping: mapping)
            }
        case .float16:
            if outputConversionBuffer.count < scalarCount {
                outputConversionBuffer = [Float32](repeating: 0, count: scalarCount)
            }
            let ptr = multiArray.dataPointer.assumingMemoryBound(to: UInt16.self)
            for index in 0..<scalarCount {
                outputConversionBuffer[index] = Float32(Float16(bitPattern: ptr[index]))
            }
            return outputConversionBuffer.withUnsafeBufferPointer { buffer in
                guard let baseAddress = buffer.baseAddress else { return [] }
                return parseDetections(from: baseAddress, shape: shape, mapping: mapping)
            }
        default:
            return []
        }
    }

    private func parseNMSDetectionsIfAvailable(
        from multiArray: MLMultiArray,
        shape: [Int],
        mapping: PreprocessMapping
    ) -> [Detection]? {
        guard shape.count >= 2 else { return nil }

        let candidateWidth = shape[shape.count - 1]
        let candidateCount = shape[shape.count - 2]
        guard candidateWidth == 6, candidateCount > 0 else { return nil }

        let scalarCount = shape.reduce(1, *)
        guard scalarCount >= candidateCount * candidateWidth else { return nil }

        let values: [Float32]
        switch multiArray.dataType {
        case .float32:
            let ptr = multiArray.dataPointer.assumingMemoryBound(to: Float32.self)
            values = Array(UnsafeBufferPointer(start: ptr, count: scalarCount))
        case .double:
            let ptr = multiArray.dataPointer.assumingMemoryBound(to: Double.self)
            values = (0..<scalarCount).map { Float32(ptr[$0]) }
        case .float16:
            let ptr = multiArray.dataPointer.assumingMemoryBound(to: UInt16.self)
            values = (0..<scalarCount).map { Float32(Float16(bitPattern: ptr[$0])) }
        default:
            return nil
        }

        var detections: [Detection] = []
        detections.reserveCapacity(min(candidateCount, 64))

        for index in 0..<candidateCount {
            let base = index * candidateWidth
            let a = values[base + 0]
            let b = values[base + 1]
            let c = values[base + 2]
            let d = values[base + 3]
            let confidence = values[base + 4]
            let classId = Int(values[base + 5].rounded())

            guard confidence.isFinite, confidence >= confThreshold else { continue }
            guard classId >= 0 else { continue }

            appendCoreMLNMSDetection(
                a: a,
                b: b,
                c: c,
                d: d,
                classId: classId,
                confidence: confidence,
                mapping: mapping,
                to: &detections
            )
        }

        return detections
    }

    private func appendCoreMLNMSDetection(
        a: Float,
        b: Float,
        c: Float,
        d: Float,
        classId: Int,
        confidence: Float,
        mapping: PreprocessMapping,
        to detections: inout [Detection]
    ) {
        // Most Core ML YOLO exports with shape [1, N, 6] emit x1, y1, x2, y2, confidence, classId.
        var modelMinX = CGFloat(a)
        var modelMinY = CGFloat(b)
        var modelMaxX = CGFloat(c)
        var modelMaxY = CGFloat(d)

        // Fallback for exporters that emit cx, cy, w, h instead of xyxy.
        if modelMaxX <= modelMinX || modelMaxY <= modelMinY {
            let cx = CGFloat(a)
            let cy = CGFloat(b)
            let width = CGFloat(c)
            let height = CGFloat(d)
            modelMinX = cx - width * 0.5
            modelMinY = cy - height * 0.5
            modelMaxX = cx + width * 0.5
            modelMaxY = cy + height * 0.5
        }

        let coordsLookNormalized = max(modelMaxX, modelMaxY) <= 2.0
        if coordsLookNormalized {
            let x = max(0, min(1, modelMinX))
            let y = max(0, min(1, modelMinY))
            let maxX = max(0, min(1, modelMaxX))
            let maxY = max(0, min(1, modelMaxY))
            guard maxX > x, maxY > y else { return }

            detections.append(
                Detection(
                    bbox: CGRect(x: x, y: y, width: maxX - x, height: maxY - y),
                    classId: classId,
                    className: classId < COCO_CLASSES.count ? COCO_CLASSES[classId] : "unknown",
                    confidence: confidence,
                    maskCoeffs: []
                )
            )
            return
        }

        let sourceMinX = (modelMinX - mapping.offsetX) / mapping.scale
        let sourceMinY = (modelMinY - mapping.offsetY) / mapping.scale
        let sourceMaxX = (modelMaxX - mapping.offsetX) / mapping.scale
        let sourceMaxY = (modelMaxY - mapping.offsetY) / mapping.scale

        let x = max(0, min(1, sourceMinX / mapping.sourceWidth))
        let y = max(0, min(1, sourceMinY / mapping.sourceHeight))
        let maxX = max(0, min(1, sourceMaxX / mapping.sourceWidth))
        let maxY = max(0, min(1, sourceMaxY / mapping.sourceHeight))
        guard maxX > x, maxY > y else { return }

        detections.append(
            Detection(
                bbox: CGRect(x: x, y: y, width: maxX - x, height: maxY - y),
                classId: classId,
                className: classId < COCO_CLASSES.count ? COCO_CLASSES[classId] : "unknown",
                confidence: confidence,
                maskCoeffs: []
            )
        )
    }

    /// Supports both `[C, N]` and `[N, C]` layouts emitted by YOLO-family models.
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

    /// Parses a contiguous `[anchor][channel]` tensor layout into `Detection` values.
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

    /// Maps one decoded YOLO box from model space back into normalized image space.
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

    // MARK: - Non-Max Suppression

    /// Removes highly overlapping detections while keeping the highest-confidence box.
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

    /// Computes intersection-over-union for suppression decisions.
    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }

        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea

        return unionArea > 0 ? interArea / unionArea : 0
    }

}

extension InferenceService.Configuration.ExecutionProfile: CustomStringConvertible {
    var description: String {
        switch self {
        case .cpuOnly:
            return "cpuOnly"
        case .cpuAndNeuralEngine:
            return "cpuAndNeuralEngine"
        case .all:
            return "all"
        }
    }

    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .cpuOnly:
            return .cpuOnly
        case .cpuAndNeuralEngine:
            return .cpuAndNeuralEngine
        case .all:
            return .all
        }
    }
}

// MARK: - Result Types

/// Result returned to the view model after a single inference pass.
struct InferenceResult {
    let detections: [Detection]
    let inferenceMs: Double
}

// MARK: - Errors

/// Errors surfaced when the inference pipeline is not ready to execute.
enum InferenceError: LocalizedError {
    case sessionNotReady
    case serviceDeallocated
    case invalidPreprocessedBuffer
    case invalidORTInputTensor
    case unsupportedCoreMLInput
    case unsupportedCoreMLOutput

    var errorDescription: String? {
        switch self {
        case .sessionNotReady:
            return "Inference session 尚未初始化完成"
        case .serviceDeallocated:
            return "InferenceService 已被釋放"
        case .invalidPreprocessedBuffer:
            return "前處理後的影像 buffer 無效"
        case .invalidORTInputTensor:
            return "ORT input tensor 尚未初始化完成"
        case .unsupportedCoreMLInput:
            return "Core ML 模型輸入格式目前不支援"
        case .unsupportedCoreMLOutput:
            return "Core ML 模型輸出格式目前不支援"
        }
    }
}
