//
//  ARCameraService.swift
//  iDriveBot
//

import ARKit
import Combine
import CoreVideo
import QuartzCore
import simd

final class ARCameraService: NSObject {

    // MARK: - Public session

    let session = ARSession()

    // MARK: - Direct frame callback
    var onFrame: ((_ pixelBuffer: CVPixelBuffer, _ context: ARFrameContext) -> Void)?

    // MARK: - Mesh publisher (no retention issue)

    let meshAnchorPublisher = PassthroughSubject<[ARMeshAnchor], Never>()

    // MARK: - Thread-safe processing gate

    private let gate = NSLock()
    private var _isProcessing = false
    private var _gateLockedAt: CFTimeInterval = 0
    private let delegateQueue = DispatchQueue(label: "com.idrivebot.arkit.session.delegate", qos: .userInteractive)

    private func tryBeginProcessing() -> Bool {
        gate.lock()
        defer { gate.unlock() }
        guard !_isProcessing else { return false }
        _isProcessing = true
        _gateLockedAt = CACurrentMediaTime()
        return true
    }

    /// Release the gate so the next frame can enter. Safe from any thread.
    func endProcessing() {
        gate.lock()
        let held = CACurrentMediaTime() - _gateLockedAt
        _isProcessing = false
        gate.unlock()
        FrameDebug.shared.recordGateHeld(held)
    }

    // MARK: - Debug: frame retention tracker
    class FrameDebug {
        static let shared = FrameDebug()

        private let lock = NSLock()
        private var delegateCalls = 0
        private var gateDrops = 0
        private var accepted = 0
        private var copyFails = 0
        private var callbackNil = 0
        private var lastReportTime: CFTimeInterval = 0
        private var lastGateHeld: CFTimeInterval = 0
        private var maxGateHeld: CFTimeInterval = 0
        private var totalCopyMs: Double = 0

        func recordDelegateCall() {
            lock.lock()
            delegateCalls += 1
            lock.unlock()
        }

        func recordGateDrop() {
            lock.lock()
            gateDrops += 1
            lock.unlock()
        }

        func recordAccepted() {
            lock.lock()
            accepted += 1
            lock.unlock()
        }

        func recordCopyFail() {
            lock.lock()
            copyFails += 1
            lock.unlock()
        }

        func recordCallbackNil() {
            lock.lock()
            callbackNil += 1
            lock.unlock()
        }

        func recordCopyTime(_ ms: Double) {
            lock.lock()
            totalCopyMs += ms
            lock.unlock()
        }

        func recordGateHeld(_ seconds: CFTimeInterval) {
            lock.lock()
            lastGateHeld = seconds
            maxGateHeld = max(maxGateHeld, seconds)
            lock.unlock()
        }

        /// Call from delegate. Prints a summary every 2 seconds.
        func reportIfNeeded() {
            let now = CACurrentMediaTime()
            lock.lock()
            guard now - lastReportTime >= 2.0 else {
                lock.unlock()
                return
            }
            let d = delegateCalls, g = gateDrops, a = accepted, cf = copyFails, cn = callbackNil
            let gh = lastGateHeld, mgh = maxGateHeld
            let avgCopy = a > 0 ? totalCopyMs / Double(a) : 0

            // reset
            delegateCalls = 0; gateDrops = 0; accepted = 0
            copyFails = 0; callbackNil = 0
            totalCopyMs = 0; maxGateHeld = 0
            lastReportTime = now
            lock.unlock()

            let thread = Thread.isMainThread ? "main" : "bg"
//            print("""
//            [FrameDebug] 2s window | thread=\(thread)
//              delegate=\(d) gateDrop=\(g) accepted=\(a) copyFail=\(cf) callbackNil=\(cn)
//              lastGateHeld=\(String(format: "%.1fms", gh*1000)) maxGateHeld=\(String(format: "%.1fms", mgh*1000))
//              avgCopy=\(String(format: "%.1fms", avgCopy))
//            """)
        }
    }

    // MARK: - Private state

    private var currentMeshAnchors: [UUID: ARMeshAnchor] = [:]
    private var pendingMeshAnchorRemovals: [UUID: TimeInterval] = [:]
    private var lastMeshPublishTime: TimeInterval = 0
    private let meshPublishInterval: TimeInterval = 0.45
    private let meshAnchorRemovalGracePeriod: TimeInterval = 2.0

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = delegateQueue
    }

    // MARK: - Lifecycle

    func start(reset: Bool = true) {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.planeDetection = [.horizontal, .vertical]

        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }

        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
            config.sceneReconstruction = .meshWithClassification
        } else if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
        }

        if let format = ARWorldTrackingConfiguration.supportedVideoFormats
            .sorted(by: { $0.imageResolution.width < $1.imageResolution.width })
            .first(where: { $0.imageResolution.width >= 1280 }) {
            config.videoFormat = format
        }

        let options: ARSession.RunOptions = reset ? [.resetTracking, .removeExistingAnchors] : []
        session.run(config, options: options)
    }

    func stop() {
        session.pause()
        currentMeshAnchors.removeAll()
        pendingMeshAnchorRemovals.removeAll()
        endProcessing()
    }

    // MARK: - Pixel buffer copy

    /// Create an independent copy so the source ARFrame can be released immediately.
    static func copyPixelBuffer(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(src)
        let height = CVPixelBufferGetHeight(src)
        let format = CVPixelBufferGetPixelFormatType(src)

        let attrs: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        var dst: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attrs as CFDictionary, &dst) == kCVReturnSuccess,
              let dst else { return nil }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
            CVPixelBufferUnlockBaseAddress(dst, [])
        }

        let planeCount = CVPixelBufferGetPlaneCount(src)
        if planeCount > 0 {
            for plane in 0..<planeCount {
                guard let srcAddr = CVPixelBufferGetBaseAddressOfPlane(src, plane),
                      let dstAddr = CVPixelBufferGetBaseAddressOfPlane(dst, plane) else { continue }
                let srcBPR = CVPixelBufferGetBytesPerRowOfPlane(src, plane)
                let dstBPR = CVPixelBufferGetBytesPerRowOfPlane(dst, plane)
                let rows = CVPixelBufferGetHeightOfPlane(src, plane)
                let copyBytes = min(srcBPR, dstBPR)
                for row in 0..<rows {
                    memcpy(dstAddr.advanced(by: row * dstBPR),
                           srcAddr.advanced(by: row * srcBPR),
                           copyBytes)
                }
            }
        } else {
            guard let srcAddr = CVPixelBufferGetBaseAddress(src),
                  let dstAddr = CVPixelBufferGetBaseAddress(dst) else { return nil }
            let bpr = CVPixelBufferGetBytesPerRow(src)
            memcpy(dstAddr, srcAddr, bpr * height)
        }

        return dst
    }

    // MARK: - Mesh publishing

    private func publishMeshAnchorsIfNeeded(timestamp: TimeInterval) {
        expireRemovedMeshAnchors(at: timestamp)
        guard timestamp - lastMeshPublishTime >= meshPublishInterval else { return }
        lastMeshPublishTime = timestamp
        meshAnchorPublisher.send(Array(currentMeshAnchors.values))
    }

    private func expireRemovedMeshAnchors(at timestamp: TimeInterval) {
        guard !pendingMeshAnchorRemovals.isEmpty else { return }

        let expiredIDs = pendingMeshAnchorRemovals.compactMap { id, removalTime in
            timestamp - removalTime >= meshAnchorRemovalGracePeriod ? id : nil
        }

        guard !expiredIDs.isEmpty else { return }
        for id in expiredIDs {
            pendingMeshAnchorRemovals.removeValue(forKey: id)
            currentMeshAnchors.removeValue(forKey: id)
        }
    }

    /// Extracts scene depth metadata without copying the depth map pixel data.
    /// CVPixelBuffer has independent ref-counting — retaining it does NOT retain ARFrame.
    private static func copySceneDepth(from frame: ARFrame) -> ARFrameContext.SceneDepthData? {
        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else {
            return nil
        }

        let depthMap = depthData.depthMap
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)

        let imageResolution = frame.camera.imageResolution
        let scaleX = Float(width) / Float(imageResolution.width)
        let scaleY = Float(height) / Float(imageResolution.height)
        var depthIntrinsics = frame.camera.intrinsics
        depthIntrinsics.columns.0.x *= scaleX
        depthIntrinsics.columns.1.y *= scaleY
        depthIntrinsics.columns.2.x *= scaleX
        depthIntrinsics.columns.2.y *= scaleY

        return ARFrameContext.SceneDepthData(
            depthMap: depthMap,
            resolution: CGSize(width: width, height: height),
            intrinsics: depthIntrinsics
        )
    }
}

// MARK: - ARSessionDelegate

extension ARCameraService: ARSessionDelegate {

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let dbg = FrameDebug.shared
        dbg.recordDelegateCall()
        dbg.reportIfNeeded()

        // Atomic gate: drop frame if previous one is still being processed
        guard tryBeginProcessing() else {
            dbg.recordGateDrop()
            return
        }

        dbg.recordAccepted()

        // Lightweight context (no CVPixelBuffer -> no ARFrame retention)
        let context = ARFrameContext(
            cameraTransform: frame.camera.transform,
            intrinsics: frame.camera.intrinsics,
            imageResolution: CGSize(
                width: CVPixelBufferGetWidth(frame.capturedImage),
                height: CVPixelBufferGetHeight(frame.capturedImage)
            ),
            sceneDepth: Self.copySceneDepth(from: frame),
            timestamp: frame.timestamp
        )

        // Copy pixel buffer so ARFrame is released when this method returns
        let t0 = CACurrentMediaTime()
        guard let copiedBuffer = Self.copyPixelBuffer(frame.capturedImage) else {
            dbg.recordCopyFail()
            endProcessing()
            return
        }
        let t1 = CACurrentMediaTime()
        dbg.recordCopyTime((t1 - t0) * 1000)

        // Safety: if no handler, release gate immediately
        guard let handler = onFrame else {
            dbg.recordCallbackNil()
            endProcessing()
            return
        }

        // Deliver to handler
        handler(copiedBuffer, context)

        publishMeshAnchorsIfNeeded(timestamp: frame.timestamp)
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for anchor in anchors {
            if let mesh = anchor as? ARMeshAnchor {
                currentMeshAnchors[mesh.identifier] = mesh
                pendingMeshAnchorRemovals.removeValue(forKey: mesh.identifier)
            }
        }
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for anchor in anchors {
            if let mesh = anchor as? ARMeshAnchor {
                currentMeshAnchors[mesh.identifier] = mesh
                pendingMeshAnchorRemovals.removeValue(forKey: mesh.identifier)
            }
        }
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        let removalTime = CACurrentMediaTime()
        for anchor in anchors {
            pendingMeshAnchorRemovals[anchor.identifier] = removalTime
        }
    }
}
