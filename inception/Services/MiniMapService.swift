//
//  MiniMapService.swift
//  iDriveBot
//
//  Clean 2.5D minimap service
//

import Foundation
import SceneKit
import ARKit
import simd
import UIKit

@MainActor
final class MiniMapService {

    enum PresentationMode: Equatable {
        case compact
        case expanded
    }

    // MARK: - Public scene

    let scene: SCNScene = SCNScene()

    // MARK: - Scene graph

    /// Screen-space pan root used only for expanded minimap navigation.
    private let panRootNode = SCNNode()

    /// Rotation root: rotates map with phone yaw
    private let rotationRootNode = SCNNode()

    /// Translation root: keeps user centered
    private let translationRootNode = SCNNode()

    private let environmentRootNode = SCNNode()
    private let routeRootNode = SCNNode()
    private let occupancyGridRootNode = SCNNode()
    private let objectsRootNode = SCNNode()

    /// User marker and heading live outside world transform tree,
    /// so they always stay fixed at minimap center.
    private let userMarkerNode = SCNNode()
    private let headingNode = SCNNode()

    private let cameraRigNode = SCNNode()
    private let cameraNode = SCNNode()
    private let lightNode = SCNNode()

    // MARK: - Cached nodes

    private var objectNodes: [Int: SCNNode] = [:]
    private var meshNodes: [UUID: SCNNode] = [:]
    private var lastMeshRefreshTime: [UUID: TimeInterval] = [:]
    /// Background queue for expensive SCNGeometry construction.
    private let geometryQueue = DispatchQueue(
        label: "inception.minimap.geometry",
        qos: .utility,
        attributes: .concurrent
    )
    private var landmarkNodes: [UUID: SCNNode] = [:]
    private let landmarksRootNode = SCNNode()
    private var selectedLandmarkID: UUID?

    // MARK: - State

    private var lastUserPosition = simd_float3.zero
    private var lastYaw: Float = 0

    /// Latest smoothed world-space user position (XZ plane).
    var userPosition: simd_float3 { lastUserPosition }

    /// Latest smoothed user heading in radians (angle from +Z axis, with landscape offset).
    var userHeading: Float { lastYaw }
    private var hasSmoothedCameraState = false
    private var presentationMode: PresentationMode = .compact
    private var expandedZoomScale: Double = 1.0
    private var expandedPanOffset = SIMD2<Float>(repeating: 0)

    // MARK: - Config

    /// If a face normal is too close to world up/down, we treat it as a horizontal surface.
    /// Raising this value keeps more slanted surfaces; lowering removes more near-horizontal faces.
    private let horizontalFaceThreshold: Float = 0.94
    private let meshRefreshInterval: TimeInterval = 0.8
    private let maxFacesPerAnchor = 2_500
    private let positionSmoothingAlpha: Float = 0.18
    private let yawSmoothingAlpha: Float = 0.14
    private let compactOrthographicScale: Double = 3.2
    private let expandedOrthographicScale: Double = 5.0
    private let minimumExpandedZoomScale: Double = 0.55
    private let maximumExpandedZoomScale: Double = 2.4

    // MARK: - Init

    init() {
        scene.rootNode.addChildNode(panRootNode)
        panRootNode.addChildNode(rotationRootNode)
        rotationRootNode.addChildNode(translationRootNode)

        translationRootNode.addChildNode(environmentRootNode)
        translationRootNode.addChildNode(occupancyGridRootNode)
        translationRootNode.addChildNode(routeRootNode)
        translationRootNode.addChildNode(landmarksRootNode)
        translationRootNode.addChildNode(objectsRootNode)

        panRootNode.addChildNode(userMarkerNode)
        panRootNode.addChildNode(headingNode)

        scene.rootNode.addChildNode(cameraRigNode)
        cameraRigNode.addChildNode(cameraNode)

        scene.rootNode.addChildNode(lightNode)

        setupScene()
        setupUserMarker()
        setupHeadingMarker()
        setupCamera()
        setupLight()
    }

    // MARK: - Setup

    private func setupScene() {
        scene.background.contents = UIColor(white: 0.96, alpha: 0.96)
    }

    private func setupUserMarker() {
        userMarkerNode.geometry = nil
        userMarkerNode.childNodes.forEach { $0.removeFromParentNode() }

        let blueMaterial = makeMaterial(
            diffuse: UIColor(red: 0.16, green: 0.53, blue: 0.98, alpha: 1.0),
            emission: .clear
        )

        let whiteMaterial = makeMaterial(
            diffuse: .white,
            emission: .clear
        )

        // 白色外框：高一點
        let outerRing = SCNTube(innerRadius: 0.138, outerRadius: 0.18, height: 0.020)
        outerRing.radialSegmentCount = 64
        outerRing.materials = Array(repeating: whiteMaterial, count: 3)
        let outerNode = SCNNode(geometry: outerRing)
        outerNode.position = SCNVector3(0, 0.010, 0)

        // 藍色底：略低一點
        let inner = SCNCylinder(radius: 0.138, height: 0.012)
        inner.radialSegmentCount = 64
        inner.materials = Array(repeating: blueMaterial, count: 3)
        let innerNode = SCNNode(geometry: inner)
        innerNode.position = SCNVector3(0, 0.006, 0)

        // Apple 風格箭頭：中後段內縮
        let arrowPath = UIBezierPath()
        
        // top tip
        arrowPath.move(to: CGPoint(x: 0.0, y: 0.102))
        
        // right shoulder
        arrowPath.addLine(to: CGPoint(x: 0.084, y: -0.082))
        
        // mid center
        arrowPath.addLine(to: CGPoint(x: 0.00, y: -0.004))
        
        // left shoulder
        arrowPath.addLine(to: CGPoint(x: -0.084, y: -0.082))

        arrowPath.close()

        // 箭頭厚度拉高，接近白框高度
        let arrow = SCNShape(path: arrowPath, extrusionDepth: 0.020)
        arrow.chamferRadius = 0.0018
        arrow.materials = [whiteMaterial]

        let arrowNode = SCNNode(geometry: arrow)
        arrowNode.eulerAngles.x = -.pi / 2

        // 箭頭底部貼近藍底上方，但整體高度和白框接近
        arrowNode.position = SCNVector3(0, 0.0105, 0)

        userMarkerNode.addChildNode(outerNode)
        userMarkerNode.addChildNode(innerNode)
        userMarkerNode.addChildNode(arrowNode)

        userMarkerNode.position = SCNVector3Zero
        userMarkerNode.scale = SCNVector3(2, 2, 2)
    }

    private func setupHeadingMarker() {
        headingNode.childNodes.forEach { $0.removeFromParentNode() }
        headingNode.position = SCNVector3Zero
        headingNode.eulerAngles = SCNVector3Zero
    }

    private func setupCamera() {
        let camera = SCNCamera()
        camera.usesOrthographicProjection = true
        camera.orthographicScale = 3.4
        camera.zNear = 0.01
        camera.zFar = 100
        camera.wantsHDR = false

        cameraNode.camera = camera
        updateMiniMapCamera()
    }

    private func setupLight() {
        let light = SCNLight()
        light.type = .omni
        light.intensity = 900
        light.color = UIColor.white

        lightNode.light = light
        lightNode.position = SCNVector3(0, 6, 4)
    }

    // MARK: - Public API

    /// User always stays centered in minimap.
    /// Map rotates with user yaw.
    func updateCamera(transform: simd_float4x4, orientation: AppOrientation) {
        let position = simd_float3(
            transform.columns.3.x,
            transform.columns.3.y,
            transform.columns.3.z
        )

        let rawYaw = yawFromTransform(transform)

        // Your confirmed correct landscape compensation
        let landscapeOffset: Float = .pi
        let flippedOffset: Float = orientation.isFlipped ? .pi : 0
        let targetYaw = rawYaw + landscapeOffset + flippedOffset

        if hasSmoothedCameraState {
            lastUserPosition = simd_mix(lastUserPosition, position, simd_float3(repeating: positionSmoothingAlpha))
            lastYaw = interpolateAngle(from: lastYaw, to: targetYaw, alpha: yawSmoothingAlpha)
        } else {
            lastUserPosition = position
            lastYaw = targetYaw
            hasSmoothedCameraState = true
        }

        // User always fixed at minimap center
        userMarkerNode.position = SCNVector3(0, 0.08, 0)
        headingNode.position = SCNVector3(0, 0.08, 0)
        headingNode.eulerAngles = SCNVector3Zero

        // Translate world so user is centered
        translationRootNode.position = SCNVector3(-lastUserPosition.x, 0, -lastUserPosition.z)

        // Rotate map so minimap top = phone facing direction
        rotationRootNode.eulerAngles = SCNVector3(0, -lastYaw, 0)

        updateMiniMapCamera()
    }

    func updateTrackedObjects(_ objects: [TrackedObject]) {
        let visibleIds = Set(objects.map(\.id))

        for object in objects {
            guard let world = object.worldPosition else { continue }

            let node: SCNNode
            if let existingNode = objectNodes[object.id],
               existingNode.name == object.detection.className {
                node = existingNode
            } else {
                objectNodes[object.id]?.removeFromParentNode()
                node = makeObjectNode(for: object)
            }
            objectNodes[object.id] = node

            // Keep object in world coordinates; relative transform is handled by root nodes
            node.position = SCNVector3(world.x, 0.12, world.z)

            if node.parent !== objectsRootNode {
                node.removeFromParentNode()
                objectsRootNode.addChildNode(node)
            }
        }

        for (id, node) in objectNodes where !visibleIds.contains(id) {
            node.removeFromParentNode()
        }
    }

    /// Updates persistent landmark pins on the minimap.
    /// Landmarks remain visible even when the camera moves away.
    func updateLandmarks(_ landmarks: [Landmark]) {
        let currentIds = Set(landmarks.map(\.id))

        for landmark in landmarks {
            let node: SCNNode
            if let existing = landmarkNodes[landmark.id] {
                node = existing
            } else {
                landmarkNodes[landmark.id]?.removeFromParentNode()
                node = makeLandmarkNode(for: landmark)
            }
            landmarkNodes[landmark.id] = node
            node.position = SCNVector3(landmark.worldPosition.x, 0.12, landmark.worldPosition.z)
            if node.parent !== landmarksRootNode {
                node.removeFromParentNode()
                landmarksRootNode.addChildNode(node)
            }
            applySelectionStyle(to: node, isSelected: landmark.id == selectedLandmarkID)
        }

        for (id, node) in landmarkNodes where !currentIds.contains(id) {
            node.removeFromParentNode()
            landmarkNodes.removeValue(forKey: id)
        }
    }

    func setPresentationMode(_ mode: PresentationMode) {
        guard presentationMode != mode else { return }
        presentationMode = mode
        if mode == .compact {
            expandedPanOffset = .zero
        }
        updateMiniMapCamera()
    }

    func setExpandedZoomScale(_ scale: Double) {
        let clampedScale = min(max(scale, minimumExpandedZoomScale), maximumExpandedZoomScale)
        guard abs(clampedScale - expandedZoomScale) > 0.001 else { return }
        expandedZoomScale = clampedScale

        guard presentationMode == .expanded else { return }
        updateMiniMapCamera()
    }

    func panExpandedMap(byScreenTranslation translation: CGPoint, viewportSize: CGSize) {
        guard presentationMode == .expanded else { return }
        guard viewportSize.width > 0, viewportSize.height > 0 else { return }

        let visibleHeight = Float((expandedOrthographicScale / expandedZoomScale) * 2)
        let visibleWidth = visibleHeight * Float(viewportSize.width / viewportSize.height)

        let worldDeltaX = Float(translation.x / viewportSize.width) * visibleWidth
        let worldDeltaZ = Float(translation.y / viewportSize.height) * visibleHeight

        expandedPanOffset.x += worldDeltaX
        expandedPanOffset.y += worldDeltaZ
        updateMiniMapCamera()
    }

    func landmarkID(at point: CGPoint, in view: SCNView) -> UUID? {
        let hitResults = view.hitTest(point, options: nil)

        for result in hitResults {
            var node: SCNNode? = result.node
            while let current = node {
                if let landmarkID = landmarkID(from: current.name) {
                    return landmarkID
                }
                node = current.parent
            }
        }

        return nil
    }

    func setSelectedLandmarkID(_ id: UUID?) {
        guard selectedLandmarkID != id else { return }
        selectedLandmarkID = id
        refreshLandmarkSelectionAppearance()
    }

    func updateNavigationRoute(_ points: [simd_float3], targetLandmarkID: UUID?) {
        routeRootNode.childNodes.forEach { $0.removeFromParentNode() }

        guard points.count >= 2 else { return }

        for index in 0..<(points.count - 1) {
            let segmentNode = makeRouteSegmentNode(from: points[index], to: points[index + 1])
            routeRootNode.addChildNode(segmentNode)
        }

        if let lastPoint = points.last {
            let endpointNode = makeRouteEndpointNode(at: lastPoint)
            routeRootNode.addChildNode(endpointNode)
        }
    }

    // MARK: - Occupancy Grid Debug Overlay

    /// Renders the current A* occupancy grid as a flat texture overlay on the minimap floor.
    /// Green = confirmed walkable floor (exploredCells).
    /// Red   = obstacle + safety-buffer cells (dilatedCells).
    func updateOccupancyGrid(_ snapshot: NavigationService.OccupancySnapshot) {
        occupancyGridRootNode.childNodes.forEach { $0.removeFromParentNode() }

        let allCells = snapshot.exploredCells + snapshot.obstacleCells
        guard !allCells.isEmpty else { return }

        let xs = allCells.map { $0.x }
        let zs = allCells.map { $0.z }
        guard let minX = xs.min(), let maxX = xs.max(),
              let minZ = zs.min(), let maxZ = zs.max() else { return }

        let cols = maxX - minX + 1
        let rows = maxZ - minZ + 1

        // Scale down so the texture stays within 512×512 px
        let maxTex = 512
        let scale  = max(1, max((cols + maxTex - 1) / maxTex,
                                (rows + maxTex - 1) / maxTex))
        let texW = max(1, (cols + scale - 1) / scale)
        let texH = max(1, (rows + scale - 1) / scale)

        // Capture value types before going to background queue
        let exploredCells = snapshot.exploredCells
        let obstacleCells = snapshot.obstacleCells
        let cs = snapshot.cellSize

        // Build the texture on a background thread (CGContext is thread-safe),
        // then add the SceneKit node on the main thread.
        geometryQueue.async { [weak self] in
            guard let self else { return }

            // Build image with CGContext directly (thread-safe, no UIKit needed)
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
            guard let cgCtx = CGContext(
                data: nil, width: texW, height: texH,
                bitsPerComponent: 8, bytesPerRow: texW * 4,
                space: colorSpace, bitmapInfo: bitmapInfo.rawValue
            ) else { return }

            // 1. Fill entire canvas with "unexplored = blocked" light red.
            //    Any cell not explicitly drawn green or dark-red will show this colour,
            //    meaning areas the user hasn't scanned are treated as impassable.
            cgCtx.setFillColor(UIColor(red: 0.75, green: 0.20, blue: 0.20, alpha: 0.28).cgColor)
            cgCtx.fill(CGRect(x: 0, y: 0, width: texW, height: texH))

            func textureRow(for z: Int) -> Int {
                let row = (z - minZ) / scale
                // After rotating the plane onto the ground, image-space +Y points toward
                // world-space -Z. Flip rows so the occupancy overlay matches AR world Z.
                return texH - 1 - row
            }

            // 2. Walkable floor — bright green (cells A* can actually route through)
            cgCtx.setFillColor(UIColor(red: 0.18, green: 0.85, blue: 0.38, alpha: 0.60).cgColor)
            for c in exploredCells {
                cgCtx.fill(CGRect(x: (c.x - minX) / scale,
                                  y: textureRow(for: c.z),
                                  width: 1, height: 1))
            }

            // 3. Confirmed wall cells — brighter/darker red (overwrites green if overlap)
            cgCtx.setFillColor(UIColor(red: 0.92, green: 0.18, blue: 0.18, alpha: 0.70).cgColor)
            for c in obstacleCells {
                cgCtx.fill(CGRect(x: (c.x - minX) / scale,
                                  y: textureRow(for: c.z),
                                  width: 1, height: 1))
            }

            guard let cgImage = cgCtx.makeImage() else { return }

            let worldW  = Float(cols) * cs
            let worldD  = Float(rows) * cs
            let worldCX = Float(minX) * cs + worldW * 0.5
            let worldCZ = Float(minZ) * cs + worldD * 0.5

            let plane = SCNPlane(width: CGFloat(worldW), height: CGFloat(worldD))
            let mat = SCNMaterial()
            mat.diffuse.contents = cgImage   // CGImage is thread-safe
            mat.isDoubleSided = true
            mat.lightingModel = .constant
            mat.blendMode = .alpha
            plane.materials = [mat]

            let node = SCNNode(geometry: plane)
            node.eulerAngles = SCNVector3(-Float.pi / 2, 0, 0)
            node.position    = SCNVector3(worldCX, 0.015, worldCZ)

            DispatchQueue.main.async { [weak self] in
                self?.occupancyGridRootNode.addChildNode(node)
            }
        }
    }

    /// Removes the occupancy grid overlay from the minimap.
    func clearOccupancyGrid() {
        occupancyGridRootNode.childNodes.forEach { $0.removeFromParentNode() }
    }

    func updateMeshAnchors(_ anchors: [ARMeshAnchor]) {
        let currentIds = Set(anchors.map(\.identifier))
        let now = CACurrentMediaTime()

        for anchor in anchors {
            let node = meshNodes[anchor.identifier] ?? SCNNode()
            let needsGeometryRefresh: Bool

            if meshNodes[anchor.identifier] == nil {
                needsGeometryRefresh = true
            } else if let lastRefresh = lastMeshRefreshTime[anchor.identifier] {
                needsGeometryRefresh = now - lastRefresh >= meshRefreshInterval
            } else {
                needsGeometryRefresh = true
            }

            node.simdTransform = anchor.transform

            if node.parent !== environmentRootNode, node.geometry != nil {
                node.removeFromParentNode()
                environmentRootNode.addChildNode(node)
            }

            guard needsGeometryRefresh else {
                meshNodes[anchor.identifier] = node
                continue
            }

            if let geometry = makeStructureGeometry(from: anchor) {
                node.geometry = geometry

                if node.parent !== environmentRootNode {
                    node.removeFromParentNode()
                    environmentRootNode.addChildNode(node)
                }

                meshNodes[anchor.identifier] = node
                lastMeshRefreshTime[anchor.identifier] = now
            } else if node.geometry != nil {
                // Geometry refresh returned nil (sampled faces were all horizontal this cycle).
                // Keep the existing geometry visible — do NOT remove the node.
                // Omitting lastMeshRefreshTime update so this anchor retries on the next publish.
                meshNodes[anchor.identifier] = node
            } else {
                // No existing geometry and couldn't build one — discard.
                node.removeFromParentNode()
                meshNodes.removeValue(forKey: anchor.identifier)
                lastMeshRefreshTime.removeValue(forKey: anchor.identifier)
            }
        }

        for (id, node) in meshNodes where !currentIds.contains(id) {
            node.removeFromParentNode()
            meshNodes.removeValue(forKey: id)
            lastMeshRefreshTime.removeValue(forKey: id)
        }
    }

    // MARK: - Camera logic

    /// Fixed 2.5D view, looking at minimap center.
    private func updateMiniMapCamera() {
        switch presentationMode {
        case .compact:
            cameraNode.camera?.orthographicScale = compactOrthographicScale
            cameraRigNode.position = SCNVector3(0, 2.4, 3.2)
            cameraNode.look(at: SCNVector3(0, 0.2, 0))
            panRootNode.position = SCNVector3Zero

        case .expanded:
            cameraNode.camera?.orthographicScale = expandedOrthographicScale / expandedZoomScale
            cameraRigNode.position = SCNVector3(0, 4.8, 6.4)
            cameraNode.look(at: SCNVector3(0, 0.15, 0))
            panRootNode.position = SCNVector3(expandedPanOffset.x, 0, expandedPanOffset.y)
        }
    }

    /// Only horizontal yaw matters. Ignore pitch/roll completely.
    private func yawFromTransform(_ transform: simd_float4x4) -> Float {
        let forward = simd_float3(
            -transform.columns.2.x,
            0,
            -transform.columns.2.z
        )

        let normalized = simd_normalize(forward)
        return atan2(normalized.x, normalized.z)
    }

    private func interpolateAngle(from current: Float, to target: Float, alpha: Float) -> Float {
        let delta = shortestAngleDelta(from: current, to: target)
        return normalizeAngle(current + delta * alpha)
    }

    private func shortestAngleDelta(from current: Float, to target: Float) -> Float {
        normalizeAngle(target - current)
    }

    private func normalizeAngle(_ angle: Float) -> Float {
        var value = angle
        while value > .pi { value -= 2 * .pi }
        while value < -.pi { value += 2 * .pi }
        return value
    }

    // MARK: - Object rendering

    private func makeObjectNode(for object: TrackedObject) -> SCNNode {
        let node = SCNNode()
        node.name = object.detection.className
        let accentColor = accentColor(for: object.detection.className)

        let base = SCNCylinder(radius: 0.12, height: 0.03)
        base.radialSegmentCount = 48
        base.materials = Array(repeating: makeFlatMaterial(color: accentColor.withAlphaComponent(0.95)), count: 3)

        let baseNode = SCNNode(geometry: base)
        baseNode.position = SCNVector3(0, 0.015, 0)
        node.addChildNode(baseNode)

        let ring = SCNTube(innerRadius: 0.12, outerRadius: 0.145, height: 0.012)
        ring.radialSegmentCount = 48
        ring.materials = Array(repeating: makeFlatMaterial(color: UIColor.white.withAlphaComponent(0.95)), count: 3)

        let ringNode = SCNNode(geometry: ring)
        ringNode.position = SCNVector3(0, 0.036, 0)
        node.addChildNode(ringNode)

        let iconPlane = SCNPlane(width: 0.52, height: 0.52)
        iconPlane.cornerRadius = 0.08
        iconPlane.firstMaterial = makeSpriteMaterial(
            image: iconImage(for: object.detection.className, tintColor: accentColor)
        )

        let iconNode = SCNNode(geometry: iconPlane)
        iconNode.name = "icon"
        iconNode.position = SCNVector3(0, 0.40, 0)
        iconNode.constraints = [SCNBillboardConstraint()]
        node.addChildNode(iconNode)

        let stem = SCNCylinder(radius: 0.012, height: 0.18)
        stem.materials = Array(repeating: makeFlatMaterial(color: accentColor.withAlphaComponent(0.7)), count: 3)

        let stemNode = SCNNode(geometry: stem)
        stemNode.position = SCNVector3(0, 0.13, 0)
        node.addChildNode(stemNode)

        return node
    }

    /// Landmark pin: hexagonal base + faded icon billboard.
    /// Visually distinct from live tracked objects (circular base, full opacity).
    private func makeLandmarkNode(for landmark: Landmark) -> SCNNode {
        let node = SCNNode()
        node.name = landmarkNodeName(for: landmark.id)
        let accent = accentColor(for: landmark.className)

        // Hexagonal ring base (6 segments) — distinguishes landmark from live circle
        let ring = SCNTube(innerRadius: 0.09, outerRadius: 0.14, height: 0.012)
        ring.radialSegmentCount = 6
        ring.materials = Array(repeating: makeFlatMaterial(color: accent.withAlphaComponent(0.55)), count: 3)
        let ringNode = SCNNode(geometry: ring)
        ringNode.position = SCNVector3(0, 0.006, 0)
        node.addChildNode(ringNode)

        // Thin stem
        let stem = SCNCylinder(radius: 0.010, height: 0.17)
        stem.materials = Array(repeating: makeFlatMaterial(color: accent.withAlphaComponent(0.45)), count: 3)
        let stemNode = SCNNode(geometry: stem)
        stemNode.position = SCNVector3(0, 0.11, 0)
        node.addChildNode(stemNode)

        // Faded icon billboard — same icon as live objects, lower opacity
        let iconPlane = SCNPlane(width: 0.46, height: 0.46)
        iconPlane.cornerRadius = 0.08
        let iconMat = makeSpriteMaterial(
            image: iconImage(for: landmark.className, tintColor: accent)
        )
        iconMat.transparency = 0.38   // 62% visible — clearly "ghost" compared to live
        iconPlane.firstMaterial = iconMat

        let iconNode = SCNNode(geometry: iconPlane)
        iconNode.name = "icon"
        iconNode.position = SCNVector3(0, 0.38, 0)
        iconNode.constraints = [SCNBillboardConstraint()]
        node.addChildNode(iconNode)

        return node
    }

    private func landmarkNodeName(for id: UUID) -> String {
        "landmark:\(id.uuidString)"
    }

    private func landmarkID(from nodeName: String?) -> UUID? {
        guard let nodeName,
              nodeName.hasPrefix("landmark:") else {
            return nil
        }

        return UUID(uuidString: String(nodeName.dropFirst("landmark:".count)))
    }

    private func refreshLandmarkSelectionAppearance() {
        for (id, node) in landmarkNodes {
            applySelectionStyle(to: node, isSelected: id == selectedLandmarkID)
        }
    }

    private func applySelectionStyle(to node: SCNNode, isSelected: Bool) {
        let scale: Float = isSelected ? 1.42 : 1.0
        node.scale = SCNVector3(scale, scale, scale)
        node.opacity = isSelected ? 1.0 : 0.9
    }

    private func makeRouteSegmentNode(from start: simd_float3, to end: simd_float3) -> SCNNode {
        let planarStart = SIMD2<Float>(start.x, start.z)
        let planarEnd = SIMD2<Float>(end.x, end.z)
        let segmentLength = simd_distance(planarStart, planarEnd)

        let geometry = SCNBox(
            width: 0.09,
            height: 0.018,
            length: max(CGFloat(segmentLength), 0.05),
            chamferRadius: 0.018
        )
        geometry.materials = Array(
            repeating: makeFlatMaterial(
                color: UIColor(red: 1.0, green: 0.72, blue: 0.16, alpha: 0.95)
            ),
            count: 6
        )

        let node = SCNNode(geometry: geometry)
        let midpoint = (start + end) * 0.5
        node.position = SCNVector3(midpoint.x, 0.03, midpoint.z)
        node.eulerAngles.y = atan2(end.x - start.x, end.z - start.z)
        return node
    }

    private func makeRouteEndpointNode(at point: simd_float3) -> SCNNode {
        let geometry = SCNTube(innerRadius: 0.12, outerRadius: 0.19, height: 0.015)
        geometry.radialSegmentCount = 36
        geometry.materials = Array(
            repeating: makeFlatMaterial(
                color: UIColor(red: 1.0, green: 0.83, blue: 0.24, alpha: 0.92)
            ),
            count: 3
        )

        let node = SCNNode(geometry: geometry)
        node.position = SCNVector3(point.x, 0.035, point.z)
        return node
    }

    private func accentColor(for className: String) -> UIColor {
        switch className {
        case "person":
            return UIColor(red: 1.0, green: 0.8, blue: 0.2, alpha: 1.0)
        case "car", "truck", "bus", "motorcycle", "bicycle":
            return UIColor(red: 0.92, green: 0.28, blue: 0.24, alpha: 1.0)
        case "tv", "laptop", "cell phone", "remote":
            return UIColor(red: 0.37, green: 0.58, blue: 0.98, alpha: 1.0)
        case "chair", "couch", "bed":
            return UIColor(red: 0.31, green: 0.72, blue: 0.49, alpha: 1.0)
        case "dining table":
            return UIColor(red: 0.64, green: 0.48, blue: 0.33, alpha: 1.0)
        default:
            return UIColor(red: 0.42, green: 0.68, blue: 0.95, alpha: 1.0)
        }
    }

    private func symbolName(for className: String) -> String {
        switch className {
        case "person":
            return "figure.walk"
        case "car", "truck", "bus":
            return "car.fill"
        case "motorcycle", "bicycle":
            return "bicycle"
        case "tv":
            return "tv.fill"
        case "laptop":
            return "laptopcomputer"
        case "cell phone":
            return "iphone"
        case "chair":
            return "chair.fill"
        case "couch":
            return "sofa.fill"
        case "bed":
            return "bed.double.fill"
        case "dining table":
            return "table.furniture.fill"
        case "potted plant":
            return "leaf.fill"
        case "bottle", "cup", "wine glass":
            return "cup.and.saucer.fill"
        default:
            return "shippingbox.fill"
        }
    }

    private func iconImage(for className: String, tintColor: UIColor) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 160, height: 160))
        let symbolConfig = UIImage.SymbolConfiguration(pointSize: 72, weight: .semibold)
        let symbol = UIImage(systemName: symbolName(for: className), withConfiguration: symbolConfig)

        return renderer.image { _ in
            let bounds = CGRect(x: 0, y: 0, width: 160, height: 160)
            let backgroundPath = UIBezierPath(roundedRect: bounds, cornerRadius: 34)
            UIColor.black.withAlphaComponent(0.88).setFill()
            backgroundPath.fill()

            let ringPath = UIBezierPath(roundedRect: bounds.insetBy(dx: 6, dy: 6), cornerRadius: 28)
            tintColor.withAlphaComponent(0.95).setStroke()
            ringPath.lineWidth = 8
            ringPath.stroke()

            let titleRect = CGRect(x: 20, y: 112, width: 120, height: 24)
            let title = abbreviatedLabel(for: className) as NSString
            let paragraph = NSMutableParagraphStyle()
            paragraph.alignment = .center
            title.draw(
                in: titleRect,
                withAttributes: [
                    .font: UIFont.systemFont(ofSize: 18, weight: .semibold),
                    .foregroundColor: UIColor.white,
                    .paragraphStyle: paragraph
                ]
            )

            let symbolRect = CGRect(x: 34, y: 24, width: 92, height: 78)
            let fallbackSymbol = UIImage(systemName: "questionmark.circle.fill", withConfiguration: symbolConfig)
            (symbol ?? fallbackSymbol)?
                .withTintColor(tintColor, renderingMode: .alwaysOriginal)
                .draw(in: symbolRect)
        }
    }

    private func abbreviatedLabel(for className: String) -> String {
        switch className {
        case "dining table":
            return "table"
        case "cell phone":
            return "phone"
        case "potted plant":
            return "plant"
        default:
            return className
        }
    }

    private func makeFlatMaterial(color: UIColor) -> SCNMaterial {
        let material = SCNMaterial()
        material.diffuse.contents = color
        material.emission.contents = color.withAlphaComponent(0.18)
        material.lightingModel = .constant
        return material
    }

    private func makeSpriteMaterial(image: UIImage) -> SCNMaterial {
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.emission.contents = image
        material.transparent.contents = image
        material.isDoubleSided = true
        material.lightingModel = .constant
        material.writesToDepthBuffer = false
        material.readsFromDepthBuffer = false
        material.blendMode = .alpha
        material.transparencyMode = .aOne
        return material
    }

    private func makeMaterial(diffuse: UIColor, emission: UIColor) -> SCNMaterial {
        let material = SCNMaterial()
        material.diffuse.contents = diffuse
        material.emission.contents = emission
        material.lightingModel = .phong
        material.specular.contents = UIColor(white: 0.85, alpha: 1.0)
        material.shininess = 22
        return material
    }

    // MARK: - Mesh helpers

    private func faceIndices(
        for faceIndex: Int,
        faces: ARGeometryElement,
        faceBuffer: UnsafeMutableRawPointer
    ) -> (UInt32, UInt32, UInt32)? {
        let bytesPerIndex = faces.bytesPerIndex
        let indicesPerFace = faces.indexCountPerPrimitive

        guard indicesPerFace == 3 else { return nil }

        let base = faceBuffer.advanced(by: faceIndex * indicesPerFace * bytesPerIndex)

        func readIndex(at offset: Int) -> UInt32 {
            let ptr = base.advanced(by: offset * bytesPerIndex)
            switch bytesPerIndex {
            case 2:
                return UInt32(ptr.assumingMemoryBound(to: UInt16.self).pointee)
            case 4:
                return ptr.assumingMemoryBound(to: UInt32.self).pointee
            default:
                return 0
            }
        }

        let i0 = readIndex(at: 0)
        let i1 = readIndex(at: 1)
        let i2 = readIndex(at: 2)

        return (i0, i1, i2)
    }

    private func vertexAt(
        index: UInt32,
        vertices: ARGeometrySource,
        vertexBuffer: UnsafeMutableRawPointer
    ) -> simd_float3 {
        let stride = vertices.stride
        let ptr = vertexBuffer
            .advanced(by: Int(index) * stride)
            .assumingMemoryBound(to: SIMD3<Float>.self)

        let v = ptr.pointee
        return simd_float3(v.x, v.y, v.z)
    }

    /// Remove near-horizontal faces.
    /// This removes floor and ceiling, while keeping vertical / tree-like / wall-like structures.
    private func shouldDropHorizontalFace(
        v0Local: simd_float3,
        v1Local: simd_float3,
        v2Local: simd_float3,
        anchorTransform: simd_float4x4
    ) -> Bool {
        let w0 = (anchorTransform * simd_float4(v0Local, 1)).xyz
        let w1 = (anchorTransform * simd_float4(v1Local, 1)).xyz
        let w2 = (anchorTransform * simd_float4(v2Local, 1)).xyz

        let e1 = w1 - w0
        let e2 = w2 - w0

        let crossValue = simd_cross(e1, e2)
        let area = simd_length(crossValue)

        // Ignore degenerate tiny faces
        if area < 1e-5 {
            return true
        }

        let normal = simd_normalize(crossValue)
        let upDot = abs(simd_dot(normal, simd_float3(0, 1, 0)))

        // If the face is close to horizontal, drop it
        return upDot > horizontalFaceThreshold
    }

    // MARK: - Structure rendering

    /// Keep vertical / structural faces, drop floor & ceiling faces.
    private func makeStructureGeometry(from anchor: ARMeshAnchor) -> SCNGeometry? {
        let geometry = anchor.geometry

        let vertices = geometry.vertices
        let faces = geometry.faces

        let vertexCount = vertices.count
        guard vertexCount > 0 else { return nil }

        let vertexStride = vertices.stride
        let vertexBuffer = vertices.buffer.contents()

        let vertexData = Data(
            bytes: vertexBuffer,
            count: vertexCount * vertexStride
        )

        let vertexSource = SCNGeometrySource(
            data: vertexData,
            semantic: .vertex,
            vectorCount: vertexCount,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: vertexStride
        )

        let faceCount = faces.count
        guard faceCount > 0 else { return nil }

        let faceBuffer = faces.buffer.contents()
        var keptIndices: [UInt32] = []
        keptIndices.reserveCapacity(faceCount * 3)

        let faceStep = max(1, faceCount / maxFacesPerAnchor)

        for faceIndex in stride(from: 0, to: faceCount, by: faceStep) {
            guard let (i0, i1, i2) = faceIndices(
                for: faceIndex,
                faces: faces,
                faceBuffer: faceBuffer
            ) else { continue }

            let v0 = vertexAt(index: i0, vertices: vertices, vertexBuffer: vertexBuffer)
            let v1 = vertexAt(index: i1, vertices: vertices, vertexBuffer: vertexBuffer)
            let v2 = vertexAt(index: i2, vertices: vertices, vertexBuffer: vertexBuffer)

            if shouldDropHorizontalFace(
                v0Local: v0,
                v1Local: v1,
                v2Local: v2,
                anchorTransform: anchor.transform
            ) {
                continue
            }

            keptIndices.append(i0)
            keptIndices.append(i1)
            keptIndices.append(i2)
        }

        guard !keptIndices.isEmpty else { return nil }

        let indexData = keptIndices.withUnsafeBufferPointer {
            Data(buffer: $0)
        }

        let fillElement = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: keptIndices.count / 3,
            bytesPerIndex: MemoryLayout<UInt32>.size
        )

        let lineElement = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: keptIndices.count / 3,
            bytesPerIndex: MemoryLayout<UInt32>.size
        )

        let scnGeometry = SCNGeometry(
            sources: [vertexSource],
            elements: [fillElement, lineElement]
        )

        // Material 1: translucent blue body
        let fillMaterial = SCNMaterial()
        fillMaterial.diffuse.contents = UIColor(
            red: 0.20,
            green: 0.65,
            blue: 1.00,
            alpha: 0.28
        )
        fillMaterial.emission.contents = UIColor(
            red: 0.08,
            green: 0.25,
            blue: 0.45,
            alpha: 1.0
        )
        fillMaterial.isDoubleSided = true
        fillMaterial.lightingModel = SCNMaterial.LightingModel.constant
        fillMaterial.readsFromDepthBuffer = true
        fillMaterial.writesToDepthBuffer = false
        fillMaterial.blendMode = .alpha

        // Material 2: wireframe structure overlay
        let lineMaterial = SCNMaterial()
        lineMaterial.diffuse.contents = UIColor(
            red: 0.10,
            green: 0.75,
            blue: 1.00,
            alpha: 1.0
        )
        lineMaterial.emission.contents = UIColor(
            red: 0.20,
            green: 0.85,
            blue: 1.00,
            alpha: 1.0
        )
        lineMaterial.isDoubleSided = true
        lineMaterial.lightingModel = SCNMaterial.LightingModel.constant
        lineMaterial.fillMode = .lines

        scnGeometry.materials = [fillMaterial, lineMaterial]
        return scnGeometry
    }
}

// MARK: - SIMD helpers

private extension simd_float4 {
    var xyz: simd_float3 {
        simd_float3(x, y, z)
    }
}
