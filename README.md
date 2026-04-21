# Inception

Inception is an iOS AR navigation prototype that combines live object detection, LiDAR-based world localization, persistent landmark memory, a 2.5D minimap, and on-device route planning.

The app uses the device camera and ARKit scene understanding to detect objects in the environment, estimate their metric world positions, remember static landmarks across frames, render the scanned structure on a minimap, and compute navigable routes through an occupancy grid built from AR mesh anchors.

## What The App Does

- Shows a live camera preview with detection overlays.
- Runs on-device object detection using an ONNX Runtime model.
- Uses ARKit scene depth to estimate metric 3D positions for detections.
- Stores static objects as persistent landmarks.
- Builds a 2D occupancy grid from AR mesh anchors.
- Renders a 2.5D minimap with the user position, heading, route, landmarks, and scanned structure.
- Computes navigation routes with A* pathfinding in AR world space.
- Continuously refreshes the route while the user moves.

## Core User Flow

1. The app starts an `ARWorldTrackingConfiguration` session.
2. The camera feed is rendered to a `MetalKit` preview.
3. Frames are sent to the inference pipeline.
4. Detections are fused with ARKit depth to recover world positions.
5. Static detections are merged into persistent landmarks.
6. AR mesh anchors are visualized in the minimap and converted into an occupancy map.
7. When a landmark is selected, the app calculates a route from the user position to that landmark.
8. The route is refreshed as the user moves through the environment.

## Architecture

### App Layer

- `inception/inception/App/inceptionApp.swift`
  Starts the SwiftUI app and loads `RootView`.
- `inception/inception/App/AppDelegate.swift`
  Locks the app to landscape mode.

### View Layer

- `inception/inception/Views/RootView.swift`
  Main composition view for camera preview, overlays, HUD, minimap, landmark panel, and navigation prompts.
- `inception/inception/Views/CameraPreviewView.swift`
  Renders the latest camera frame using `MTKView` and `CIContext`.
- `inception/inception/Views/DetectionOverlayView.swift`
  Draws bounding boxes and labels aligned with the aspect-filled preview.
- `inception/inception/Views/MiniMapView.swift`
  Wraps `SCNView` for the 2.5D minimap.
- `inception/inception/Views/HUDView.swift`
  Displays inference latency and detection count.

### View Model

- `inception/inception/ViewModel/DriveViewModel.swift`
  Coordinates AR frames, inference, landmarks, minimap updates, and navigation state.

### Services

- `inception/inception/Services/ARCameraService.swift`
  Owns the `ARSession`, publishes camera frames and mesh anchors, and manages AR processing flow.
- `inception/inception/Services/InferenceService.swift`
  Runs ONNX inference and parses detections.
- `inception/inception/Services/LandmarkStore.swift`
  Stores persistent static landmarks across frames.
- `inception/inception/Services/MiniMapService.swift`
  Builds and updates the SceneKit minimap scene.
- `inception/inception/Services/NavigationService.swift`
  Builds an occupancy grid and computes routes with A*.

### Models

- `Detection`
  2D object detection result.
- `TrackedObject`
  Detection enriched with depth and world position.
- `Landmark`
  Persistent static object remembered across frames.
- `ARFrameContext`
  Lightweight AR frame metadata used across the pipeline.

## Main Technologies

### 1. SwiftUI

SwiftUI is used for the main application UI, overlays, HUD, navigation prompts, and layout coordination.

### 2. ARKit

ARKit provides:

- World tracking
- Scene depth
- Mesh reconstruction
- Camera intrinsics
- Camera transform

Current session configuration:

- `worldAlignment = .gravity`
- `planeDetection = [.horizontal, .vertical]`
- `frameSemantics = .sceneDepth` when supported
- `sceneReconstruction = .meshWithClassification` when supported, otherwise `.mesh`

### 3. SceneKit

SceneKit is used to render the minimap:

- User marker
- Heading direction
- Landmarks
- Route segments
- Occupancy grid
- AR mesh structure

### 4. MetalKit + Core Image

The camera preview is rendered with:

- `MTKView`
- `CIContext`
- continuous rendering at `60 FPS`

This keeps the camera preview visually smooth even when inference or route computation is slower than the display refresh rate.

### 5. ONNX Runtime + Core ML Execution Provider

Object detection runs through ONNX Runtime using the Core ML execution provider.

Current configuration:

- Model file: `yolo26n.ort`
- Input tensor size: `640 x 640`
- Confidence threshold: `0.25`
- IoU threshold: `0.45`
- Intra-op threads: `2`
- Core ML compute units: `CPUAndNeuralEngine`
- Model format: `MLProgram`

The project also includes intermediate model artifacts:

- `yolov8n-seg.mlpackage`
- `yolo26n.onnx`
- `yolo26n.ort`

## Detection Pipeline

The inference pipeline works as follows:

1. Copy the AR camera frame.
2. Resize and letterbox the image to `640 x 640`.
3. Run ONNX inference on-device.
4. Parse detections from the output tensor.
5. Project the center of each detection into scene depth.
6. Use ARKit intrinsics and camera transform to recover a metric world-space point.

Important parameters:

- Detection confidence threshold: `0.25`
- Non-maximum suppression IoU threshold: `0.45`
- Adaptive inference interval targets about `65%` Neural Engine duty cycle
- Adaptive frame rate clamp: between about `6 FPS` and `15 FPS`

Depth sampling details:

- The app samples a `5 x 5` neighborhood around the detection center in the depth map.
- It uses the median valid depth value for robustness.

## Landmark Memory

Static detections are persisted as landmarks so they remain available after the object leaves the current frame.

Dynamic classes are intentionally excluded, including:

- people
- vehicles
- animals

Key parameters in `inception/inception/Services/LandmarkStore.swift`:

- Merge radius: `1.5 m`
- Minimum landmark confidence: `0.35`
- Minimum observations before showing on the map: `2`
- Maximum stored landmarks: `120`
- Position update smoothing alpha: `0.15`

Landmarks of the same class are merged if they are close enough in the XZ plane.

## Minimap System

The minimap is a 2.5D SceneKit scene that stays centered on the user and rotates with user heading.

It displays:

- scanned AR structure
- route segments
- route endpoint
- occupancy grid
- tracked objects
- persistent landmarks
- user position and heading

Current minimap rendering parameters:

- Continuous rendering: enabled
- Preferred frame rate: `60 FPS`
- Camera position in compact mode: `(0, 2.4, 3.2)`
- Camera position in expanded mode: `(0, 4.8, 6.4)`
- Compact orthographic scale: `3.2`
- Expanded orthographic scale: `5.0`
- Expanded zoom range: `0.55` to `2.4`
- Position smoothing alpha: `0.18`
- Yaw smoothing alpha: `0.14`

Mesh visualization parameters:

- Horizontal-face rejection threshold: `0.94`
- Mesh geometry refresh interval: `0.8 s`
- Maximum sampled faces per anchor: `2500`

AR mesh publishing parameters:

- Mesh publish interval: `0.45 s`
- Mesh anchor removal grace period: `2.0 s`

The grace period helps avoid visible mesh flicker when ARKit temporarily removes and re-adds anchors during remapping.

## Navigation System

Navigation is based on a 2D occupancy grid built from AR mesh geometry.

### Occupancy Classification

For each sampled triangle in the AR mesh:

- `abs(dot(normal, up)) < 0.5` -> obstacle or wall
- `dot(normal, up) > 0.7` -> navigable floor
- otherwise -> ignored

### Navigation Parameters

Current `NavigationService` parameters:

- Grid cell size: `0.25 m`
- Obstacle dilation cells: `0`
- Maximum A* iterations: `8000`
- Mesh face sampling divisor: `800`
- Floor expansion radius: `4` cells in X and Z

The floor expansion corresponds to about `1.0 m` of walkable padding to bridge sparse AR floor scans.

### Pathfinding Strategy

Routing uses a two-pass A* strategy:

1. Prefer paths through confirmed scanned floor cells only.
2. If that fails, fall back to obstacle-only routing through non-blocked space.

Additional pathfinding details:

- 8-directional A*
- Octile distance heuristic
- diagonal corner cutting is blocked
- final path is smoothed with greedy line-of-sight simplification

### Route Refresh Parameters

Current route refresh parameters in `inception/inception/ViewModel/DriveViewModel.swift`:

- Route refresh interval: `0.6 s`
- Route refresh movement threshold: `0.3 m`
- Minimap camera update interval: about `33 ms` (`30 FPS`)

## Performance Design

The app is designed so rendering stays smooth even when heavy processing is slower.

Performance choices currently implemented:

- camera preview rendering is decoupled from SwiftUI view recomposition
- camera preview uses `MTKView` continuous rendering at `60 FPS`
- minimap uses `SCNView` continuous rendering at `60 FPS`
- inference runs on a dedicated background queue
- model setup runs on a separate setup queue
- navigation work runs on a utility-priority queue
- AR mesh geometry creation is offloaded from the main thread where possible
- route planning and occupancy rebuilding are not done on every frame

## Current Limitations

- Route quality depends heavily on AR mesh quality and scene depth stability.
- Screen recording can reduce AR mesh quality and hurt route generation because ARKit scene reconstruction is resource-intensive.
- Narrow passages may be rejected if mesh sampling marks too many cells as blocked.
- Landmark IDs are regenerated each inference pass; the current system is detection-driven, not a full multi-object tracker.
- The app currently focuses on landscape usage.

## Repository Contents

- `inception/inception/`
  Main iOS source code
- `inception/Products/inception.app`
  Built application bundle
- `inception/yolov8n-seg.mlpackage`
  Core ML package artifact
- `inception/yolo26n.onnx`
  ONNX model artifact
- `inception/yolo26n.ort`
  ONNX Runtime optimized model used at runtime

## Summary

Inception is an on-device AR navigation prototype that combines:

- ARKit world understanding
- ONNX-based object detection
- scene-depth-based 3D localization
- persistent landmark memory
- SceneKit minimap rendering
- occupancy-grid navigation
- A* route planning

The result is a real-time system that can detect objects, remember landmarks, reconstruct nearby structure, and guide the user through an AR-scanned environment.
