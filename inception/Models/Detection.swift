//
//  Detection.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/8/26.
//
import SwiftUI
import CoreGraphics

/// Raw object detection decoded from the model output tensor.
struct Detection {
    let bbox: CGRect
    let classId: Int
    let className: String
    let confidence: Float
    let maskCoeffs: [Float]
}

/// COCO class list expected by the bundled YOLO model.
let COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

extension Detection {
    /// Shared UI accent color for overlays and minimap markers.
    var color: UIColor {
        switch className {
        case "car", "truck", "bus", "motorcycle", "bicycle": return .red
        case "person": return .yellow
        default: return .green
        }
    }

    /// Human-readable label shown in the camera overlay.
    var label: String {
        "\(className) \(Int(confidence * 100))%"
    }
}
