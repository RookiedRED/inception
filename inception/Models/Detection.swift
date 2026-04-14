//
//  Detection.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/8/26.
//
import SwiftUI
import CoreGraphics

struct Detection {
    let bbox: CGRect
    let classId: Int
    let className: String
    let confidence: Float
    let maskCoeffs: [Float]
}

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
    var color: UIColor {
        switch className {
        case "car", "truck", "bus", "motorcycle", "bicycle": return .red
        case "person": return .yellow
        default: return .green
        }
    }

    var label: String {
        "\(className) \(Int(confidence * 100))%"
    }
}
