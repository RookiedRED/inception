//
//  AppOrientation.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/8/26.
//

import UIKit

enum AppOrientation {
    case normal   // landscapeLeft，手機正常橫拿
    case flipped  // landscapeRight，手機上下翻轉

    init(from deviceOrientation: UIDeviceOrientation) {
        switch deviceOrientation {
        case .landscapeRight: self = .flipped  // 物理上下翻
        default:              self = .normal
        }
    }

    var isFlipped: Bool { self == .flipped }
}
