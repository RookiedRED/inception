//
//  AppOrientation.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/8/26.
//

import UIKit

/// Normalizes raw device orientation values into the app's supported landscape states.
enum AppOrientation {
    case normal
    case flipped

    /// Maps physical device orientation to the app's overlay/minimap coordinate system.
    init(from deviceOrientation: UIDeviceOrientation) {
        switch deviceOrientation {
        case .landscapeRight:
            self = .flipped
        default:
            self = .normal
        }
    }

    /// Indicates whether UI content should be visually rotated to match the flipped hold.
    var isFlipped: Bool { self == .flipped }
}
