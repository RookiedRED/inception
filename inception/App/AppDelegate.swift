//
//  AppDelegate.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/6/26.
//

import UIKit

/// Centralizes application-wide UIKit configuration that SwiftUI still depends on.
final class AppDelegate: NSObject, UIApplicationDelegate {
    /// Locks the app to a single landscape orientation so the AR preview,
    /// overlay, and minimap stay in the same coordinate system.
    func application(
        _ application: UIApplication,
        supportedInterfaceOrientationsFor window: UIWindow?
    ) -> UIInterfaceOrientationMask {
        return .landscapeRight
    }
}
