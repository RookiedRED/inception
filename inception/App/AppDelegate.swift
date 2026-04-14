//
//  AppDelegate.swift
//  iDriveBot
//
//  Created by Lin, Hung Yu on 4/6/26.
//

import UIKit

final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        supportedInterfaceOrientationsFor window: UIWindow?
    ) -> UIInterfaceOrientationMask {
        return .landscapeRight  // 只允許一個方向，系統不會自動轉
    }
}
