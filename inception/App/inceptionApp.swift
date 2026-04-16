//
//  inceptionApp.swift
//  inception
//
//  Created by Lin, Hung Yu on 4/14/26.
//

import SwiftUI

@main
struct inceptionApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    init() {
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}
