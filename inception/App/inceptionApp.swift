//
//  inceptionApp.swift
//  inception
//
//  Created by Lin, Hung Yu on 4/14/26.
//

import SwiftUI

@main
/// Application entry point responsible for bootstrapping the root SwiftUI scene.
struct inceptionApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    init() {
        // Orientation updates are consumed by the view model to keep overlays aligned.
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}
