//
//  FurnitApp.swift
//  Furnit
//
//  Created by Sitaramaswamy Ponnamanda on 07/09/25.
//

import SwiftUI

@main
struct FurnitApp: App {
    // Global app state manager
    private let appStateManager = AppStateManager.shared
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.appState, appStateManager)
                .onAppear {
                    // Mark app as launched on first startup
                    if appStateManager.isFirstLaunch {
                        appStateManager.markAsLaunched()
                        print("🎉 Welcome to Furnit! First launch detected.")
                    }
                }
        }
    }
}
