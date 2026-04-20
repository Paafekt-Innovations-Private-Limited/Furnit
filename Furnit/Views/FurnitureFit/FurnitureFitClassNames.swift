// FurnitureFitClassNames.swift
// Localized YOLO class display names from classes.json (*.lproj; same keys as Android assets/classes.json).

import Foundation

enum FurnitureFitClassNames {
    static func loadLocalizedTable(debugMode: Bool, log: (String) -> Void) -> [Int: String] {
        guard let url = Bundle.main.url(forResource: "classes", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            if debugMode { log("⚠️ Failed to load localized classes.json (check xx.lproj)") }
            return [:]
        }
        var result: [Int: String] = [:]
        for (key, value) in dict {
            if let id = Int(key) {
                result[id] = value
            }
        }
        if debugMode { log("✅ Loaded \(result.count) class names") }
        return result
    }
}
