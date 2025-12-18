import Foundation

// Script to populate blacklist.json with actual class names from classes.json
func populateBlacklistWithClassNames() {
    // Load classes.json
    guard let classesURL = Bundle.main.url(forResource: "classes", withExtension: "json"),
          let classesData = try? Data(contentsOf: classesURL),
          let classNames = try? JSONSerialization.jsonObject(with: classesData) as? [String: String] else {
        print("❌ Failed to load classes.json")
        return
    }
    
    // Load current blacklist.json
    guard let blacklistURL = Bundle.main.url(forResource: "blacklist", withExtension: "json"),
          let blacklistData = try? Data(contentsOf: blacklistURL),
          let blacklistIds = try? JSONSerialization.jsonObject(with: blacklistData) as? [String: String] else {
        print("❌ Failed to load blacklist.json")
        return
    }
    
    // Create updated blacklist with real class names
    var updatedBlacklist: [String: String] = [:]
    
    for (idString, _) in blacklistIds {
        if let className = classNames[idString] {
            updatedBlacklist[idString] = className
        } else {
            updatedBlacklist[idString] = "UNKNOWN_CLASS_\(idString)"
            print("⚠️ No class name found for ID: \(idString)")
        }
    }
    
    // Write updated blacklist back to file
    do {
        let outputData = try JSONSerialization.data(withJSONObject: updatedBlacklist, options: [.prettyPrinted, .sortedKeys])
        if let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let outputURL = documentsURL.appendingPathComponent("blacklist.json")
            try outputData.write(to: outputURL)
            print("✅ Updated blacklist.json written to: \(outputURL.path)")
        }
    } catch {
        print("❌ Failed to write updated blacklist.json: \(error)")
    }
}

// Extension to add blacklist population method to SmartyPantsContainerView
extension SmartyPantsContainerView {
    
    /// Call this method to update the blacklist.json with actual class names from classes.json
    func updateBlacklistWithClassNames() {
        populateBlacklistWithClassNames()
    }
    
    /// Get a formatted class name for display
    func displayClassName(_ id: Int) -> String {
        let name = classNames[id] ?? "unknown"
        return name.replacingOccurrences(of: "_", with: " ").capitalized
    }
}

// Call the function if running as a standalone script
// populateBlacklistWithClassNames()
