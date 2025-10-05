import Foundation
import UIKit

struct USDZModel: Identifiable {
    let id = UUID()
    let name: String
    let fileName: String
    
    // Computed property for display name
    var displayName: String {
        // Remove file extension and replace underscores with spaces
        let nameWithoutExtension = fileName.replacingOccurrences(of: ".usdz", with: "")
        return name.isEmpty ? nameWithoutExtension
            .replacingOccurrences(of: "_", with: " ")
            .capitalized : name
    }
    
    // Check if the model file exists in the bundle
    var dataAsset: NSDataAsset? {
        // First check if it's a dollhouse model in Documents
        if fileName.contains("dollhouse_") {
            let documentsURL = FileManager.default.urls(
                for: .documentDirectory,
                in: .userDomainMask
            ).first!
            let fileURL = documentsURL.appendingPathComponent(fileName)
            
            if FileManager.default.fileExists(atPath: fileURL.path) {
                // Return a dummy NSDataAsset to indicate the file exists
                // (We'll handle actual loading differently for dollhouse models)
                return NSDataAsset(name: "placeholder") // This might return nil, which is fine
            }
            return nil
        }
        
        // Check in bundle for regular models
        return NSDataAsset(name: fileName) ?? NSDataAsset(name: fileName.replacingOccurrences(of: ".usdz", with: ""))
    }
    
    // Check if this is a dollhouse model
    var isDollhouse: Bool {
        return fileName.contains("dollhouse_")
    }
    
    // Get the file URL for the model
    var fileURL: URL? {
        if isDollhouse {
            // Dollhouse models are in Documents directory
            let documentsURL = FileManager.default.urls(
                for: .documentDirectory,
                in: .userDomainMask
            ).first!
            return documentsURL.appendingPathComponent(fileName)
        } else {
            // Regular models are in the bundle
            return Bundle.main.url(forResource: fileName, withExtension: nil)
                ?? Bundle.main.url(forResource: fileName.replacingOccurrences(of: ".usdz", with: ""), withExtension: "usdz")
        }
    }
}

// Extension for Equatable conformance
extension USDZModel: Equatable {
    static func == (lhs: USDZModel, rhs: USDZModel) -> Bool {
        return lhs.id == rhs.id
    }
}

// Extension for Hashable conformance (useful for Sets and Dictionaries)
extension USDZModel: Hashable {
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
