import Foundation
import UIKit

struct USDZModel: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let fileName: String
    let dataAsset: NSDataAsset?
    
    init(name: String, fileName: String) {
        self.name = name
        self.fileName = fileName
        self.dataAsset = NSDataAsset(name: fileName)
    }
    
    var displayName: String {
        return name.replacingOccurrences(of: "_", with: " ").capitalized
    }
    
    // Create a temporary URL from the data asset
    var temporaryURL: URL? {
        guard let data = dataAsset?.data else {
            print("❌ [USDZModel] No data for \(fileName)")
            return nil
        }
        
        let tempDirectory = FileManager.default.temporaryDirectory
        let tempURL = tempDirectory.appendingPathComponent("\(fileName)_\(id.uuidString).usdz")
        
        do {
            try data.write(to: tempURL)
            print("✅ [USDZModel] Wrote temporary file for \(fileName) at \(tempURL)")
            return tempURL
        } catch {
            print("❌ [USDZModel] Failed to write temporary file for \(fileName): \(error)")
            return nil
        }
    }
}
