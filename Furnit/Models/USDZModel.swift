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
}