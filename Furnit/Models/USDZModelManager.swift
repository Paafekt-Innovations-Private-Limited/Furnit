import Foundation
import UIKit

class USDZModelManager: ObservableObject {
    @Published var models: [USDZModel] = []
    
    init() {
        print("📦 [USDZModelManager] Initializing...")
        loadModels()
        print("📦 [USDZModelManager] Initialization complete. Loaded \(models.count) models")
    }
    
    private func loadModels() {
        print("📦 [USDZModelManager] Starting to load models...")
        
        let modelNames = [
            "vintage_living_room",
            "cozy_living_room_baked"
        ]
        
        print("📦 [USDZModelManager] Model names to load: \(modelNames)")
        
        // Check if files exist in bundle
        for name in modelNames {
            if let url = Bundle.main.url(forResource: name, withExtension: "usdz") {
                print("✅ [USDZModelManager] Found file in bundle: \(name).usdz at \(url)")
            } else {
                print("❌ [USDZModelManager] File NOT found in bundle: \(name).usdz")
            }
        }
        
        models = modelNames.compactMap { name in
            print("📦 [USDZModelManager] Attempting to create model: \(name)")
            let model = USDZModel(name: name, fileName: name)
            
            if model.dataAsset != nil {
                print("✅ [USDZModelManager] Model created successfully: \(name)")
                print("   - displayName: \(model.displayName)")
                print("   - fileName: \(model.fileName)")
                return model
            } else {
                print("❌ [USDZModelManager] Model creation failed (dataAsset is nil): \(name)")
                return nil
            }
        }
        
        print("📦 [USDZModelManager] Final models array: \(models.map { $0.fileName })")
    }
    
    func getModel(by fileName: String) -> USDZModel? {
        return models.first { $0.fileName == fileName }
    }
}
