import Foundation
import UIKit

class USDZModelManager: ObservableObject {
    @Published var models: [USDZModel] = []
    
    init() {
        loadModels()
    }
    
    private func loadModels() {
        let modelNames = [
            "vintage_living_room",
            "cozy_living_room_baked"
        ]
        
        models = modelNames.compactMap { name in
            let model = USDZModel(name: name, fileName: name)
            return model.dataAsset != nil ? model : nil
        }
    }
    
    func getModel(by fileName: String) -> USDZModel? {
        return models.first { $0.fileName == fileName }
    }
}