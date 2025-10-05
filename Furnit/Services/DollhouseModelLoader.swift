import SwiftUI
import RealityKit
import SceneKit

// MARK: - Dollhouse Model Loader (Standalone)
struct DollhouseModelLoader {
    
    static func loadModel(fileName: String) -> ModelEntity? {
        print("\n📦 Loading model: \(fileName)")
        
        if fileName.contains("dollhouse_") {
            // Load from Documents directory
            return loadDollhouseModel(fileName: fileName)
        } else {
            // Load from bundle (existing models)
            return loadBundleModel(fileName: fileName)
        }
    }
    
    static private func loadDollhouseModel(fileName: String) -> ModelEntity? {
        print("🏠 Loading dollhouse model from Documents")
        
        let documentsURL = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!
        
        let fileURL = documentsURL.appendingPathComponent(fileName)
        
        print("📍 Looking for file at: \(fileURL.path)")
        print("📁 File exists: \(FileManager.default.fileExists(atPath: fileURL.path))")
        
        if FileManager.default.fileExists(atPath: fileURL.path) {
            do {
                // Try loading as ModelEntity
                let entity = try ModelEntity.loadModel(contentsOf: fileURL)
                print("✅ Dollhouse loaded successfully as ModelEntity")
                
                // Make it viewable from inside
                entity.scale *= -1
                
                return entity
            } catch {
                print("⚠️ ModelEntity load failed: \(error)")
                print("🔄 Using fallback loading...")
                
                // Return a simple box as fallback
                return createFallbackBox()
            }
        } else {
            print("❌ Dollhouse file not found")
            return nil
        }
    }
    
    static private func loadBundleModel(fileName: String) -> ModelEntity? {
        print("📚 Loading model from bundle: \(fileName)")
        
        guard let url = Bundle.main.url(forResource: fileName, withExtension: nil) else {
            print("❌ Bundle file not found: \(fileName)")
            return nil
        }
        
        do {
            let entity = try ModelEntity.loadModel(contentsOf: url)
            print("✅ Bundle model loaded successfully")
            return entity
        } catch {
            print("❌ Failed to load bundle model: \(error)")
            return nil
        }
    }
    
    static private func createFallbackBox() -> ModelEntity {
        print("🔨 Creating fallback box")
        
        let mesh = MeshResource.generateBox(size: 3)
        var material = SimpleMaterial()
        material.color = .init(tint: .white.withAlphaComponent(0.5))
        material.roughness = 1.0
        
        let entity = ModelEntity(mesh: mesh, materials: [material])
        entity.scale = SIMD3<Float>(repeating: -1) // Inside-out box
        
        return entity
    }
}

// MARK: - Simple Photo Room Viewer (No USDZ needed)
struct SimplePhotoRoomView: View {
    let roomName: String
    @State private var rotationY: Double = 0
    @State private var scale: CGFloat = 1
    
    // For demo, using color placeholders - replace with actual photos
    let wallColors: [Color] = [.red, .blue, .green, .yellow, .purple, .orange]
    
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            
            VStack {
                Text(roomName)
                    .font(.title2)
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(10)
                
                Spacer()
                
                // 3D-like room view
                ZStack {
                    // Create a box-like view with photos/colors
                    ForEach(0..<6, id: \.self) { index in
                        Rectangle()
                            .fill(wallColors[index])
                            .frame(width: 200, height: 200)
                            .rotation3DEffect(
                                .degrees(getRotation(for: index)),
                                axis: getAxis(for: index)
                            )
                            .offset(getOffset(for: index))
                            .opacity(getOpacity(for: index, rotation: rotationY))
                    }
                }
                .rotation3DEffect(
                    .degrees(rotationY),
                    axis: (x: 0, y: 1, z: 0)
                )
                .scaleEffect(scale)
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            rotationY = Double(value.translation.width)
                        }
                )
                .gesture(
                    MagnificationGesture()
                        .onChanged { value in
                            scale = value
                        }
                )
                
                Spacer()
                
                Text("Drag to rotate • Pinch to zoom")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.7))
                    .padding()
            }
        }
    }
    
    private func getRotation(for index: Int) -> Double {
        switch index {
        case 1: return 90  // Right wall
        case 2: return 180 // Back wall
        case 3: return -90 // Left wall
        case 4: return 90  // Floor
        case 5: return -90 // Ceiling
        default: return 0  // Front wall
        }
    }
    
    private func getAxis(for index: Int) -> (x: CGFloat, y: CGFloat, z: CGFloat) {
        switch index {
        case 1, 2, 3: return (0, 1, 0) // Walls rotate around Y
        case 4, 5: return (1, 0, 0)    // Floor/ceiling rotate around X
        default: return (0, 1, 0)
        }
    }
    
    private func getOffset(for index: Int) -> CGSize {
        switch index {
        case 0: return CGSize(width: 0, height: 0)    // Front
        case 1: return CGSize(width: 100, height: 0)  // Right
        case 2: return CGSize(width: 0, height: 0)    // Back
        case 3: return CGSize(width: -100, height: 0) // Left
        case 4: return CGSize(width: 0, height: 100)  // Floor
        case 5: return CGSize(width: 0, height: -100) // Ceiling
        default: return .zero
        }
    }
    
    private func getOpacity(for index: Int, rotation: Double) -> Double {
        // Adjust opacity based on rotation to hide back faces
        let normalizedRotation = rotation.truncatingRemainder(dividingBy: 360)
        
        switch index {
        case 0: // Front
            return (normalizedRotation > -90 && normalizedRotation < 90) ? 1 : 0.3
        case 2: // Back
            return (normalizedRotation < -90 || normalizedRotation > 90) ? 1 : 0.3
        default:
            return 0.8
        }
    }
}

// MARK: - Usage in ModelViewerView
// Add this check to your ModelViewerView:
/*
if model.fileName.contains("dollhouse_") {
    SimplePhotoRoomView(roomName: model.displayName)
} else {
    // Your existing RealityKit view
    RealityKitView(model: model, ...)
}
*/
