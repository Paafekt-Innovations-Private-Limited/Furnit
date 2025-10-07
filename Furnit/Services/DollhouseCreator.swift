import SwiftUI
import SceneKit
import UIKit

// MARK: - 2.5D Dollhouse Creator - Single Plane Approach
class DollhouseCreator {
    
    // Add to DollhouseCreator class
    static func clearAllDollhouseFiles() {
        guard let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return
        }
        
        do {
            let files = try FileManager.default.contentsOfDirectory(at: documentsPath, includingPropertiesForKeys: nil)
            for file in files where file.lastPathComponent.contains("dollhouse_") {
                try? FileManager.default.removeItem(at: file)
                print("🗑️ Deleted: \(file.lastPathComponent)")
            }
        } catch {
            print("Error clearing: \(error)")
        }
    }
    
    static func createDollhouseFile(from photos: [UIImage], fileName: String) -> URL? {
        print("\n========== DOLLHOUSE CREATION START ==========")
        clearAllDollhouseFiles()
        
        guard photos.count >= 6 else {
            print("ERROR: Need 6 photos")
            return nil
        }
        
        guard let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return nil
        }
        
        // Create a temp directory for textures alongside the USDZ
        let tempDir = documentsPath.appendingPathComponent("temp_textures_\(Date().timeIntervalSince1970)")
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        
        // Save textures to temp directory
        var imagePaths: [URL] = []
        for (index, photo) in photos.enumerated() {
            let imagePath = tempDir.appendingPathComponent("texture_\(index).png")
            if let pngData = photo.pngData() {
                try? pngData.write(to: imagePath)
                imagePaths.append(imagePath)
                print("💾 Saved texture \(index): \(imagePath.lastPathComponent)")
            }
        }
        
        guard imagePaths.count == 6 else {
            print("ERROR: Failed to save textures")
            try? FileManager.default.removeItem(at: tempDir)
            return nil
        }
        
        let scene = SCNScene()
        
        let roomWidth: Float = 4
        let roomHeight: Float = 3
        let roomDepth: Float = 4
        
        print("📐 Room dimensions: W:\(roomWidth) H:\(roomHeight) D:\(roomDepth)")

        // Front Wall
        print("\n🟦 Creating FRONT wall...")
        let frontWall = createWall(width: CGFloat(roomWidth), height: CGFloat(roomHeight), imagePath: imagePaths[0])
        frontWall.position = SCNVector3(0, 0, -roomDepth/2)
        frontWall.eulerAngles = SCNVector3(0, 0, -Float.pi/2)
        frontWall.name = "FrontWall"
        scene.rootNode.addChildNode(frontWall)

        // Right Wall
        print("\n🟩 Creating RIGHT wall...")
        let rightWall = createWall(width: CGFloat(roomDepth), height: CGFloat(roomHeight), imagePath: imagePaths[1])
        rightWall.position = SCNVector3(roomWidth/2, 0, 0)
        rightWall.eulerAngles = SCNVector3(0, -Float.pi/2, Float.pi/2)
        rightWall.name = "RightWall"
        scene.rootNode.addChildNode(rightWall)

        // Back Wall
        print("\n🟨 Creating BACK wall...")
        let backWall = createWall(width: CGFloat(roomWidth), height: CGFloat(roomHeight), imagePath: imagePaths[2])
        backWall.position = SCNVector3(0, 0, roomDepth/2)
        backWall.eulerAngles = SCNVector3(0, Float.pi, -Float.pi/2)
        backWall.name = "BackWall"
        scene.rootNode.addChildNode(backWall)

        // Left Wall
        print("\n🟧 Creating LEFT wall...")
        let leftWall = createWall(width: CGFloat(roomDepth), height: CGFloat(roomHeight), imagePath: imagePaths[3])
        leftWall.position = SCNVector3(-roomWidth/2, 0, 0)
        leftWall.eulerAngles = SCNVector3(0, Float.pi/2, -Float.pi/2)
        leftWall.name = "LeftWall"
        scene.rootNode.addChildNode(leftWall)

        // Floor
        print("\n🟫 Creating FLOOR...")
        let floor = createFloor(width: CGFloat(roomWidth), depth: CGFloat(roomDepth), imagePath: imagePaths[4])
        floor.position = SCNVector3(0, -roomHeight/2, 0)
        floor.name = "Floor"
        scene.rootNode.addChildNode(floor)

        // Ceiling
        print("\n⬜ Creating CEILING...")
        let ceiling = createCeiling(width: CGFloat(roomWidth), depth: CGFloat(roomDepth), imagePath: imagePaths[5])
        ceiling.position = SCNVector3(0, roomHeight/2, 0)
        ceiling.name = "Ceiling"
        scene.rootNode.addChildNode(ceiling)

        // Camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.fieldOfView = 60
        cameraNode.position = SCNVector3(-3, 1, 4)
        cameraNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(cameraNode)
        
        // Export to temp location FIRST (same directory as textures)
        let tempUsdzPath = tempDir.appendingPathComponent(fileName)
        
        print("\n📦 Exporting USDZ with textures...")
        let success = scene.write(to: tempUsdzPath, options: nil, delegate: nil, progressHandler: nil)
        
        if success {
            print("✅ USDZ created successfully in temp location")
            Thread.sleep(forTimeInterval: 0.5)  // Give it time to finalize
            
            // Now move the USDZ to final location
            let finalURL = documentsPath.appendingPathComponent(fileName)
            if FileManager.default.fileExists(atPath: finalURL.path) {
                try? FileManager.default.removeItem(at: finalURL)
            }
            
            do {
                try FileManager.default.moveItem(at: tempUsdzPath, to: finalURL)
                print("✅ Moved USDZ to final location")
                
                // Clean up temp directory
                try? FileManager.default.removeItem(at: tempDir)
                print("🗑️ Cleaned up temp textures")
                
                return finalURL
            } catch {
                print("❌ Error moving USDZ: \(error)")
            }
        } else {
            print("❌ Failed to write USDZ")
        }
        
        // Clean up on failure
        try? FileManager.default.removeItem(at: tempDir)
        return nil
    }

    // UPDATED: Use file path instead of UIImage
    private static func createWall(width: CGFloat, height: CGFloat, imagePath: URL) -> SCNNode {
        let plane = SCNPlane(width: width, height: height)
        let material = SCNMaterial()
        material.diffuse.contents = imagePath  // Use URL not UIImage
        material.isDoubleSided = true
        material.lightingModel = .constant
        plane.materials = [material]
        
        return SCNNode(geometry: plane)
    }

    private static func createFloor(width: CGFloat, depth: CGFloat, imagePath: URL) -> SCNNode {
        let plane = SCNPlane(width: width, height: depth)
        let material = SCNMaterial()
        material.diffuse.contents = imagePath  // Use URL not UIImage
        material.isDoubleSided = true
        material.lightingModel = .constant
        plane.materials = [material]
        
        let node = SCNNode(geometry: plane)
        node.eulerAngles = SCNVector3(-Float.pi/2, 0, 0)
        return node
    }

    private static func createCeiling(width: CGFloat, depth: CGFloat, imagePath: URL) -> SCNNode {
        let plane = SCNPlane(width: width, height: depth)
        let material = SCNMaterial()
        material.diffuse.contents = imagePath  // Use URL not UIImage
        material.isDoubleSided = true
        material.lightingModel = .constant
        plane.materials = [material]
        
        let node = SCNNode(geometry: plane)
        node.eulerAngles = SCNVector3(Float.pi/2, 0, 0)
        return node
    }
    
    private static func createWall(width: CGFloat, height: CGFloat, image: UIImage) -> SCNNode {
        let plane = SCNPlane(width: width, height: height)
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.isDoubleSided = true
        material.lightingModel = .constant
        plane.materials = [material]
        
        return SCNNode(geometry: plane)
    }
    
    private static func createFloor(width: CGFloat, depth: CGFloat, image: UIImage) -> SCNNode {
        let plane = SCNPlane(width: width, height: depth)
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.isDoubleSided = true
        material.lightingModel = .constant
        plane.materials = [material]
        
        let node = SCNNode(geometry: plane)
        node.eulerAngles = SCNVector3(-Float.pi/2, 0, 0)
        return node
    }
    
    private static func createCeiling(width: CGFloat, depth: CGFloat, image: UIImage) -> SCNNode {
        let plane = SCNPlane(width: width, height: depth)
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.isDoubleSided = true
        material.lightingModel = .constant
        plane.materials = [material]
        
        let node = SCNNode(geometry: plane)
        node.eulerAngles = SCNVector3(Float.pi/2, 0, 0)
        return node
    }
}

// MARK: - Scanner View (unchanged)
struct DollhouseRoomScannerView: View {
    @Binding var isPresented: Bool
    @ObservedObject var modelManager: USDZModelManager
    @State private var capturedPhotos: [UIImage] = []
    @State private var showingCamera = false
    @State private var isProcessing = false
    @State private var currentPhotoIndex = 0
    @State private var currentImage: UIImage?
    @State private var generatedModelURL: URL?
    @State private var scanComplete = false
    @State private var errorMessage: String?
    
    let photoInstructions = [
        "FRONT WALL - Stand at back, capture front wall",
        "RIGHT WALL - Turn right, capture right wall",
        "BACK WALL - Turn around, capture back wall",
        "LEFT WALL - Turn left, capture left wall",
        "FLOOR - Point camera down at floor",
        "CEILING - Point camera up at ceiling"
    ]
    
    var body: some View {
        NavigationView {
            ZStack {
                LinearGradient(gradient: Gradient(colors: [.black, .blue.opacity(0.3)]), startPoint: .top, endPoint: .bottom)
                    .ignoresSafeArea()
                
                if let error = errorMessage {
                    errorView(message: error)
                } else if currentPhotoIndex < 6 && !scanComplete {
                    instructionsView
                } else if isProcessing {
                    processingView
                } else if scanComplete {
                    successView
                }
            }
            .navigationTitle("2.5D Room Capture")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") { isPresented = false }
                        .foregroundColor(.white)
                }
            }
            .preferredColorScheme(.dark)
        }
        .sheet(isPresented: $showingCamera, onDismiss: {
            if let image = currentImage {
                capturedPhotos.append(image)
                currentImage = nil
                currentPhotoIndex += 1
                if currentPhotoIndex >= 6 { processPhotos() }
            }
        }) {
            ImagePicker(image: $currentImage, sourceType: .camera)
        }
    }
    
    private var instructionsView: some View {
        VStack(spacing: 20) {
            HStack {
                Text("Photo \(currentPhotoIndex + 1) of 6")
                    .font(.headline)
                    .foregroundColor(.white)
                Spacer()
                HStack(spacing: 8) {
                    ForEach(0..<6) { index in
                        Circle()
                            .fill(index < capturedPhotos.count ? Color.green : Color.white.opacity(0.3))
                            .frame(width: 12, height: 12)
                    }
                }
            }
            .padding()
            
            Spacer()
            
            Text(photoInstructions[currentPhotoIndex])
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
                .padding()
                .background(Color.black.opacity(0.5))
                .cornerRadius(16)
            
            Spacer()
            
            Button(action: { showingCamera = true }) {
                HStack {
                    Image(systemName: "camera.fill")
                    Text("Take Photo")
                }
                .font(.headline)
                .foregroundColor(.white)
                .padding(.horizontal, 40)
                .padding(.vertical, 16)
                .background(LinearGradient(gradient: Gradient(colors: [.blue, .purple]), startPoint: .leading, endPoint: .trailing))
                .cornerRadius(25)
            }
            .padding(.bottom, 50)
            
            if !capturedPhotos.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(capturedPhotos.enumerated()), id: \.offset) { index, photo in
                            Image(uiImage: photo)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: 60, height: 60)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                    .padding(.horizontal)
                }
                .frame(height: 80)
            }
        }
    }
    
    private var processingView: some View {
        VStack(spacing: 20) {
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                .scaleEffect(2)
            Text("Creating 2.5D Room")
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(.white)
        }
        .padding(40)
        .background(Color.black.opacity(0.8))
        .cornerRadius(20)
    }
    
    private var successView: some View {
        VStack(spacing: 30) {
            Image(systemName: generatedModelURL != nil ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .font(.system(size: 80))
                .foregroundColor(generatedModelURL != nil ? .green : .orange)
            Text(generatedModelURL != nil ? "2.5D Room Created!" : "Error Creating Room")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.white)
            Button(action: { isPresented = false }) {
                Text("View in Collection")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(width: 200, height: 50)
                    .background(Color.green)
                    .cornerRadius(25)
            }
        }
        .padding()
    }
    
    private func errorView(message: String) -> some View {
        VStack(spacing: 20) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 60))
                .foregroundColor(.red)
            Text("Error")
                .font(.title)
                .foregroundColor(.white)
            Text(message)
                .font(.body)
                .foregroundColor(.white.opacity(0.8))
                .multilineTextAlignment(.center)
                .padding()
            Button("Try Again") {
                errorMessage = nil
                resetScanner()
            }
            .padding()
            .background(Color.blue)
            .foregroundColor(.white)
            .cornerRadius(10)
        }
        .padding()
    }
    
    private func processPhotos() {
        isProcessing = true
        DispatchQueue.global(qos: .userInitiated).async {
            let fileName = "dollhouse_\(Date().timeIntervalSince1970).usdz"
            let url = DollhouseCreator.createDollhouseFile(from: capturedPhotos, fileName: fileName)
            DispatchQueue.main.async {
                generatedModelURL = url
                if let url = url {
                    saveToCollection(usdzURL: url, fileName: fileName)
                } else {
                    errorMessage = "Failed to create room"
                }
                isProcessing = false
                scanComplete = true
            }
        }
    }
    
    private func saveToCollection(usdzURL: URL, fileName: String) {
        guard FileManager.default.fileExists(atPath: usdzURL.path) else { return }
        NotificationCenter.default.post(name: USDZModelManager.didAddModelNotification, object: nil, userInfo: ["fileName": fileName])
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            self.modelManager.refreshModels()
        }
    }
    
    private func resetScanner() {
        capturedPhotos = []
        currentPhotoIndex = 0
        scanComplete = false
        generatedModelURL = nil
        errorMessage = nil
    }
}
