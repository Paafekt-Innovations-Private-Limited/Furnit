import SwiftUI
import SceneKit
import UIKit

// MARK: - 2.5D Dollhouse Creator with Debug Logging
class DollhouseCreator {
    
    // Creates a USDZ file from photos using SceneKit
    static func createDollhouseFile(from photos: [UIImage], fileName: String) -> URL? {
        print("\n========== DOLLHOUSE CREATION START ==========")
        print("📸 Input: \(photos.count) photos")
        print("📝 Filename: \(fileName)")
        
        guard photos.count >= 6 else {
            print("❌ ERROR: Need at least 6 photos, got \(photos.count)")
            return nil
        }
        
        // Log photo details
        for (index, photo) in photos.enumerated() {
            print("  Photo \(index + 1): \(photo.size.width)x\(photo.size.height) pixels")
        }
        
        // Create scene
        let scene = SCNScene()
        print("✅ Created SCNScene")
        
        // Room dimensions
        let roomWidth: CGFloat = 10
        let roomHeight: CGFloat = 6
        let roomDepth: CGFloat = 10
        print("📐 Room dimensions: \(roomWidth)x\(roomHeight)x\(roomDepth)")
        
        // Create individual walls
        print("🔨 Creating walls...")
        let walls = [
            createWall(width: roomWidth, height: roomHeight, image: photos[0], name: "Front"),
            createWall(width: roomDepth, height: roomHeight, image: photos[1], name: "Right"),
            createWall(width: roomWidth, height: roomHeight, image: photos[2], name: "Back"),
            createWall(width: roomDepth, height: roomHeight, image: photos[3], name: "Left"),
            createFloor(width: roomWidth, depth: roomDepth, image: photos[4]),
            createCeiling(width: roomWidth, depth: roomDepth, image: photos[5])
        ]
        
        // Position walls
        walls[0].position = SCNVector3(0, 0, -roomDepth/2)
        walls[0].eulerAngles = SCNVector3(0, 0, 0)
        walls[0].name = "FrontWall"
        
        walls[1].position = SCNVector3(roomWidth/2, 0, 0)
        walls[1].eulerAngles = SCNVector3(0, CGFloat.pi/2, 0)
        walls[1].name = "RightWall"
        
        walls[2].position = SCNVector3(0, 0, roomDepth/2)
        walls[2].eulerAngles = SCNVector3(0, CGFloat.pi, 0)
        walls[2].name = "BackWall"
        
        walls[3].position = SCNVector3(-roomWidth/2, 0, 0)
        walls[3].eulerAngles = SCNVector3(0, -CGFloat.pi/2, 0)
        walls[3].name = "LeftWall"
        
        walls[4].position = SCNVector3(0, -roomHeight/2, 0)
        walls[4].name = "Floor"
        
        walls[5].position = SCNVector3(0, roomHeight/2, 0)
        walls[5].name = "Ceiling"
        
        // Add all walls to scene
        for wall in walls {
            scene.rootNode.addChildNode(wall)
            print("  ✅ Added \(wall.name ?? "wall") to scene")
        }
        
        // Add camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.fieldOfView = 60
        cameraNode.position = SCNVector3(0, 0, 0)
        scene.rootNode.addChildNode(cameraNode)
        print("📷 Added camera at origin")
        
        // Add ambient light
        let lightNode = SCNNode()
        lightNode.light = SCNLight()
        lightNode.light?.type = .ambient
        lightNode.light?.color = UIColor.white
        lightNode.light?.intensity = 1000
        scene.rootNode.addChildNode(lightNode)
        print("💡 Added ambient light")
        
        // Get documents directory
        let documentsPath = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!
        print("📁 Documents path: \(documentsPath.path)")
        
        let exportURL = documentsPath.appendingPathComponent(fileName)
        print("🎯 Export target: \(exportURL.path)")
        
        // Delete existing file if it exists
        if FileManager.default.fileExists(atPath: exportURL.path) {
            print("⚠️ File already exists, deleting...")
            try? FileManager.default.removeItem(at: exportURL)
        }
        
        // Export the scene
        print("📤 Starting USDZ export...")
        let success = scene.write(
            to: exportURL,
            options: nil,
            delegate: nil,
            progressHandler: { totalProgress, error, stop in
                print("  Export progress: \(totalProgress * 100)%")
                if let error = error {
                    print("  ❌ Export error: \(error)")
                }
            }
        )
        
        if success {
            // Verify file was created
            if FileManager.default.fileExists(atPath: exportURL.path) {
                let fileSize = (try? FileManager.default.attributesOfItem(atPath: exportURL.path)[.size] as? Int) ?? 0
                print("✅ USDZ created successfully!")
                print("  📏 File size: \(fileSize) bytes")
                print("  📍 Location: \(exportURL.path)")
                
                // List all files in documents
                listDocumentsDirectory()
                
                return exportURL
            } else {
                print("❌ File write succeeded but file doesn't exist!")
                return nil
            }
        } else {
            print("❌ Scene write failed!")
            return nil
        }
    }
    
    // List all files in documents directory for debugging
    private static func listDocumentsDirectory() {
        print("\n📂 Documents Directory Contents:")
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        do {
            let fileURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil)
            for fileURL in fileURLs {
                let size = (try? FileManager.default.attributesOfItem(atPath: fileURL.path)[.size] as? Int) ?? 0
                print("  - \(fileURL.lastPathComponent) (\(size) bytes)")
            }
        } catch {
            print("  Error listing files: \(error)")
        }
    }
    
    // Create a wall with texture
    private static func createWall(width: CGFloat, height: CGFloat, image: UIImage, name: String = "Wall") -> SCNNode {
        print("  Creating \(name): \(width)x\(height) with image \(image.size)")
        
        let plane = SCNPlane(width: width, height: height)
        
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.isDoubleSided = true
        material.lightingModel = .constant
        
        plane.materials = [material]
        
        let node = SCNNode(geometry: plane)
        node.name = name
        
        return node
    }
    
    // Create floor
    private static func createFloor(width: CGFloat, depth: CGFloat, image: UIImage) -> SCNNode {
        print("  Creating Floor: \(width)x\(depth) with image \(image.size)")
        
        let plane = SCNPlane(width: width, height: depth)
        
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.isDoubleSided = true
        material.lightingModel = .constant
        
        plane.materials = [material]
        
        let node = SCNNode(geometry: plane)
        node.eulerAngles = SCNVector3(-CGFloat.pi/2, 0, 0)
        node.name = "Floor"
        
        return node
    }
    
    // Create ceiling
    private static func createCeiling(width: CGFloat, depth: CGFloat, image: UIImage) -> SCNNode {
        print("  Creating Ceiling: \(width)x\(depth) with image \(image.size)")
        
        let plane = SCNPlane(width: width, height: depth)
        
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.isDoubleSided = true
        material.lightingModel = .constant
        
        plane.materials = [material]
        
        let node = SCNNode(geometry: plane)
        node.eulerAngles = SCNVector3(CGFloat.pi/2, 0, 0)
        node.name = "Ceiling"
        
        return node
    }
}

// MARK: - Enhanced Dollhouse Scanner with Debug
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
        "📸 FRONT WALL - Stand at back, capture front wall",
        "📸 RIGHT WALL - Turn right, capture right wall",
        "📸 BACK WALL - Turn around, capture back wall",
        "📸 LEFT WALL - Turn left, capture left wall",
        "📸 FLOOR - Point camera down at floor",
        "📸 CEILING - Point camera up at ceiling"
    ]
    
    var body: some View {
        NavigationView {
            ZStack {
                // Background
                LinearGradient(
                    gradient: Gradient(colors: [.black, .blue.opacity(0.3)]),
                    startPoint: .top,
                    endPoint: .bottom
                )
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
                    Button("Cancel") {
                        isPresented = false
                    }
                    .foregroundColor(.white)
                }
            }
            .preferredColorScheme(.dark)
        }
        .sheet(isPresented: $showingCamera, onDismiss: {
            if let image = currentImage {
                print("📸 Photo \(currentPhotoIndex + 1) captured: \(image.size)")
                capturedPhotos.append(image)
                currentImage = nil
                currentPhotoIndex += 1
                
                if currentPhotoIndex >= 6 {
                    print("✅ All 6 photos captured, starting processing...")
                    processPhotos()
                }
            }
        }) {
            ImagePicker(image: $currentImage, sourceType: .camera)
        }
    }
    
    private var instructionsView: some View {
        VStack(spacing: 20) {
            // Progress
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
            
            VStack(spacing: 20) {
                Text(photoInstructions[currentPhotoIndex])
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                
                Text("Capture all details: tiles, curtains, furniture")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.7))
            }
            .padding()
            .background(Color.black.opacity(0.5))
            .cornerRadius(16)
            
            Spacer()
            
            Button(action: {
                print("👆 Take Photo button pressed for photo \(currentPhotoIndex + 1)")
                showingCamera = true
            }) {
                HStack {
                    Image(systemName: "camera.fill")
                    Text("Take Photo")
                }
                .font(.headline)
                .foregroundColor(.white)
                .padding(.horizontal, 40)
                .padding(.vertical, 16)
                .background(
                    LinearGradient(
                        gradient: Gradient(colors: [.blue, .purple]),
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .cornerRadius(25)
            }
            .padding(.bottom, 50)
            
            // Photo preview strip
            if !capturedPhotos.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(capturedPhotos.enumerated()), id: \.offset) { index, photo in
                            Image(uiImage: photo)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: 60, height: 60)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                .overlay(
                                    Text("\(index + 1)")
                                        .font(.caption2)
                                        .foregroundColor(.white)
                                        .padding(2)
                                        .background(Color.black.opacity(0.7))
                                        .cornerRadius(4)
                                        .padding(2),
                                    alignment: .topLeading
                                )
                        }
                    }
                    .padding(.horizontal)
                }
                .frame(height: 80)
                .background(Color.black.opacity(0.5))
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
            
            Text("Building your room from 6 photos...")
                .font(.body)
                .foregroundColor(.white.opacity(0.8))
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
            
            Text(generatedModelURL != nil ? "2.5D Room Created!" : "Room Saved (Using Placeholder)")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            if generatedModelURL != nil {
                Text("USDZ file created at:\n\(generatedModelURL!.lastPathComponent)")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
                    .multilineTextAlignment(.center)
            } else {
                Text("Photos captured but USDZ creation failed.\nUsing existing model as placeholder.")
                    .font(.body)
                    .foregroundColor(.orange)
                    .multilineTextAlignment(.center)
            }
            
            Button(action: {
                isPresented = false
            }) {
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
        print("\n🎬 PROCESSING START")
        print("📊 Photo count: \(capturedPhotos.count)")
        
        DispatchQueue.global(qos: .userInitiated).async {
            let fileName = "dollhouse_\(Date().timeIntervalSince1970).usdz"
            print("📝 Generating filename: \(fileName)")
            
            generatedModelURL = DollhouseCreator.createDollhouseFile(
                from: capturedPhotos,
                fileName: fileName
            )
            
            DispatchQueue.main.async {
                if let url = generatedModelURL {
                    print("✅ SUCCESS: USDZ created at \(url.path)")
                    saveToCollection(usdzURL: url)
                } else {
                    print("⚠️ WARNING: USDZ creation failed, using fallback")
                    saveToCollectionWithFallback()
                }
            }
        }
    }
    
    private func saveToCollection(usdzURL: URL) {
        let roomName = "2.5D Room - \(Date().formatted(date: .abbreviated, time: .shortened))"
        
        let newModel = USDZModel(
            name: roomName,
            fileName: usdzURL.lastPathComponent
        )
        
        print("\n💾 SAVING TO COLLECTION")
        print("  Name: \(roomName)")
        print("  File: \(usdzURL.lastPathComponent)")
        print("  Models before: \(modelManager.models.count)")
        
        modelManager.models.append(newModel)
        
        print("  Models after: \(modelManager.models.count)")
        
        isProcessing = false
        scanComplete = true
    }
    
    private func saveToCollectionWithFallback() {
        let roomName = "2.5D Room - \(Date().formatted(date: .abbreviated, time: .shortened))"
        
        // Use existing model as fallback
        let fallbackFile = modelManager.models.first?.fileName ?? "room.usdz"
        
        let newModel = USDZModel(
            name: roomName,
            fileName: fallbackFile
        )
        
        print("\n⚠️ FALLBACK SAVE")
        print("  Name: \(roomName)")
        print("  Fallback file: \(fallbackFile)")
        
        modelManager.models.append(newModel)
        
        isProcessing = false
        scanComplete = true
    }
    
    private func resetScanner() {
        capturedPhotos = []
        currentPhotoIndex = 0
        scanComplete = false
        generatedModelURL = nil
    }
}
