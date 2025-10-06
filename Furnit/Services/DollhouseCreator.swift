import SwiftUI
import SceneKit
import UIKit

// MARK: - 2.5D Dollhouse Creator
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
        let roomWidth: CGFloat = 4
        let roomHeight: CGFloat = 3
        let roomDepth: CGFloat = 4
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
        
        // Position walls correctly in 3D space
        // Front wall
        walls[0].position = SCNVector3(0, 0, -Float(roomDepth/2))
        walls[0].eulerAngles = SCNVector3(0, 0, -Float.pi/2)
        walls[0].name = "FrontWall"
        
        // Right wall - Y rotated 90° clockwise
        walls[1].position = SCNVector3(Float(roomWidth/2), 0, 0)
        walls[1].eulerAngles = SCNVector3(0, -Float.pi/2, 0)
        walls[1].name = "RightWall"
        
        // Back wall
        walls[2].position = SCNVector3(0, 0, Float(roomDepth/2))
        walls[2].eulerAngles = SCNVector3(0, Float.pi, -Float.pi/2)
        walls[2].name = "BackWall"
        
        // Left wall - flipped 180° upside down
        walls[3].position = SCNVector3(-Float(roomWidth/2), 0, 0)
        walls[3].eulerAngles = SCNVector3(-Float.pi/2, Float.pi/2, Float.pi/2)
        walls[3].name = "LeftWall"
        
        // Floor
        walls[4].position = SCNVector3(0, -Float(roomHeight/2), 0)
        walls[4].name = "Floor"
        
        // Ceiling
        walls[5].position = SCNVector3(0, Float(roomHeight/2), 0)
        walls[5].name = "Ceiling"
        
        // Add all walls to scene
        for wall in walls {
            scene.rootNode.addChildNode(wall)
            print("  ✅ Added \(wall.name ?? "wall") at position: \(wall.position)")
        }
        
        // Camera positioned at top left middle of room
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.fieldOfView = 60
        cameraNode.position = SCNVector3(-Float(roomWidth) * 0.8, Float(roomHeight) * 0.8, Float(roomDepth) * 1.2)
        cameraNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(cameraNode)
        print("📷 Camera at: \(cameraNode.position)")
        
        // Add ambient light
        let lightNode = SCNNode()
        lightNode.light = SCNLight()
        lightNode.light?.type = .ambient
        lightNode.light?.color = UIColor.white
        lightNode.light?.intensity = 1000
        scene.rootNode.addChildNode(lightNode)
        
        // Get documents directory
        guard let documentsPath = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first else {
            print("❌ Could not access Documents directory")
            return nil
        }
        
        let exportURL = documentsPath.appendingPathComponent(fileName)
        
        // Delete existing file if exists
        if FileManager.default.fileExists(atPath: exportURL.path) {
            try? FileManager.default.removeItem(at: exportURL)
        }
        
        // Export scene
        print("📤 Exporting USDZ...")
        let success = scene.write(
            to: exportURL,
            options: nil,
            delegate: nil,
            progressHandler: { progress, error, _ in
                print("  Progress: \(Int(progress * 100))%")
            }
        )
        
        if success {
            Thread.sleep(forTimeInterval: 0.2)
            
            if FileManager.default.fileExists(atPath: exportURL.path),
               let fileSize = try? FileManager.default.attributesOfItem(atPath: exportURL.path)[.size] as? Int,
               fileSize > 0 {
                print("✅ Success! Size: \(fileSize) bytes")
                return exportURL
            }
        }
        
        print("❌ Export failed")
        return nil
    }
    
    // Create wall - single plane with isDoubleSided
    private static func createWall(width: CGFloat, height: CGFloat, image: UIImage, name: String = "Wall") -> SCNNode {
        print("  Creating \(name): \(width)x\(height)")
        
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
    
    // Create floor - single plane with isDoubleSided
    private static func createFloor(width: CGFloat, depth: CGFloat, image: UIImage) -> SCNNode {
        print("  Creating Floor: \(width)x\(depth)")
        
        let plane = SCNPlane(width: width, height: depth)
        
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.isDoubleSided = true
        material.lightingModel = .constant
        
        plane.materials = [material]
        
        let node = SCNNode(geometry: plane)
        node.eulerAngles = SCNVector3(-Float.pi/2, 0, 0)
        node.name = "Floor"
        
        return node
    }
    
    // Create ceiling - single plane with isDoubleSided
    private static func createCeiling(width: CGFloat, depth: CGFloat, image: UIImage) -> SCNNode {
        print("  Creating Ceiling: \(width)x\(depth)")
        
        let plane = SCNPlane(width: width, height: depth)
        
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.isDoubleSided = true
        material.lightingModel = .constant
        
        plane.materials = [material]
        
        let node = SCNNode(geometry: plane)
        node.eulerAngles = SCNVector3(Float.pi/2, 0, 0)
        node.name = "Ceiling"
        
        return node
    }
}

// MARK: - DollhouseRoomScannerView (unchanged from your file)
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
                capturedPhotos.append(image)
                currentImage = nil
                currentPhotoIndex += 1
                
                if currentPhotoIndex >= 6 {
                    processPhotos()
                }
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
            
            VStack(spacing: 20) {
                Text(photoInstructions[currentPhotoIndex])
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                
                Text("Capture all details")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.7))
            }
            .padding()
            .background(Color.black.opacity(0.5))
            .cornerRadius(16)
            
            Spacer()
            
            Button(action: {
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
        
        NotificationCenter.default.post(
            name: USDZModelManager.didAddModelNotification,
            object: nil,
            userInfo: ["fileName": fileName]
        )
        
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

// MARK: - Enhanced Dollhouse Scanner with Proper Integration
//struct DollhouseRoomScannerView: View {
//    @Binding var isPresented: Bool
//    @ObservedObject var modelManager: USDZModelManager
//    @State private var capturedPhotos: [UIImage] = []
//    @State private var showingCamera = false
//    @State private var isProcessing = false
//    @State private var currentPhotoIndex = 0
//    @State private var currentImage: UIImage?
//    @State private var generatedModelURL: URL?
//    @State private var scanComplete = false
//    @State private var errorMessage: String?
//    
//    let photoInstructions = [
//        "📸 FRONT WALL - Stand at back, capture front wall",
//        "📸 RIGHT WALL - Turn right, capture right wall",
//        "📸 BACK WALL - Turn around, capture back wall",
//        "📸 LEFT WALL - Turn left, capture left wall",
//        "📸 FLOOR - Point camera down at floor",
//        "📸 CEILING - Point camera up at ceiling"
//    ]
//    
//    var body: some View {
//        NavigationView {
//            ZStack {
//                // Background
//                LinearGradient(
//                    gradient: Gradient(colors: [.black, .blue.opacity(0.3)]),
//                    startPoint: .top,
//                    endPoint: .bottom
//                )
//                .ignoresSafeArea()
//                
//                if let error = errorMessage {
//                    errorView(message: error)
//                } else if currentPhotoIndex < 6 && !scanComplete {
//                    instructionsView
//                } else if isProcessing {
//                    processingView
//                } else if scanComplete {
//                    successView
//                }
//            }
//            .navigationTitle("2.5D Room Capture")
//            .navigationBarTitleDisplayMode(.inline)
//            .toolbar {
//                ToolbarItem(placement: .navigationBarLeading) {
//                    Button("Cancel") {
//                        isPresented = false
//                    }
//                    .foregroundColor(.white)
//                }
//            }
//            .preferredColorScheme(.dark)
//        }
//        .sheet(isPresented: $showingCamera, onDismiss: {
//            if let image = currentImage {
//                print("📸 Photo \(currentPhotoIndex + 1) captured: \(image.size)")
//                capturedPhotos.append(image)
//                currentImage = nil
//                currentPhotoIndex += 1
//                
//                if currentPhotoIndex >= 6 {
//                    print("✅ All 6 photos captured, starting processing...")
//                    processPhotos()
//                }
//            }
//        }) {
//            ImagePicker(image: $currentImage, sourceType: .camera)
//        }
//    }
//    
//    private var instructionsView: some View {
//        VStack(spacing: 20) {
//            // Progress
//            HStack {
//                Text("Photo \(currentPhotoIndex + 1) of 6")
//                    .font(.headline)
//                    .foregroundColor(.white)
//                
//                Spacer()
//                
//                HStack(spacing: 8) {
//                    ForEach(0..<6) { index in
//                        Circle()
//                            .fill(index < capturedPhotos.count ? Color.green : Color.white.opacity(0.3))
//                            .frame(width: 12, height: 12)
//                    }
//                }
//            }
//            .padding()
//            
//            Spacer()
//            
//            VStack(spacing: 20) {
//                Text(photoInstructions[currentPhotoIndex])
//                    .font(.title3)
//                    .fontWeight(.bold)
//                    .foregroundColor(.white)
//                    .multilineTextAlignment(.center)
//                
//                Text("Capture all details: tiles, curtains, furniture")
//                    .font(.caption)
//                    .foregroundColor(.white.opacity(0.7))
//            }
//            .padding()
//            .background(Color.black.opacity(0.5))
//            .cornerRadius(16)
//            
//            Spacer()
//            
//            Button(action: {
//                print("👆 Take Photo button pressed for photo \(currentPhotoIndex + 1)")
//                showingCamera = true
//            }) {
//                HStack {
//                    Image(systemName: "camera.fill")
//                    Text("Take Photo")
//                }
//                .font(.headline)
//                .foregroundColor(.white)
//                .padding(.horizontal, 40)
//                .padding(.vertical, 16)
//                .background(
//                    LinearGradient(
//                        gradient: Gradient(colors: [.blue, .purple]),
//                        startPoint: .leading,
//                        endPoint: .trailing
//                    )
//                )
//                .cornerRadius(25)
//            }
//            .padding(.bottom, 50)
//            
//            // Photo preview strip
//            if !capturedPhotos.isEmpty {
//                ScrollView(.horizontal, showsIndicators: false) {
//                    HStack(spacing: 8) {
//                        ForEach(Array(capturedPhotos.enumerated()), id: \.offset) { index, photo in
//                            Image(uiImage: photo)
//                                .resizable()
//                                .aspectRatio(contentMode: .fill)
//                                .frame(width: 60, height: 60)
//                                .clipShape(RoundedRectangle(cornerRadius: 8))
//                                .overlay(
//                                    Text("\(index + 1)")
//                                        .font(.caption2)
//                                        .foregroundColor(.white)
//                                        .padding(2)
//                                        .background(Color.black.opacity(0.7))
//                                        .cornerRadius(4)
//                                        .padding(2),
//                                    alignment: .topLeading
//                                )
//                        }
//                    }
//                    .padding(.horizontal)
//                }
//                .frame(height: 80)
//                .background(Color.black.opacity(0.5))
//            }
//        }
//    }
//    
//    private var processingView: some View {
//        VStack(spacing: 20) {
//            ProgressView()
//                .progressViewStyle(CircularProgressViewStyle(tint: .white))
//                .scaleEffect(2)
//            
//            Text("Creating 2.5D Room")
//                .font(.title)
//                .fontWeight(.bold)
//                .foregroundColor(.white)
//            
//            Text("Building your room from 6 photos...")
//                .font(.body)
//                .foregroundColor(.white.opacity(0.8))
//        }
//        .padding(40)
//        .background(Color.black.opacity(0.8))
//        .cornerRadius(20)
//    }
//    
//    private var successView: some View {
//        VStack(spacing: 30) {
//            Image(systemName: generatedModelURL != nil ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
//                .font(.system(size: 80))
//                .foregroundColor(generatedModelURL != nil ? .green : .orange)
//            
//            Text(generatedModelURL != nil ? "2.5D Room Created!" : "Error Creating Room")
//                .font(.largeTitle)
//                .fontWeight(.bold)
//                .foregroundColor(.white)
//            
//            if let url = generatedModelURL {
//                VStack(spacing: 10) {
//                    Text("USDZ file saved successfully!")
//                        .font(.body)
//                        .foregroundColor(.white.opacity(0.9))
//                    
//                    Text(url.lastPathComponent)
//                        .font(.caption)
//                        .foregroundColor(.white.opacity(0.7))
//                        .multilineTextAlignment(.center)
//                        .padding(.horizontal)
//                }
//            } else {
//                Text("Room creation failed. Please try again.")
//                    .font(.body)
//                    .foregroundColor(.orange)
//                    .multilineTextAlignment(.center)
//                    .padding()
//            }
//            
//            Button(action: {
//                print("🏠 Dismissing scanner - returning to home view")
//                isPresented = false
//            }) {
//                Text(generatedModelURL != nil ? "View in Collection" : "Go Back")
//                    .font(.headline)
//                    .foregroundColor(.white)
//                    .frame(width: 200, height: 50)
//                    .background(generatedModelURL != nil ? Color.green : Color.blue)
//                    .cornerRadius(25)
//            }
//        }
//        .padding()
//    }
//    
//    private func errorView(message: String) -> some View {
//        VStack(spacing: 20) {
//            Image(systemName: "exclamationmark.triangle")
//                .font(.system(size: 60))
//                .foregroundColor(.red)
//            
//            Text("Error")
//                .font(.title)
//                .foregroundColor(.white)
//            
//            Text(message)
//                .font(.body)
//                .foregroundColor(.white.opacity(0.8))
//                .multilineTextAlignment(.center)
//                .padding()
//            
//            Button("Try Again") {
//                errorMessage = nil
//                resetScanner()
//            }
//            .padding()
//            .background(Color.blue)
//            .foregroundColor(.white)
//            .cornerRadius(10)
//        }
//        .padding()
//    }
//    
//    private func processPhotos() {
//        isProcessing = true
//        print("\n🎬 PROCESSING START")
//        print("📊 Photo count: \(capturedPhotos.count)")
//        
//        DispatchQueue.global(qos: .userInitiated).async {
//            let timestamp = Date().timeIntervalSince1970
//            let fileName = "dollhouse_\(timestamp).usdz"
//            print("📝 Generating filename: \(fileName)")
//            
//            let url = DollhouseCreator.createDollhouseFile(
//                from: capturedPhotos,
//                fileName: fileName
//            )
//            
//            DispatchQueue.main.async {
//                generatedModelURL = url
//                
//                if let url = url {
//                    print("✅ SUCCESS: USDZ created at \(url.path)")
//                    saveToCollection(usdzURL: url, fileName: fileName)
//                } else {
//                    print("❌ FAILURE: USDZ creation failed")
//                    errorMessage = "Failed to create 2.5D room model"
//                }
//                
//                isProcessing = false
//                scanComplete = true
//            }
//        }
//    }
//    
//    private func saveToCollection(usdzURL: URL, fileName: String) {
//        print("\n💾 SAVING TO COLLECTION")
//        print("  File: \(fileName)")
//        print("  Full path: \(usdzURL.path)")
//        print("  Models before: \(modelManager.models.count)")
//        
//        // Verify file exists before notifying
//        guard FileManager.default.fileExists(atPath: usdzURL.path) else {
//            print("❌ File doesn't exist at path: \(usdzURL.path)")
//            errorMessage = "File was created but cannot be found"
//            return
//        }
//        
//        // Get file size for verification
//        if let fileSize = try? FileManager.default.attributesOfItem(atPath: usdzURL.path)[.size] as? Int {
//            print("  ✅ File verified: \(fileSize) bytes")
//        }
//        
//        // Notify the model manager to add this new model
//        print("📢 Posting notification to USDZModelManager...")
//        NotificationCenter.default.post(
//            name: USDZModelManager.didAddModelNotification,
//            object: nil,
//            userInfo: ["fileName": fileName]
//        )
//        
//        // Also trigger a refresh for good measure
//        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
//            print("🔄 Triggering manual refresh...")
//            self.modelManager.refreshModels()
//            print("  Models after refresh: \(self.modelManager.models.count)")
//            
//            // List the models to verify
//            print("📋 Current models:")
//            for (index, model) in self.modelManager.models.enumerated() {
//                print("  \(index + 1). \(model.displayName) - \(model.fileName)")
//            }
//        }
//        
//        print("✅ Save process complete")
//    }
//    
//    private func resetScanner() {
//        capturedPhotos = []
//        currentPhotoIndex = 0
//        scanComplete = false
//        generatedModelURL = nil
//        errorMessage = nil
//        print("🔄 Scanner reset")
//    }
//}

// MARK: - Image Picker
//struct ImagePicker: UIViewControllerRepresentable {
//    @Binding var image: UIImage?
//    let sourceType: UIImagePickerController.SourceType
//    
//    func makeUIViewController(context: Context) -> UIImagePickerController {
//        let picker = UIImagePickerController()
//        picker.sourceType = sourceType
//        picker.delegate = context.coordinator
//        return picker
//    }
//    
//    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
//    
//    func makeCoordinator() -> Coordinator {
//        Coordinator(self)
//    }
//    
//    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
//        let parent: ImagePicker
//        
//        init(_ parent: ImagePicker) {
//            self.parent = parent
//        }
//        
//        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
//            if let image = info[.originalImage] as? UIImage {
//                parent.image = image
//            }
//            picker.dismiss(animated: true)
//        }
//        
//        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
//            picker.dismiss(animated: true)
//        }
//    }
//}
