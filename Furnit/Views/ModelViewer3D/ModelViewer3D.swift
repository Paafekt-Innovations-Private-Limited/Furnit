import SwiftUI
import SceneKit
import AVFoundation
import CoreML
import Vision

// MARK: - 3D Model Viewer (Renamed to avoid conflicts)
struct ModelViewer3D: View {
    @State private var selectedModel: Room3DModel?
    @State private var showRoomCapture = false
    @State private var capturedRooms: [Room3DModel] = []
    @State private var showARScanner = false
    
    // Pre-defined models
    let preDefinedModels = [
        Room3DModel(id: UUID(), name: "Living Room", type: .predefined, icon: "sofa"),
        Room3DModel(id: UUID(), name: "Bedroom", type: .predefined, icon: "bed.double"),
        Room3DModel(id: UUID(), name: "Kitchen", type: .predefined, icon: "refrigerator"),
        Room3DModel(id: UUID(), name: "Office", type: .predefined, icon: "desktopcomputer")
    ]
    
    var allModels: [Room3DModel] {
        preDefinedModels + capturedRooms
    }
    
    var body: some View {
        ZStack {
            // Main 3D View
            if let model = selectedModel {
                Room3DSceneView(model: model)
                    .edgesIgnoringSafeArea(.all)
            } else {
                Empty3DRoomView()
            }
            
            // Top Bar - Room Selector
            VStack {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 15) {
                        // Pre-defined rooms
                        ForEach(preDefinedModels) { model in
                            Room3DModelCard(
                                model: model,
                                isSelected: selectedModel?.id == model.id,
                                action: { selectedModel = model }
                            )
                        }
                        
                        // Divider
                        if !capturedRooms.isEmpty {
                            Divider()
                                .frame(height: 60)
                        }
                        
                        // User captured rooms
                        ForEach(capturedRooms) { model in
                            Room3DModelCard(
                                model: model,
                                isSelected: selectedModel?.id == model.id,
                                action: { selectedModel = model }
                            )
                        }
                    }
                    .padding()
                }
                .background(
                    VisualEffectBlur(style: .systemUltraThinMaterialDark)
                        .opacity(0.9)
                )
                
                Spacer()
            }
            
            // Bottom Left - Camera Button
            VStack {
                Spacer()
                
                HStack {
                    // Camera Capture Button
                    Button(action: {
                        showRoomCapture = true
                    }) {
                        ZStack {
                            Circle()
                                .fill(
                                    LinearGradient(
                                        gradient: Gradient(colors: [Color.blue, Color.purple]),
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                                .frame(width: 60, height: 60)
                            
                            Image(systemName: "camera.fill")
                                .font(.system(size: 24))
                                .foregroundColor(.white)
                        }
                        .shadow(radius: 5)
                    }
                    .padding(.leading, 20)
                    .padding(.bottom, 30)
                    
                    Spacer()
                }
            }
        }
        .sheet(isPresented: $showRoomCapture) {
            Room3DCaptureFlow(onComplete: { newRoom in
                capturedRooms.append(newRoom)
                selectedModel = newRoom
            })
        }
        .onAppear {
            if selectedModel == nil {
                selectedModel = preDefinedModels.first
            }
        }
    }
}

// MARK: - Room 3D Model (Renamed)
struct Room3DModel: Identifiable {
    let id: UUID
    let name: String
    let type: ModelType
    let icon: String
    var capturedImages: RoomImageSet?
    var generatedGeometry: Generated3DModel?
    
    enum ModelType {
        case predefined
        case captured
        case arScanned
    }
}

struct RoomImageSet {
    var front: UIImage?
    var back: UIImage?
    var left: UIImage?
    var right: UIImage?
    var floor: UIImage?
    var ceiling: UIImage?
}

struct Generated3DModel {
    let geometry: SCNNode
    let textures: [String: UIImage]
    let detectedElements: DetectedRoomElements
}

struct DetectedRoomElements {
    var hasTiles: Bool = false
    var hasCurtains: Bool = false
    var floorType: String = "Unknown"
    var wallColor: UIColor = .white
}

// MARK: - Room Model Card (Renamed)
struct Room3DModelCard: View {
    let model: Room3DModel
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(isSelected ? Color.blue : Color.gray.opacity(0.3))
                        .frame(width: 70, height: 70)
                    
                    if model.type == .captured {
                        if let floorImage = model.capturedImages?.floor {
                            Image(uiImage: floorImage)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: 65, height: 65)
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                        } else {
                            Image(systemName: "camera.fill")
                                .font(.system(size: 24))
                                .foregroundColor(.white)
                        }
                    } else {
                        Image(systemName: model.icon)
                            .font(.system(size: 24))
                            .foregroundColor(.white)
                    }
                    
                    if model.type == .captured {
                        VStack {
                            HStack {
                                Spacer()
                                Circle()
                                    .fill(Color.green)
                                    .frame(width: 16, height: 16)
                                    .overlay(
                                        Image(systemName: "checkmark")
                                            .font(.system(size: 10))
                                            .foregroundColor(.white)
                                    )
                            }
                            Spacer()
                        }
                        .padding(4)
                    }
                }
                
                Text(model.name)
                    .font(.caption)
                    .lineLimit(1)
                    .foregroundColor(isSelected ? .blue : .primary)
            }
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Room Capture Flow (Renamed)
struct Room3DCaptureFlow: View {
    let onComplete: (Room3DModel) -> Void
    @Environment(\.dismiss) var dismiss
    @State private var currentStep = 0
    @State private var capturedImages = RoomImageSet()
    @State private var roomName = ""
    @State private var isProcessing = false
    @State private var generatedModel: Generated3DModel?
    
    let captureSteps = [
        PhotoCaptureStep(name: "Front Wall", icon: "arrow.up", key: "front"),
        PhotoCaptureStep(name: "Back Wall", icon: "arrow.down", key: "back"),
        PhotoCaptureStep(name: "Left Wall", icon: "arrow.left", key: "left"),
        PhotoCaptureStep(name: "Right Wall", icon: "arrow.right", key: "right"),
        PhotoCaptureStep(name: "Floor", icon: "arrow.down.square", key: "floor"),
        PhotoCaptureStep(name: "Ceiling", icon: "arrow.up.square", key: "ceiling")
    ]
    
    struct PhotoCaptureStep {
        let name: String
        let icon: String
        let key: String
    }
    
    var body: some View {
        NavigationView {
            VStack {
                if currentStep < captureSteps.count {
                    // Capture Phase
                    RoomPhotoCaptureView(
                        step: captureSteps[currentStep],
                        onCapture: { image in
                            saveImage(image, for: captureSteps[currentStep].key)
                            if currentStep < captureSteps.count - 1 {
                                currentStep += 1
                            } else {
                                processImages()
                            }
                        }
                    )
                } else if isProcessing {
                    // Processing Phase
                    ModelProcessingView()
                } else if generatedModel != nil {
                    // Review Phase
                    Review3DModelView(
                        model: generatedModel!,
                        roomName: $roomName,
                        onSave: saveRoom,
                        onRetake: retakePhotos
                    )
                }
            }
            .navigationTitle("Capture Room")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    if currentStep < captureSteps.count {
                        Text("\(currentStep + 1)/\(captureSteps.count)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
    }
    
    func saveImage(_ image: UIImage, for key: String) {
        switch key {
        case "front": capturedImages.front = image
        case "back": capturedImages.back = image
        case "left": capturedImages.left = image
        case "right": capturedImages.right = image
        case "floor": capturedImages.floor = image
        case "ceiling": capturedImages.ceiling = image
        default: break
        }
    }
    
    func processImages() {
        isProcessing = true
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            let processor = Room3DProcessor()
            generatedModel = processor.generateModel(from: capturedImages)
            isProcessing = false
            currentStep += 1
        }
    }
    
    func saveRoom() {
        let newRoom = Room3DModel(
            id: UUID(),
            name: roomName.isEmpty ? "Room \(Date().timeIntervalSince1970)" : roomName,
            type: .captured,
            icon: "cube",
            capturedImages: capturedImages,
            generatedGeometry: generatedModel
        )
        onComplete(newRoom)
        dismiss()
    }
    
    func retakePhotos() {
        currentStep = 0
        capturedImages = RoomImageSet()
        generatedModel = nil
    }
}

// MARK: - Room Photo Capture View (Renamed)
struct RoomPhotoCaptureView: View {
    let step: Room3DCaptureFlow.PhotoCaptureStep
    let onCapture: (UIImage) -> Void
    @State private var capturedImage: UIImage?
    @State private var showingImagePicker = true
    
    var body: some View {
        ZStack {
            if let image = capturedImage {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                
                VStack {
                    Spacer()
                    
                    HStack(spacing: 30) {
                        Button(action: {
                            capturedImage = nil
                            showingImagePicker = true
                        }) {
                            Label("Retake", systemImage: "arrow.counterclockwise")
                                .padding()
                                .background(Color.orange)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                        
                        Button(action: {
                            if let image = capturedImage {
                                onCapture(image)
                            }
                        }) {
                            Label("Use Photo", systemImage: "checkmark.circle")
                                .padding()
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    }
                    .padding(.bottom, 30)
                }
            } else {
                VStack(spacing: 20) {
                    Image(systemName: step.icon)
                        .font(.system(size: 80))
                        .foregroundColor(.blue)
                    
                    Text("Capture \(step.name)")
                        .font(.title2)
                    
                    Text("Position camera to capture the \(step.name.lowercased())")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                }
            }
        }
        .sheet(isPresented: $showingImagePicker) {
            ImagePickerView(image: $capturedImage)
        }
    }
}

// MARK: - Model Processing View (Renamed)
struct ModelProcessingView: View {
    @State private var progress: Double = 0
    @State private var currentTask = "Analyzing images..."
    
    let tasks = [
        "Analyzing images...",
        "Detecting floor patterns...",
        "Identifying materials...",
        "Finding curtains and windows...",
        "Generating 3D model...",
        "Applying textures..."
    ]
    
    var body: some View {
        VStack(spacing: 30) {
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle())
                .scaleEffect(2)
            
            Text("Processing Room")
                .font(.title)
                .fontWeight(.bold)
            
            Text(currentTask)
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            ProgressView(value: progress)
                .progressViewStyle(LinearProgressViewStyle())
                .padding(.horizontal, 40)
        }
        .onAppear {
            animateProgress()
        }
    }
    
    func animateProgress() {
        for (index, task) in tasks.enumerated() {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(index) * 0.5) {
                currentTask = task
                withAnimation {
                    progress = Double(index + 1) / Double(tasks.count)
                }
            }
        }
    }
}

// MARK: - Review 3D Model View (Renamed)
struct Review3DModelView: View {
    let model: Generated3DModel
    @Binding var roomName: String
    let onSave: () -> Void
    let onRetake: () -> Void
    
    var body: some View {
        VStack {
            // 3D Preview
            SceneView(
                scene: createPreviewScene(),
                options: [.allowsCameraControl, .autoenablesDefaultLighting]
            )
            .frame(height: 300)
            .cornerRadius(15)
            .padding()
            
            // Detected Features
            VStack(alignment: .leading, spacing: 15) {
                Text("Detected Features")
                    .font(.headline)
                
                RoomFeatureRow(
                    icon: "square.grid.3x3",
                    title: "Floor Type",
                    value: model.detectedElements.floorType,
                    hasFeature: model.detectedElements.hasTiles
                )
                
                RoomFeatureRow(
                    icon: "curtains.closed",
                    title: "Window Treatment",
                    value: model.detectedElements.hasCurtains ? "Curtains" : "None",
                    hasFeature: model.detectedElements.hasCurtains
                )
            }
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(10)
            .padding(.horizontal)
            
            TextField("Room Name", text: $roomName)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding(.horizontal)
            
            HStack(spacing: 20) {
                Button(action: onRetake) {
                    Label("Retake", systemImage: "arrow.counterclockwise")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.orange)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                
                Button(action: onSave) {
                    Label("Save Room", systemImage: "checkmark.circle")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
            }
            .padding(.horizontal)
            
            Spacer()
        }
    }
    
    func createPreviewScene() -> SCNScene {
        let scene = SCNScene()
        scene.rootNode.addChildNode(model.geometry)
        
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(x: 0, y: 2, z: 5)
        scene.rootNode.addChildNode(cameraNode)
        
        let lightNode = SCNNode()
        lightNode.light = SCNLight()
        lightNode.light?.type = .omni
        lightNode.position = SCNVector3(x: 0, y: 5, z: 0)
        scene.rootNode.addChildNode(lightNode)
        
        return scene
    }
}

// MARK: - Room Feature Row (Renamed)
struct RoomFeatureRow: View {
    let icon: String
    let title: String
    let value: String
    let hasFeature: Bool
    var color: UIColor?
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(hasFeature ? .green : .gray)
                .frame(width: 30)
            
            Text(title)
                .font(.subheadline)
            
            Spacer()
            
            Text(value)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Room 3D Scene View (Renamed)
struct Room3DSceneView: UIViewRepresentable {
    let model: Room3DModel
    
    func makeUIView(context: Context) -> SCNView {
        let sceneView = SCNView()
        
        if let generatedModel = model.generatedGeometry {
            let scene = SCNScene()
            scene.rootNode.addChildNode(generatedModel.geometry)
            sceneView.scene = scene
        } else {
            sceneView.scene = createPlaceholderScene(for: model)
        }
        
        sceneView.allowsCameraControl = true
        sceneView.autoenablesDefaultLighting = true
        sceneView.backgroundColor = .black
        
        return sceneView
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {}
    
    func createPlaceholderScene(for model: Room3DModel) -> SCNScene {
        let scene = SCNScene()
        
        let box = SCNBox(width: 4, height: 2.5, length: 4, chamferRadius: 0)
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.systemBlue
        box.materials = [material]
        
        let node = SCNNode(geometry: box)
        scene.rootNode.addChildNode(node)
        
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(x: 0, y: 2, z: 8)
        scene.rootNode.addChildNode(cameraNode)
        
        return scene
    }
}

// MARK: - Empty Room View (Renamed)
struct Empty3DRoomView: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "cube")
                .font(.system(size: 80))
                .foregroundColor(.gray)
            
            Text("No Room Selected")
                .font(.title2)
                .foregroundColor(.gray)
            
            Text("Select a room from above or capture a new one")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
    }
}

// MARK: - Room 3D Processor (Renamed)
class Room3DProcessor {
    func generateModel(from images: RoomImageSet) -> Generated3DModel {
        let roomNode = createTexturedRoom(from: images)
        let features = detectFeatures(from: images)
        
        return Generated3DModel(
            geometry: roomNode,
            textures: extractTextures(from: images),
            detectedElements: features
        )
    }
    
    private func createTexturedRoom(from images: RoomImageSet) -> SCNNode {
        let box = SCNBox(width: 4, height: 2.5, length: 4, chamferRadius: 0)
        
        var materials: [SCNMaterial] = []
        let imageOrder = [images.floor, images.ceiling, images.front,
                         images.back, images.left, images.right]
        
        for image in imageOrder {
            let material = SCNMaterial()
            material.diffuse.contents = image ?? UIColor.gray
            materials.append(material)
        }
        
        box.materials = materials
        return SCNNode(geometry: box)
    }
    
    private func detectFeatures(from images: RoomImageSet) -> DetectedRoomElements {
        var features = DetectedRoomElements()
        
        if images.floor != nil {
            features.hasTiles = Bool.random() // Replace with ML
            features.floorType = features.hasTiles ? "Ceramic Tiles" : "Hardwood"
        }
        
        if images.front != nil {
            features.hasCurtains = Bool.random() // Replace with ML
        }
        
        features.wallColor = UIColor(
            red: .random(in: 0.5...0.9),
            green: .random(in: 0.5...0.9),
            blue: .random(in: 0.5...0.9),
            alpha: 1.0
        )
        
        return features
    }
    
    private func extractTextures(from images: RoomImageSet) -> [String: UIImage] {
        var textures: [String: UIImage] = [:]
        
        if let floor = images.floor { textures["floor"] = floor }
        if let ceiling = images.ceiling { textures["ceiling"] = ceiling }
        
        return textures
    }
}

// MARK: - Visual Effect Blur (Renamed)
struct VisualEffectBlur: UIViewRepresentable {
    var style: UIBlurEffect.Style
    
    func makeUIView(context: Context) -> UIVisualEffectView {
        UIVisualEffectView(effect: UIBlurEffect(style: style))
    }
    
    func updateUIView(_ uiView: UIVisualEffectView, context: Context) {}
}

// MARK: - Image Picker View (Renamed)
struct ImagePickerView: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.dismiss) var dismiss
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .camera
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePickerView
        
        init(_ parent: ImagePickerView) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController,
                                  didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage {
                parent.image = uiImage
            }
            parent.dismiss()
        }
    }
}
