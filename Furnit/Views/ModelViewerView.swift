import SwiftUI
import RealityKit
import Combine
import Photos
import CoreML
import AVFoundation
import UIKit

struct ModelViewerView: View {
    @State private var mlModel: MLModel? = nil
    @Environment(\.horizontalSizeClass) var horizontalSizeClass
    @Environment(\.verticalSizeClass) var verticalSizeClass
    let model: USDZModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.presentationMode) var presentationMode

    // Camera movement state
    @StateObject private var cameraMovementManager = RealityKitCameraMovementManager()
    @State private var shouldResetCamera = false  // ✅ Trigger camera reset on appear

    // Required for RealityKitView
    @StateObject private var arObjectPlacementManager = RealityKitObjectPlacementManager()
    @State private var isARActive = false

    // Camera/Segmentation state
    @State private var showingCameraPreview = false
    @State private var showingSegmentExamine = false
    @State private var showingSegmentForeground = false
    @State private var showingSegmentFurniture = false
    @State private var showingSmartyPants = false  // NEW: YOLOE
    @State private var capturedImage: UIImage? = nil
    @State private var roomSnapshot: UIImage? = nil
    
    // ARView snapshot trigger (proper way to capture 3D content)
    @State private var shouldCaptureARViewSnapshot = false
    
    // Furniture hint
    @State private var showFurnitureHint = true

    @State private var isCapturingSnapshot = false

//    // Edge fill mode for SmartyPants segmentation
//    @State private var selectedEdgeFillMode: EdgeFillMode = .chairType

    init(model: USDZModel) {
        self.model = model
    }

    private var anyCameraOverlayActive: Bool {
        showingCameraPreview || showingSegmentExamine || showingSegmentForeground || showingSegmentFurniture || showingSmartyPants
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Main content ZStack
                ZStack {
                    RealityKitView(
                        model: model,
                        cameraMovementManager: cameraMovementManager,
                        arObjectPlacementManager: arObjectPlacementManager,
                        isARActive: isARActive,
                        shouldCaptureSnapshot: $shouldCaptureARViewSnapshot,
                        capturedSnapshot: $roomSnapshot,
                        shouldResetCamera: $shouldResetCamera  // ✅ Camera reset trigger
                    )
                    .allowsHitTesting(!(showingCameraPreview || showingSegmentExamine || showingSegmentForeground || showingSegmentFurniture || showingSmartyPants))
                    .ignoresSafeArea(.all)
                    // NEW: SmartyPants overlay
                    if showingSmartyPants {
                        SmartyPantsUIView(
                            capturedImage: $capturedImage,
                            roomImage: roomSnapshot,
                            mlModel: mlModel,
                            processInterval: 0.07,
                            active: true
                        )
                        .zIndex(9000)
                    }

                    if isLandscape(geometry: geometry) {
                        landscapeControls
                    } else {
                        portraitControls
                    }
                }
                
                // TOPMOST BACK BUTTON - ALWAYS ON TOP
                VStack {
                    HStack {
                        backButton
                            .allowsHitTesting(true)
                        Spacer()
                    }
                    .padding()
                    Spacer()
                }
                .opacity(isCapturingSnapshot ? 0 : 1)
                .zIndex(99999) // HIGHEST POSSIBLE Z-INDEX
                .allowsHitTesting(true)
                
                // TOPMOST BRAIN ICON - ALWAYS ON TOP
                VStack {
                    Spacer()
                    HStack {
                        VStack(spacing: 16) {
                            // Brain button
                            Button(action: {
                                logDebug("BRAIN FLOW: tap received")
                                // Dismiss hint on first touch
                                showFurnitureHint = false
                                
                                if showingSmartyPants {
                                    showingSmartyPants = false
                                } else {
                                    logDebug("BRAIN FLOW: requesting snapshot")
                                    // Trigger ARView snapshot
                                    shouldCaptureARViewSnapshot = true
                                    
                                    // Hide other overlays
                                    showingCameraPreview = false
                                    showingSegmentExamine = false
                                    showingSegmentForeground = false
                                    showingSegmentFurniture = false
                                    
                                    // Wait briefly for snapshot to complete
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                                        logDebug("BRAIN FLOW: showing SmartyPants overlay")
                                        self.showingSmartyPants = true
                                    }
                                }
                            }) {
                                Image(systemName: "brain.head.profile")
                                    .font(.system(size: 28))
                                    .foregroundColor(.white)
                                    .frame(width: 60, height: 60)
                                    .background(Circle().fill(showingSmartyPants ? Color.green : Color.blue).shadow(radius: 5))
                            }
                            .contentShape(Circle())
                            .frame(width: 76, height: 76)
                        }
                        .padding(.leading, 16)
                        .padding(.bottom, 20)
                        Spacer()
                    }
                }
                .opacity(isCapturingSnapshot ? 0 : 1)
                .zIndex(99998) // SECOND HIGHEST Z-INDEX
                .allowsHitTesting(true)
                
                // ✅ GLOBAL JOYSTICK - uses GlobalCameraController
                SimpleJoystickOverlay()
                    .opacity(isCapturingSnapshot ? 0 : 1)
                    .zIndex(99997)

                // SmartyPants snapshot button (separate from joystick)
                if showingSmartyPants {
                    VStack {
                        Spacer()
                        HStack {
                            Spacer()
                            Button(action: {
                                let screen = UIScreen.main.bounds.size
                                saveSmartyPantsSnapshot(screen)
                            }) {
                                Image(systemName: "square.and.arrow.down")
                                    .font(.system(size: 28, weight: .regular))
                                    .foregroundColor(.white)
                                    .frame(width: 48, height: 48)
                                    .background(
                                        Circle()
                                            .fill(Color.blue)
                                    )
                            }
                            .disabled(isCapturingSnapshot)
                            .padding(.trailing, 30)
                            .padding(.bottom, 40)
                        }
                    }
                    .opacity(isCapturingSnapshot ? 0 : 1)
                    .zIndex(99996)
                }
            }
        }
        .navigationBarHidden(true)
        .statusBarHidden(true)
        .preferredColorScheme(.dark)
        .onChange(of: showingCameraPreview) { _, _ in manageARSessionForOverlays() }
        .onChange(of: showingSegmentExamine) { _, _ in manageARSessionForOverlays() }
        .onChange(of: showingSegmentForeground) { _, _ in manageARSessionForOverlays() }
        .onChange(of: showingSegmentFurniture) { _, _ in manageARSessionForOverlays() }
        .onChange(of: showingSmartyPants) { _, _ in manageARSessionForOverlays() }
        .onAppear {
            isARActive = true
            loadModelOnce()
            // ✅ Reset camera to optimal position every time view appears
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                shouldResetCamera = true
            }
        }
    }

    // MARK: - Overlays & Controls

    private func isLandscape(geometry: GeometryProxy) -> Bool {
        geometry.size.width > geometry.size.height
    }

    // PORTRAIT controls
    private var portraitControls: some View {
        VStack {
            Spacer()
            
            HStack(alignment: .bottom, spacing: 0) {
                VStack(spacing: 16) {
                    // cameraButton
                    // segmentExamineButton
                    // segmentForegroundButton
                    // segmentFurnitureButton
                    // smartyPantsButton - NOW AT TOPMOST LEVEL
                }
                .padding(.leading, 16)
                .padding(.bottom, 20)
                
                Spacer()
                
                // VirtualJoystick - NOW AT TOPMOST LEVEL
                
                Spacer()
            }
            .padding(.bottom, 20)
        }
    }

    // LANDSCAPE controls
    private var landscapeControls: some View {
        HStack {
            Spacer()
            
            VStack(spacing: 16) {
                // cameraButton
                // segmentExamineButton
                // segmentForegroundButton
                // segmentFurnitureButton
                // smartyPantsButton - NOW AT TOPMOST LEVEL
                Spacer()
                // VirtualJoystick - NOW AT TOPMOST LEVEL
                Spacer()
            }
            .padding()
        }
    }

    private var backButton: some View {
        Button(action: {
            // Try both dismiss methods for better compatibility
            if #available(iOS 15.0, *) {
                dismiss()
            } else {
                presentationMode.wrappedValue.dismiss()
            }
        }) {
            HStack(spacing: 6) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 16, weight: .medium))
                Text("Back")
                    .font(.system(size: 16, weight: .medium))
            }
            .foregroundColor(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(Color.black.opacity(0.7))
            .cornerRadius(20)
        }
    }

    private var cameraButton: some View {
        Button(action: {
            if showingCameraPreview {
                showingCameraPreview = false
            } else {
                showingSegmentExamine = false
                showingSegmentForeground = false
                showingSegmentFurniture = false
                showingSmartyPants = false
                showingCameraPreview = true
            }
        }) {
            Image(systemName: "camera.viewfinder")
                .font(.system(size: 28)).foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(Circle().fill(showingCameraPreview ? Color.green : Color.blue).shadow(radius: 5))
        }
    }

    private var segmentExamineButton: some View {
        Button(action: {
            showingCameraPreview = false
            showingSegmentForeground = false
            showingSegmentFurniture = false
            showingSmartyPants = false
            showingSegmentExamine.toggle()
        }) {
            Image(systemName: "crop.rotate")
                .font(.system(size: 28)).foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(Circle().fill(showingSegmentExamine ? Color.green : Color.blue).shadow(radius: 5))
        }
    }

    private var segmentForegroundButton: some View {
        Button(action: {
            showingCameraPreview = false
            showingSegmentExamine = false
            showingSegmentFurniture = false
            showingSmartyPants = false
            showingSegmentForeground.toggle()
        }) {
            Image(systemName: "viewfinder")
                .font(.system(size: 28)).foregroundColor(.white)
                .frame(width: 60, height: 60)
                .background(Circle().fill(showingSegmentForeground ? Color.purple : Color.blue).shadow(radius: 5))
        }
    }

    private var segmentFurnitureButton: some View {
        ZStack {
            Button(action: {
                // Dismiss hint on first touch
                showFurnitureHint = false
                
                if showingSegmentFurniture {
                    showingSegmentFurniture = false
                } else {
                    // Trigger ARView snapshot (will be handled by RealityKitView)
                    shouldCaptureARViewSnapshot = true
                    
                    // Hide other overlays
                    showingCameraPreview = false
                    showingSegmentExamine = false
                    showingSegmentForeground = false
                    showingSmartyPants = false
                    
                    // Wait briefly for snapshot to complete, then open scanner
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        self.showingSegmentFurniture = true
                    }
                }
            }) {
                Image(systemName: "chair.fill")
                    .font(.system(size: 28)).foregroundColor(.white)
                    .frame(width: 60, height: 60)
                    .background(Circle().fill(showingSegmentFurniture ? Color.green : Color.blue).shadow(radius: 5))
            }
            
            // Hint badge on button
            if showFurnitureHint {
                VStack {
                    HStack {
                        Spacer()
                        Image(systemName: "hand.tap.fill")
                            .font(.caption)
                            .foregroundColor(.white)
                            .padding(4)
                            .background(Circle().fill(Color.orange))
                    }
                    Spacer()
                }
                .frame(width: 60, height: 60)
            }
        }
    }

    // Edge fill mode toggle for SmartyPants
//    private var edgeFillModeToggle: some View {
//        HStack {
//            Spacer()
//            Picker("Edge Mode", selection: $selectedEdgeFillMode) {
//                Text("Cloth").tag(EdgeFillMode.clothBased)
//                Text("Chair").tag(EdgeFillMode.chairType)
//                Text("Furni").tag(EdgeFillMode.furniMaterial)
//            }
//            .pickerStyle(.segmented)
//            .frame(width: 240)
//            .padding(8)
//            .background(Color.black.opacity(0.6))
//            .cornerRadius(12)
//            .padding(.top, 60)
//            .padding(.trailing, 16)
//        }
//    }

    // NEW: SmartyPants Button (YOLOE - 168 furniture classes!)
    private var smartyPantsButton: some View {
        ZStack {
            Button(action: {
                // Dismiss hint on first touch
                showFurnitureHint = false
                
                if showingSmartyPants {
                    showingSmartyPants = false
                } else {
                    // Trigger ARView snapshot
                    shouldCaptureARViewSnapshot = true
                    
                    // Hide other overlays
                    showingCameraPreview = false
                    showingSegmentExamine = false
                    showingSegmentForeground = false
                    showingSegmentFurniture = false
                    
                    // Wait briefly for snapshot to complete
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                        self.showingSmartyPants = true
                    }
                }
            }) {
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 28)).foregroundColor(.white)
                    .frame(width: 60, height: 60)
                    .background(Circle().fill(showingSmartyPants ? Color.green : Color.blue).shadow(radius: 5))
            }
            
            // Hint badge on button
            if showFurnitureHint {
                VStack {
                    HStack {
                        Spacer()
                        Image(systemName: "hand.tap.fill")
                            .font(.caption)
                            .foregroundColor(.white)
                            .padding(4)
                            .background(Circle().fill(Color.orange))
                    }
                    Spacer()
                }
                .frame(width: 60, height: 60)
            }
        }
    }

    private func manageARSessionForOverlays() {
        let shouldRunAR = !anyCameraOverlayActive
        if isARActive != shouldRunAR {
            isARActive = shouldRunAR
        }
    }
    
    private func saveUIImageToPhotos(_ image: UIImage) {
        let saveBlock = {
            UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
            logDebug("✅ Saved image to Photos via UIImageWriteToSavedPhotosAlbum")
        }
        if #available(iOS 14, *) {
            let status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
            switch status {
            case .authorized, .limited:
                saveBlock()
            case .denied, .restricted:
                logDebug("❌ Photos add-only access denied or restricted")
            case .notDetermined:
                PHPhotoLibrary.requestAuthorization(for: .addOnly) { newStatus in
                    DispatchQueue.main.async {
                        if newStatus == .authorized || newStatus == .limited {
                            saveBlock()
                        } else {
                            logDebug("❌ Photos add-only access not granted")
                        }
                    }
                }
            @unknown default:
                logDebug("❌ Unknown Photos authorization status")
            }
        } else {
            // Fallback for iOS < 14
            let status = PHPhotoLibrary.authorizationStatus()
            switch status {
            case .authorized, .limited:
                saveBlock()
            case .denied, .restricted:
                logDebug("❌ Photos access denied or restricted")
            case .notDetermined:
                PHPhotoLibrary.requestAuthorization { newStatus in
                    DispatchQueue.main.async {
                        if newStatus == .authorized || newStatus == .limited {
                            saveBlock()
                        } else {
                            logDebug("❌ Photos access not granted")
                        }
                    }
                }
            @unknown default:
                logDebug("❌ Unknown Photos authorization status (legacy)")
            }
        }
    }
    
    private func saveSmartyPantsSnapshot(_ size: CGSize) {
        // Hide UI chrome briefly so it doesn't appear in the snapshot
        isCapturingSnapshot = true
        // Allow the UI to update before capturing
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.06) {
            guard let uiImage = captureAppWindowImage() else {
                isCapturingSnapshot = false
                logDebug("❌ Failed to capture app window image")
                return
            }
            self.saveUIImageToPhotos(uiImage)
            DispatchQueue.main.async {
                self.isCapturingSnapshot = false
            }
        }
    }
    
    private func captureAppWindowImage() -> UIImage? {
        // Prefer the windows from the active window scenes.  Avoid using
        // `UIApplication.shared.windows`, which was deprecated in iOS 15.  We
        // instead collect windows from all connected scenes and search for
        // the key window.  If none exists, we use the first available window.
        let windows: [UIWindow] = {
            // Gather windows from all connected scenes
            let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            let sceneWindows = scenes.flatMap { $0.windows }
            // Fallback: if no scenes produce windows (unlikely on modern iOS),
            // return an empty array.
            return sceneWindows
        }()
        guard let window = windows.first(where: { $0.isKeyWindow }) ?? windows.first else {
            return nil
        }
        let format = UIGraphicsImageRendererFormat()
        format.scale = UIScreen.main.scale
        let renderer = UIGraphicsImageRenderer(bounds: window.bounds, format: format)
        let image = renderer.image { _ in
            // Using drawHierarchy provides a rendered snapshot with effects
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }
        return image
    }
    
    private func loadModelOnce() {
        guard mlModel == nil else {
            logDebug("✅ ML Model already loaded")
            return
        }

        // Prefer .mlpackage (often smaller on memory footprint when loaded),
        // and try both 'l' and 's' variants. Adjust names as needed for your bundle.
        let candidateNames = [
            // ("yoloe-11l-seg-pf", "mlpackage"),
            ("yoloe-11l-seg-pf", "mlmodelc"),
            // ("yoloe-11s-seg-pf", "mlpackage"),
            // ("yoloe-11s-seg-pf", "mlmodelc")
        ]

        logDebug("🔍 Looking for ML Model (preferring .mlpackage):", candidateNames.map { "\($0.0).\($0.1)" }.joined(separator: ", "))

        var loaded: MLModel? = nil

        for (name, ext) in candidateNames {
            if let url = Bundle.main.url(forResource: name, withExtension: ext) {
                logDebug("📦 Found model file at: \(url.lastPathComponent)")
                do {
                    let config = MLModelConfiguration()
                    // Use CPU-only to avoid ANE "No memory object bound to port" crashes
                    config.computeUnits = .cpuOnly
                    let model = try MLModel(contentsOf: url, configuration: config)
                    loaded = model
                    logDebug("✅ Loaded MLModel '\(name).\(ext)' with computeUnits=\(config.computeUnits)")

                    // Log input constraint to verify shape & dtype (Float16 vs Float32)
                    let inputs = model.modelDescription.inputDescriptionsByName
                    if let img = inputs["image"] {
                        if let mac = img.multiArrayConstraint {
                            let shp = mac.shape.map { $0.intValue }
                            logDebug("📥 Model 'image' MultiArray constraint: shape=\(shp) dataType=\(mac.dataType)")
                            if mac.dataType == .float32 {
                                logDebug("⚠️ Model expects Float32 input — higher memory usage. Consider exporting an FP16 model to reduce memory.")
                            }
                        } else if let ic = img.imageConstraint {
                            logDebug("📥 Model 'image' Image constraint: \(ic.pixelsWide)x\(ic.pixelsHigh) pixelFormat=\(ic.pixelFormatType)")
                        } else {
                            logDebug("ℹ️ No detailed constraint for 'image' input")
                        }
                    }

                    break
                } catch {
                    logDebug("❌ Failed to load \(name).\(ext):", error)
                    continue
                }
            }
        }

        if let model = loaded {
            self.mlModel = model
        } else {
            logDebug("❌ No matching model found in bundle (tried candidates above)")
            // List bundle contents to help debug
            if let bundleContents = try? FileManager.default.contentsOfDirectory(atPath: Bundle.main.bundlePath) {
                logDebug("📁 Bundle contents (mlmodel/mlpackage only):")
                bundleContents.filter { $0.contains("mlmodel") || $0.contains("mlpackage") }.forEach { file in
                    logDebug("   - \(file)")
                }
            }
        }
    }
    

}

struct SmartyPantsUIView: UIViewRepresentable {
    @Binding var capturedImage: UIImage?
    
    var roomImage: UIImage?
    var mlModel: MLModel?
    var processInterval: Double = 0.07
    var scoreThreshold: Float = 0.25
    var active: Bool = true
    
    func makeUIView(context: Context) -> SmartyPantsContainerView {
        let view = SmartyPantsContainerView()
        view.setModel(mlModel)
        return view
    }

    func updateUIView(_ uiView: SmartyPantsContainerView, context: Context) {
        uiView.setModel(mlModel)
        uiView.processInterval = processInterval
//        uiView.scoreThreshold = scoreThreshold
        
        if active { uiView.startIfNeeded() } else { uiView.stop() }
    }
}

