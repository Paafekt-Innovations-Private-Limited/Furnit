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

    // Camera movement state
    @StateObject private var cameraMovementManager = RealityKitCameraMovementManager()
    @State private var joystickOffset: CGSize = .zero

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

    init(model: USDZModel) {
        self.model = model
    }

    private var anyCameraOverlayActive: Bool {
        showingCameraPreview || showingSegmentExamine || showingSegmentForeground || showingSegmentFurniture || showingSmartyPants
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                RealityKitView(
                    model: model,
                    cameraMovementManager: cameraMovementManager,
                    arObjectPlacementManager: arObjectPlacementManager,
                    isARActive: isARActive,
                    shouldCaptureSnapshot: $shouldCaptureARViewSnapshot,
                    capturedSnapshot: $roomSnapshot
                )
                .allowsHitTesting(!(showingCameraPreview || showingSegmentExamine || showingSegmentForeground || showingSegmentFurniture || showingSmartyPants))
                .ignoresSafeArea(.all)

                if showingCameraPreview {
                    SimpleCameraOverlay(
                        capturedImage: $capturedImage,
                        isShowingCamera: $showingCameraPreview
                    )
                    .zIndex(9000)
                }

                if showingSegmentExamine {
                    SegmentExamine(
                        capturedImage: $capturedImage,
                        isShowingCamera: $showingSegmentExamine
                    )
                    .zIndex(9999)
                }

                if showingSegmentForeground {
                    SegmentForeground(
                        capturedImage: $capturedImage,
                        isShowingCamera: $showingSegmentForeground
                    )
                    .zIndex(9000)
                }

                if showingSegmentFurniture {
                    SegmentFurniture(
                        capturedImage: $capturedImage,
                        isShowingCamera: $showingSegmentFurniture,
                        roomImage: roomSnapshot
                    )
                    .zIndex(9000)
                }

                // NEW: SmartyPants overlay
                if showingSmartyPants {
                    SmartyPantsViewSwiftUI(
                        mlModel: mlModel,
                        processInterval: 0.07,
//                        scoreThreshold: 0.25,
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
        }
    }

    // MARK: - Overlays & Controls

    private func isLandscape(geometry: GeometryProxy) -> Bool {
        geometry.size.width > geometry.size.height
    }

    // PORTRAIT controls
    private var portraitControls: some View {
        VStack {
            HStack {
                backButton
                Spacer()
            }.padding()
            
            Spacer()
            
            HStack(alignment: .bottom, spacing: 0) {
                VStack(spacing: 16) {
                    cameraButton
                    segmentExamineButton
                    segmentForegroundButton
                    segmentFurnitureButton
                    smartyPantsButton  // NEW
                }
                .padding(.leading, 16)
                .padding(.bottom, 20)
                
                Spacer()
                
                VirtualJoystick(joystickOffset: $joystickOffset)
                    .onChange(of: joystickOffset) { _, newOffset in
                        cameraMovementManager.updateJoystickInput(newOffset)
                    }
                    .padding(.bottom, 20)
                    .padding(.trailing, 100)
                
                Spacer()
            }
            .padding(.bottom, 20)
        }
    }

    // LANDSCAPE controls
    private var landscapeControls: some View {
        HStack {
            VStack {
                HStack(spacing: 12) {
                    backButton
                }
                Spacer()
            }
            .padding()
            
            Spacer()
            
            VStack(spacing: 16) {
                cameraButton
                segmentExamineButton
                segmentForegroundButton
                segmentFurnitureButton
                smartyPantsButton  // NEW
                Spacer()
                VirtualJoystick(joystickOffset: $joystickOffset)
                    .onChange(of: joystickOffset) { _, newOffset in
                        cameraMovementManager.updateJoystickInput(newOffset)
                    }
                    .padding(.top, 20)
                Spacer()
            }
            .padding()
        }
    }

    private var backButton: some View {
        Button("Back") { dismiss() }
            .foregroundColor(.white)
            .padding(.horizontal, 16).padding(.vertical, 8)
            .background(Color.black.opacity(0.7)).cornerRadius(20)
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

    // NEW: SmartyPants Button (YOLOE - 168 furniture classes!)
    private var smartyPantsButton: some View {
        Button(action: {
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
    }

    private func manageARSessionForOverlays() {
        let shouldRunAR = !anyCameraOverlayActive
        if isARActive != shouldRunAR {
            isARActive = shouldRunAR
        }
    }
    
    private func loadModelOnce() {
        guard mlModel == nil else {
            print("✅ ML Model already loaded")
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

        print("🔍 Looking for ML Model (preferring .mlpackage):", candidateNames.map { "\($0.0).\($0.1)" }.joined(separator: ", "))

        var loaded: MLModel? = nil

        for (name, ext) in candidateNames {
            if let url = Bundle.main.url(forResource: name, withExtension: ext) {
                print("📦 Found model file at: \(url.lastPathComponent)")
                do {
                    let config = MLModelConfiguration()
                    // Use CPU-only to avoid ANE "No memory object bound to port" crashes
                    config.computeUnits = .cpuOnly
                    let model = try MLModel(contentsOf: url, configuration: config)
                    loaded = model
                    print("✅ Loaded MLModel '\(name).\(ext)' with computeUnits=\(config.computeUnits)")

                    // Log input constraint to verify shape & dtype (Float16 vs Float32)
                    let inputs = model.modelDescription.inputDescriptionsByName
                    if let img = inputs["image"] {
                        if let mac = img.multiArrayConstraint {
                            let shp = mac.shape.map { $0.intValue }
                            print("📥 Model 'image' MultiArray constraint: shape=\(shp) dataType=\(mac.dataType)")
                            if mac.dataType == .float32 {
                                print("⚠️ Model expects Float32 input — higher memory usage. Consider exporting an FP16 model to reduce memory.")
                            }
                        } else if let ic = img.imageConstraint {
                            print("📥 Model 'image' Image constraint: \(ic.pixelsWide)x\(ic.pixelsHigh) pixelFormat=\(ic.pixelFormatType)")
                        } else {
                            print("ℹ️ No detailed constraint for 'image' input")
                        }
                    }

                    break
                } catch {
                    print("❌ Failed to load \(name).\(ext):", error)
                    continue
                }
            }
        }

        if let model = loaded {
            self.mlModel = model
        } else {
            print("❌ No matching model found in bundle (tried candidates above)")
            // List bundle contents to help debug
            if let bundleContents = try? FileManager.default.contentsOfDirectory(atPath: Bundle.main.bundlePath) {
                print("📁 Bundle contents (mlmodel/mlpackage only):")
                bundleContents.filter { $0.contains("mlmodel") || $0.contains("mlpackage") }.forEach { file in
                    print("   - \(file)")
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

