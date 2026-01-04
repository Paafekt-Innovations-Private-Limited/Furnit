import SwiftUI
import MetalKit
import UIKit

// MARK: - SplatView (SwiftUI wrapper for Metal splat rendering)
struct SplatView: UIViewRepresentable {
    @ObservedObject var viewModel: SplatViewModel

    func makeUIView(context: Context) -> MTKView {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }

        let mtkView = MTKView(frame: .zero, device: device)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.12, alpha: 1.0)
        mtkView.preferredFramesPerSecond = 60
        mtkView.enableSetNeedsDisplay = false  // Continuous rendering
        mtkView.isPaused = false

        // Create renderer
        if let renderer = SplatRenderer(device: device) {
            viewModel.renderer = renderer
            mtkView.delegate = renderer
        }

        // Add gesture recognizers
        let panGesture = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePan(_:)))
        mtkView.addGestureRecognizer(panGesture)

        let pinchGesture = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePinch(_:)))
        mtkView.addGestureRecognizer(pinchGesture)

        return mtkView
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        // Update splats if they changed - dispatch to avoid publishing during view update
        if viewModel.needsUpdate {
            DispatchQueue.main.async {
                viewModel.updateRenderer()
            }
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(viewModel: viewModel)
    }

    // MARK: - Coordinator for gesture handling
    class Coordinator: NSObject {
        var viewModel: SplatViewModel

        init(viewModel: SplatViewModel) {
            self.viewModel = viewModel
        }

        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard let renderer = viewModel.renderer else { return }

            let translation = gesture.translation(in: gesture.view)
            renderer.orbit(deltaX: Float(translation.x), deltaY: Float(-translation.y))
            gesture.setTranslation(.zero, in: gesture.view)
        }

        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            guard let renderer = viewModel.renderer else { return }

            if gesture.state == .changed {
                let delta = Float(gesture.scale - 1.0) * 5.0
                renderer.zoom(delta: delta)
                gesture.scale = 1.0
            }
        }
    }
}

// MARK: - SplatViewModel
class SplatViewModel: ObservableObject {
    @Published var isLoading = false
    @Published var splatCount = 0
    @Published var statusMessage = "Ready"

    var renderer: SplatRenderer?
    var pendingSplats: [SinglePhotoRoomReconstructor.GaussianSplat] = []
    var needsUpdate = false
    private var isProcessing = false  // Guard against duplicate dispatch

    // Room textures for Metal renderer
    var floorTexture: UIImage?
    var ceilingTexture: UIImage?
    var frontWallTexture: UIImage?
    var leftWallTexture: UIImage?
    var rightWallTexture: UIImage?
    var texturesNeedUpdate = false

    func loadSplats(_ splats: [SinglePhotoRoomReconstructor.GaussianSplat]) {
        pendingSplats = splats
        needsUpdate = true
        splatCount = splats.count
        statusMessage = "Loading \(splats.count) splats..."
    }

    func loadTextures(floor: UIImage?, ceiling: UIImage?, front: UIImage?, left: UIImage?, right: UIImage?) {
        floorTexture = floor
        ceilingTexture = ceiling
        frontWallTexture = front
        leftWallTexture = left
        rightWallTexture = right
        texturesNeedUpdate = true
    }

    func updateRenderer() {
        guard let renderer = renderer,
              !pendingSplats.isEmpty,
              !isProcessing else { return }

        isProcessing = true
        needsUpdate = false  // Clear immediately to prevent re-entry
        isLoading = true
        let splats = pendingSplats

        // Capture textures for background thread
        let floor = floorTexture
        let ceiling = ceilingTexture
        let front = frontWallTexture
        let left = leftWallTexture
        let right = rightWallTexture

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            // Convert to Splat array on background thread
            let metalSplats = [Splat].fromSHARP(splats, targetSize: 4.0)
                .filtered(minOpacity: 0.05)

            // Estimate USDZ file size based on textures
            let estimatedSize = self?.estimateUSDZSize(floor: floor, ceiling: ceiling, front: front, left: left, right: right) ?? 0

            DispatchQueue.main.async {
                renderer.loadSplats(metalSplats)

                // Update room textures if available
                if floor != nil || ceiling != nil || front != nil || left != nil || right != nil {
                    renderer.updateRoomTextures(floor: floor, ceiling: ceiling, front: front, left: left, right: right)
                }

                self?.isProcessing = false
                self?.isLoading = false
                self?.texturesNeedUpdate = false
                self?.statusMessage = "USDZ ~\(self?.formatFileSize(estimatedSize) ?? "?")"
            }
        }
    }

    /// Estimate USDZ file size based on textures
    private func estimateUSDZSize(floor: UIImage?, ceiling: UIImage?, front: UIImage?, left: UIImage?, right: UIImage?) -> Int {
        var totalBytes = 0

        // Base geometry overhead (~50KB)
        totalBytes += 50_000

        // Estimate each texture's compressed size (JPEG in USDZ is ~10-15% of raw)
        let textures = [floor, ceiling, front, left, right]
        for texture in textures {
            if let img = texture {
                // Raw size = width * height * 4 bytes (RGBA)
                let rawSize = Int(img.size.width * img.size.height * 4)
                // JPEG compression ratio ~10-12%
                totalBytes += Int(Double(rawSize) * 0.12)
            }
        }

        return totalBytes
    }

    /// Format bytes to human readable string
    private func formatFileSize(_ bytes: Int) -> String {
        if bytes < 1024 {
            return "\(bytes) B"
        } else if bytes < 1024 * 1024 {
            return String(format: "%.1f KB", Double(bytes) / 1024)
        } else {
            return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
        }
    }
}

// MARK: - Preview Container
struct SplatViewContainer: View {
    @StateObject private var viewModel = SplatViewModel()
    @ObservedObject var reconstructor: SinglePhotoRoomReconstructor

    var body: some View {
        ZStack {
            SplatView(viewModel: viewModel)
                .ignoresSafeArea()

            // Overlay UI
            VStack {
                Spacer()

                HStack {
                    Text(viewModel.statusMessage)
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding(8)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(8)

                    Spacer()

                    if viewModel.isLoading {
                        ProgressView()
                            .tint(.white)
                    }
                }
                .padding()
            }
        }
        .onAppear {
            // Load splats when view appears
            // This would be called after SHARP inference
        }
    }
}

// MARK: - Standalone Splat Viewer (for testing)
struct StandaloneSplatViewer: View {
    @StateObject private var viewModel = SplatViewModel()
    let splats: [SinglePhotoRoomReconstructor.GaussianSplat]

    // Optional room textures for textured room planes
    var floorTexture: UIImage?
    var ceilingTexture: UIImage?
    var frontWallTexture: UIImage?
    var leftWallTexture: UIImage?
    var rightWallTexture: UIImage?

    // Optional reconstructor for save functionality
    var reconstructor: SinglePhotoRoomReconstructor?

    // Joystick states
    @State private var moveJoystick: CGSize = .zero
    @State private var lookJoystick: CGSize = .zero

    // Save state
    @State private var showingSaveAlert = false
    @State private var saveMessage = ""
    @State private var showingShareSheet = false
    @State private var savedFileURL: URL?

    var body: some View {
        ZStack {
            SplatView(viewModel: viewModel)
                .ignoresSafeArea()

            // Top HUD
            VStack {
                HStack {
                    // Room dimensions
                    if let dims = reconstructor?.estimatedDimensions {
                        Text(String(format: "%.1f x %.1f x %.1fm", dims.width, dims.depth, dims.height))
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding(8)
                            .background(Color.black.opacity(0.5))
                            .cornerRadius(8)
                    }

                    Spacer()

                    // Save button
                    if reconstructor != nil {
                        Button(action: {
                            Task {
                                if let url = await reconstructor?.saveRoom() {
                                    savedFileURL = url
                                    saveMessage = "Saved: \(url.lastPathComponent)"
                                    showingShareSheet = true
                                } else {
                                    saveMessage = "Failed to save room"
                                    showingSaveAlert = true
                                }
                            }
                        }) {
                            Image(systemName: reconstructor?.isSaving == true ? "arrow.2.circlepath" : "square.and.arrow.down")
                                .font(.title3)
                                .foregroundColor(.white)
                                .padding(8)
                                .background(Color.blue.opacity(0.7))
                                .clipShape(Circle())
                        }
                        .disabled(reconstructor?.isSaving == true)
                    }

                    // Reset button
                    Button(action: {
                        viewModel.renderer?.resetCamera()
                    }) {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.title3)
                            .foregroundColor(.white)
                            .padding(8)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }

                    Text(viewModel.statusMessage)
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                        .padding(8)
                        .background(Color.black.opacity(0.5))
                        .cornerRadius(8)
                }
                .padding()

                Spacer()

                // Bottom: Joysticks + zoom
                HStack(alignment: .bottom) {
                    // Left joystick: Move/Pan
                    VStack(spacing: 4) {
                        Text("MOVE")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.6))
                        VirtualJoystick(joystickOffset: $moveJoystick)
                    }
                    .padding(.leading, 20)

                    Spacer()

                    // Center: Zoom buttons
                    VStack(spacing: 8) {
                        Button(action: {
                            viewModel.renderer?.zoom(delta: 2.0)
                        }) {
                            Image(systemName: "plus.magnifyingglass")
                                .font(.title2)
                                .foregroundColor(.white)
                                .padding(10)
                                .background(Color.black.opacity(0.5))
                                .clipShape(Circle())
                        }
                        Button(action: {
                            viewModel.renderer?.zoom(delta: -2.0)
                        }) {
                            Image(systemName: "minus.magnifyingglass")
                                .font(.title2)
                                .foregroundColor(.white)
                                .padding(10)
                                .background(Color.black.opacity(0.5))
                                .clipShape(Circle())
                        }
                    }

                    Spacer()

                    // Right joystick: Look/Orbit
                    VStack(spacing: 4) {
                        Text("LOOK")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.6))
                        VirtualJoystick(joystickOffset: $lookJoystick)
                    }
                    .padding(.trailing, 20)
                }
                .padding(.bottom, 30)
            }
        }
        .background(Color.black)
        .onAppear {
            loadAll()
        }
        .onChange(of: moveJoystick) { _, newValue in
            // Normalize to -1...1 range (maxDistance is 30)
            let x = Float(newValue.width / 30.0)
            let z = Float(-newValue.height / 30.0)  // Invert Y for natural up=forward
            viewModel.renderer?.pan(deltaX: x, deltaZ: z)
        }
        .onChange(of: lookJoystick) { _, newValue in
            // Normalize and scale for orbit
            let x = Float(newValue.width / 30.0) * 3.0
            let y = Float(newValue.height / 30.0) * 3.0
            viewModel.renderer?.orbit(deltaX: x, deltaY: y)
        }
        .alert("Save Room", isPresented: $showingSaveAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(saveMessage)
        }
        .sheet(isPresented: $showingShareSheet) {
            if let url = savedFileURL {
                ShareSheet(items: [url])
            }
        }
    }

    private func loadAll() {
        viewModel.loadTextures(
            floor: floorTexture,
            ceiling: ceilingTexture,
            front: frontWallTexture,
            left: leftWallTexture,
            right: rightWallTexture
        )
        viewModel.loadSplats(splats)
    }
}

// MARK: - ShareSheet for sharing saved files
struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
