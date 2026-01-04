import SwiftUI
import MetalKit

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

    func loadSplats(_ splats: [SinglePhotoRoomReconstructor.GaussianSplat]) {
        pendingSplats = splats
        needsUpdate = true
        splatCount = splats.count
        statusMessage = "Loading \(splats.count) splats..."
    }

    func updateRenderer() {
        guard let renderer = renderer,
              !pendingSplats.isEmpty,
              !isProcessing else { return }

        isProcessing = true
        needsUpdate = false  // Clear immediately to prevent re-entry
        isLoading = true
        let splats = pendingSplats

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            // Convert to Splat array on background thread
            let metalSplats = [Splat].fromSHARP(splats, targetSize: 4.0)
                .filtered(minOpacity: 0.05)

            DispatchQueue.main.async {
                renderer.loadSplats(metalSplats)
                self?.isProcessing = false
                self?.isLoading = false
                self?.statusMessage = "Loaded \(metalSplats.count) splats"
            }
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

    var body: some View {
        ZStack {
            SplatView(viewModel: viewModel)
                .ignoresSafeArea()

            VStack {
                HStack {
                    Text("Splats: \(viewModel.splatCount)")
                        .font(.headline)
                        .foregroundColor(.white)

                    Spacer()

                    Button("Reload") {
                        viewModel.loadSplats(splats)
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding()

                Spacer()

                Text(viewModel.statusMessage)
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.7))
                    .padding()
            }
        }
        .background(Color.black)
        .onAppear {
            viewModel.loadSplats(splats)
        }
    }
}
