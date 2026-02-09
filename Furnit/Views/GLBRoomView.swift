import SwiftUI
import WebKit
import UIKit
import CoreML

// MARK: - Navigation Back Swipe Disabler
/// Wraps content and disables the navigation back swipe gesture
struct NavigationBackSwipeDisabler<Content: View>: UIViewControllerRepresentable {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    func makeUIViewController(context: Context) -> UIHostingController<Content> {
        let hostingController = BackSwipeDisabledHostingController(rootView: content)
        return hostingController
    }

    func updateUIViewController(_ uiViewController: UIHostingController<Content>, context: Context) {
        uiViewController.rootView = content
    }
}

class BackSwipeDisabledHostingController<Content: View>: UIHostingController<Content> {
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        navigationController?.interactivePopGestureRecognizer?.isEnabled = false
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        navigationController?.interactivePopGestureRecognizer?.isEnabled = false
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        // Re-enable for other views
        navigationController?.interactivePopGestureRecognizer?.isEnabled = true
    }
}

// MARK: - View Modifier to Disable Back Swipe
struct DisableBackSwipeModifier: ViewModifier {
    func body(content: Content) -> some View {
        content.background(BackSwipeDisablerView())
    }
}

struct BackSwipeDisablerView: UIViewRepresentable {
    func makeUIView(context: Context) -> UIView {
        let view = BackSwipeBlockerUIView()
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}
}

class BackSwipeBlockerUIView: UIView {
    private var observer: NSObjectProtocol?

    override func didMoveToWindow() {
        super.didMoveToWindow()
        disableBackSwipe()

        // Also observe when the scene becomes active in case navigation controller wasn't ready
        observer = NotificationCenter.default.addObserver(
            forName: UIScene.didActivateNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.disableBackSwipe()
        }
    }

    override func willMove(toWindow newWindow: UIWindow?) {
        super.willMove(toWindow: newWindow)
        if newWindow == nil {
            // Re-enable when leaving
            enableBackSwipe()
            if let observer = observer {
                NotificationCenter.default.removeObserver(observer)
            }
        }
    }

    private func disableBackSwipe() {
        DispatchQueue.main.async { [weak self] in
            self?.findAndDisableBackSwipe()
        }
    }

    private func enableBackSwipe() {
        var responder: UIResponder? = self
        while responder != nil {
            if let nav = responder as? UINavigationController {
                nav.interactivePopGestureRecognizer?.isEnabled = true
                return
            }
            if let vc = responder as? UIViewController, let nav = vc.navigationController {
                nav.interactivePopGestureRecognizer?.isEnabled = true
                return
            }
            responder = responder?.next
        }
    }

    private func findAndDisableBackSwipe() {
        // Method 1: Walk responder chain
        var responder: UIResponder? = self
        while responder != nil {
            if let nav = responder as? UINavigationController {
                nav.interactivePopGestureRecognizer?.isEnabled = false
                return
            }
            if let vc = responder as? UIViewController, let nav = vc.navigationController {
                nav.interactivePopGestureRecognizer?.isEnabled = false
                return
            }
            responder = responder?.next
        }

        // Method 2: Search through all window scenes
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        for scene in scenes {
            for window in scene.windows {
                if let rootVC = window.rootViewController {
                    findAndDisableInViewController(rootVC)
                }
            }
        }
    }

    private func findAndDisableInViewController(_ vc: UIViewController) {
        if let nav = vc as? UINavigationController {
            nav.interactivePopGestureRecognizer?.isEnabled = false
        }
        if let nav = vc.navigationController {
            nav.interactivePopGestureRecognizer?.isEnabled = false
        }
        for child in vc.children {
            findAndDisableInViewController(child)
        }
        if let presented = vc.presentedViewController {
            findAndDisableInViewController(presented)
        }
    }
}

extension View {
    func disableBackSwipe() -> some View {
        self.modifier(DisableBackSwipeModifier())
    }
}

// MARK: - UIView Extension to Find Navigation Controller
extension UIView {
    func findNavigationController() -> UINavigationController? {
        var responder: UIResponder? = self
        while responder != nil {
            if let nav = responder as? UINavigationController {
                return nav
            }
            if let vc = responder as? UIViewController, let nav = vc.navigationController {
                return nav
            }
            responder = responder?.next
        }
        return nil
    }
}

/// WebGL-based GLB/GLTF room viewer - loads and renders GLB 3D models using Three.js
struct GLBRoomView: View {
    let glbURL: URL
    let photoOrientation: PhotoOrientation
    let roomWidth: Float?
    let roomHeight: Float?

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var authManager: AuthenticationManager

    @State private var isLoading = true
    @State private var error: String? = nil
    @State private var showingFurnitureFit = false
    @ObservedObject private var yoloeService = YOLOEModelService.shared

    var body: some View {
        ZStack {
            // WebGL GLB viewer - OrbitControls in Three.js handles touch directly
            GLBWebGLView(
                glbURL: glbURL,
                photoOrientation: photoOrientation,
                onLoaded: {
                    isLoading = false
                },
                onError: { errorMessage in
                    error = errorMessage
                    isLoading = false
                }
            )
            .ignoresSafeArea()
            .allowsHitTesting(!isLoading)

            // Loading overlay
            if isLoading {
                VStack(spacing: 16) {
                    ProgressView()
                        .scaleEffect(1.5)
                        .tint(.white)

                    Text(NSLocalizedString("photoRoom.loading", comment: ""))
                        .font(.subheadline)
                        .foregroundColor(.white)
                }
                .padding(24)
                .background(Color.black.opacity(0.7))
                .cornerRadius(12)
            }

            // Error overlay
            if let error = error {
                VStack(spacing: 16) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle)
                        .foregroundColor(.orange)
                    Text(error)
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                }
                .padding(24)
                .background(Color.black.opacity(0.7))
                .cornerRadius(12)
            }

            // FurnitureFit overlay (when active) - full screen camera for furniture detection
            if showingFurnitureFit {
                FurnitureFitUIView(
                    capturedImage: .constant(nil),
                    roomImage: nil,
                    mlModel: yoloeService.model,
                    processInterval: 0.07,
                    active: true,
                    lockedOrientation: photoOrientation,
                    roomWidthMeters: roomWidth ?? 4.0,
                    roomHeightMeters: roomHeight ?? 3.0,
                    onFurnitureSizeEstimated: { _, _ in }
                )
                .ignoresSafeArea()
                .zIndex(100)
            }

            // Controls based on orientation
            if photoOrientation == .landscape {
                landscapeControls
            } else {
                portraitControls
            }
        }
        .background(Color.gray)
        .navigationTitle(formatDimensions())
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button(action: {
                    dismiss()
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                        Text("Back")
                    }
                }
            }
            ToolbarItemGroup(placement: .navigationBarTrailing) {
                // Recenter button
                Button(action: {
                    NotificationCenter.default.post(name: NSNotification.Name("RecenterGLBCamera"), object: nil)
                }) {
                    Image(systemName: "viewfinder")
                }
                .disabled(isLoading)
            }
        }
        .onAppear {
            yoloeService.ensureModelLoaded()
            // Lock orientation based on photo orientation
            if photoOrientation == .landscape {
                OrientationLockManager.shared.lockToLandscape()
            } else {
                OrientationLockManager.shared.lockToPortrait()
            }
        }
        .onDisappear {
            OrientationLockManager.shared.unlock()
        }
        .defersSystemGestures(on: .all)
        .disableBackSwipe()
    }

    private func formatDimensions() -> String {
        if let w = roomWidth, let h = roomHeight {
            return String(format: "%.1f × %.1f m", w, h)
        }
        return "3D Room"
    }

    // MARK: - YOLOE model loaded via YOLOEModelService (ODR)

    // MARK: - Portrait Controls
    private var portraitControls: some View {
        VStack {
            Spacer()

            // Orientation label
            HStack(spacing: 6) {
                Image(systemName: "iphone")
                    .font(.caption)
                Text(NSLocalizedString("orientation.heldVertically", comment: ""))
                    .font(.caption2)
                Text("-")
                    .font(.caption2)
                Text(NSLocalizedString("orientation.portrait", comment: ""))
                    .font(.caption2)
                    .fontWeight(.medium)
            }
            .foregroundColor(.white.opacity(0.9))
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.black.opacity(0.5))
            .cornerRadius(8)
            .padding(.bottom, 12)

            // Bottom controls: Brain icon (left) and Camera (right)
            HStack {
                // Brain button (bottom-left)
                Button(action: {
                    showingFurnitureFit.toggle()
                }) {
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 28))
                        .foregroundColor(.white)
                        .frame(width: 60, height: 60)
                        .background(Circle().fill(showingFurnitureFit ? Color.green : Color.blue).shadow(radius: 5))
                }
                .disabled(isLoading)
                .padding(.leading, 16)

                Spacer()

                // Screenshot button (bottom-right)
                Button(action: {
                    takeScreenshot()
                }) {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 28))
                        .foregroundColor(.white)
                        .frame(width: 60, height: 60)
                        .background(Circle().fill(Color.blue).shadow(radius: 5))
                }
                .disabled(isLoading)
                .padding(.trailing, 16)
            }
            .padding(.bottom, 20)
        }
    }

    // MARK: - Landscape Controls
    private var landscapeControls: some View {
        VStack {
            Spacer()

            // Horizontal bottom bar
            HStack(spacing: 20) {
                // Brain button (left)
                Button(action: {
                    showingFurnitureFit.toggle()
                }) {
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 28))
                        .foregroundColor(.white)
                        .frame(width: 60, height: 60)
                        .background(Circle().fill(showingFurnitureFit ? Color.green : Color.blue).shadow(radius: 5))
                }
                .disabled(isLoading)

                // Orientation label
                HStack(spacing: 6) {
                    Image(systemName: "iphone.landscape")
                        .font(.caption)
                    Text(NSLocalizedString("orientation.heldHorizontally", comment: ""))
                        .font(.caption2)
                    Text("-")
                        .font(.caption2)
                    Text(NSLocalizedString("orientation.landscape", comment: ""))
                        .font(.caption2)
                        .fontWeight(.medium)
                }
                .foregroundColor(.white.opacity(0.9))
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.5))
                .cornerRadius(8)

                Spacer()

                // Screenshot button (right)
                Button(action: {
                    takeScreenshot()
                }) {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 28))
                        .foregroundColor(.white)
                        .frame(width: 60, height: 60)
                        .background(Circle().fill(Color.blue).shadow(radius: 5))
                }
                .disabled(isLoading)
            }
            .padding(.horizontal, 30)
            .padding(.bottom, 20)
        }
    }

    // MARK: - Take Screenshot
    private func takeScreenshot() {
        logDebug("📸 [GLBRoomView] Taking screenshot...")

        // Capture the window
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let windows = scenes.flatMap { $0.windows }
        guard let window = windows.first(where: { $0.isKeyWindow }) ?? windows.first else {
            logDebug("❌ [GLBRoomView] No window found")
            return
        }

        let format = UIGraphicsImageRendererFormat()
        format.scale = UIScreen.main.scale
        let renderer = UIGraphicsImageRenderer(bounds: window.bounds, format: format)
        let image = renderer.image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }

        logDebug("📸 [GLBRoomView] Screenshot captured, saving to Photos...")

        // Save to photos
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        logDebug("✅ [GLBRoomView] Screenshot saved to Photos")
    }
}

// MARK: - WebGL GLB View (UIViewRepresentable)
struct GLBWebGLView: UIViewRepresentable {
    let glbURL: URL
    let photoOrientation: PhotoOrientation
    let onLoaded: () -> Void
    let onError: (String) -> Void

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true

        // Add message handler for JS -> Swift communication
        config.userContentController.add(context.coordinator, name: "glbViewer")

        let webView = WKWebView(frame: .zero, configuration: config)
        context.coordinator.webView = webView

        // Completely disable WebView scrolling - let Three.js handle all touch
        webView.scrollView.isScrollEnabled = false
        webView.scrollView.bounces = false
        webView.scrollView.bouncesZoom = false
        webView.scrollView.minimumZoomScale = 1.0
        webView.scrollView.maximumZoomScale = 1.0
        webView.scrollView.contentInsetAdjustmentBehavior = .never

        // Disable all scroll view gesture recognizers
        for gestureRecognizer in webView.scrollView.gestureRecognizers ?? [] {
            gestureRecognizer.isEnabled = false
        }

        webView.isOpaque = false
        webView.backgroundColor = .gray
        webView.isUserInteractionEnabled = true
        webView.isMultipleTouchEnabled = true

        // Add edge pan gesture to block system back swipe
        let edgePan = UIScreenEdgePanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleEdgePan(_:)))
        edgePan.edges = .left
        edgePan.cancelsTouchesInView = false
        edgePan.delaysTouchesBegan = false
        edgePan.delegate = context.coordinator
        webView.addGestureRecognizer(edgePan)

        // Find and disable the navigation controller's back gesture
        DispatchQueue.main.async {
            if let navController = webView.findNavigationController() {
                navController.interactivePopGestureRecognizer?.isEnabled = false
            }
        }

        // Load GLB data and generate HTML
        if let glbData = try? Data(contentsOf: glbURL) {
            let html = generateGLBViewerHTML(glbData: glbData)
            webView.loadHTMLString(html, baseURL: nil)
        } else {
            logDebug("❌ [GLBRoomView] Failed to load GLB file: \(glbURL.path)")
            DispatchQueue.main.async {
                onError("Failed to load 3D model file")
            }
        }

        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onLoaded: onLoaded, onError: onError)
    }

    class Coordinator: NSObject, WKScriptMessageHandler, UIGestureRecognizerDelegate {
        let onLoaded: () -> Void
        let onError: (String) -> Void
        weak var webView: WKWebView?

        init(onLoaded: @escaping () -> Void, onError: @escaping (String) -> Void) {
            self.onLoaded = onLoaded
            self.onError = onError
            super.init()

            // Listen for recenter notification
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(recenterCamera),
                name: NSNotification.Name("RecenterGLBCamera"),
                object: nil
            )
        }

        deinit {
            NotificationCenter.default.removeObserver(self)
        }

        @objc private func recenterCamera() {
            webView?.evaluateJavaScript("if (typeof recenterCamera === 'function') recenterCamera();", completionHandler: nil)
        }

        @objc func handleEdgePan(_ gesture: UIScreenEdgePanGestureRecognizer) {
            // Do nothing - this gesture just blocks the system back swipe
        }

        // Always recognize our edge gesture, blocking others
        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldBeRequiredToFailBy otherGestureRecognizer: UIGestureRecognizer) -> Bool {
            return true
        }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            guard let body = message.body as? [String: Any],
                  let event = body["event"] as? String else { return }

            switch event {
            case "loaded":
                DispatchQueue.main.async {
                    self.onLoaded()
                }
            case "error":
                let errorMessage = body["message"] as? String ?? "Unknown error"
                DispatchQueue.main.async {
                    self.onError(errorMessage)
                }
            default:
                break
            }
        }
    }

    // MARK: - Generate Three.js HTML for GLB Loading
    private func generateGLBViewerHTML(glbData: Data) -> String {
        let base64GLB = glbData.base64EncodedString()
        let isPortrait = photoOrientation == .portrait

        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <style>
                * { margin: 0; padding: 0; }
                html, body {
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    background: #808080;
                    touch-action: none;
                }
                canvas {
                    display: block;
                    width: 100%;
                    height: 100%;
                    touch-action: none;
                }
            </style>
        </head>
        <body>
            <script type="importmap">
            {
                "imports": {
                    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
                }
            }
            </script>
            <script type="module">
                import * as THREE from 'three';
                import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
                import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

                const isPortrait = \(isPortrait ? "true" : "false");

                console.log('[GLBViewer] Starting...');

                // Scene setup
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x808080);

                // Camera
                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
                camera.position.set(0, 2, 5);

                // Renderer
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                document.body.appendChild(renderer.domElement);

                // Orbit controls
                const controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;  // Quick response
                controls.rotateSpeed = 3.0;     // Fast rotation for touch
                controls.zoomSpeed = 2.5;       // Fast zoom
                controls.enableZoom = true;
                controls.enablePan = false;
                controls.minDistance = 0.5;
                controls.maxDistance = 20;

                // Listen for orbit commands from Swift
                window.orbitCamera = function(deltaX, deltaY) {
                    const spherical = new THREE.Spherical();
                    const offset = new THREE.Vector3();
                    offset.copy(camera.position).sub(controls.target);
                    spherical.setFromVector3(offset);

                    spherical.theta -= deltaX * 0.012;  // Increased for faster response
                    spherical.phi -= deltaY * 0.012;
                    spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

                    offset.setFromSpherical(spherical);
                    camera.position.copy(controls.target).add(offset);
                    camera.lookAt(controls.target);
                };

                // Lighting
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);

                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(5, 10, 5);
                scene.add(directionalLight);

                // Load GLB from base64
                const base64GLB = '\(base64GLB)';

                try {
                    // Decode base64 to ArrayBuffer
                    const binaryString = atob(base64GLB);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }
                    const arrayBuffer = bytes.buffer;

                    console.log('[GLBViewer] GLB data size:', arrayBuffer.byteLength);

                    // Load with GLTFLoader
                    const loader = new GLTFLoader();
                    loader.parse(arrayBuffer, '', function(gltf) {
                        console.log('[GLBViewer] GLB loaded successfully');

                        const model = gltf.scene;
                        scene.add(model);

                        // Get model bounds
                        const box = new THREE.Box3().setFromObject(model);
                        const center = box.getCenter(new THREE.Vector3());
                        const size = box.getSize(new THREE.Vector3());

                        console.log('[GLBViewer] Model bounds - center:', center, 'size:', size);

                        // Center the model horizontally but keep floor at y=0
                        model.position.x = -center.x;
                        model.position.z = -center.z;
                        // Adjust Y so floor is at 0
                        model.position.y = -box.min.y;

                        // Position camera inside the room looking at front wall
                        // After centering: room spans -size/2 to +size/2 in X and Z
                        // Front wall is at Z = -roomDepth/2, back (open) is at Z = +roomDepth/2
                        const roomWidth = size.x;
                        const roomHeight = size.y;
                        const roomDepth = size.z;

                        // Camera at back part of room, eye level, looking at front wall
                        const eyeHeight = roomHeight * 0.5;
                        // Position camera 30% into the room from back (+Z side)
                        const cameraZ = roomDepth * 0.2;
                        // Look at center-front of room (towards -Z)
                        const targetZ = -roomDepth * 0.2;

                        camera.position.set(0, eyeHeight, cameraZ);
                        controls.target.set(0, eyeHeight, targetZ);
                        controls.update();

                        // Save initial camera state for recenter
                        const initialCameraPos = camera.position.clone();
                        const initialTarget = controls.target.clone();

                        // Recenter function
                        window.recenterCamera = function() {
                            camera.position.copy(initialCameraPos);
                            controls.target.copy(initialTarget);
                            controls.update();
                            console.log('[GLBViewer] Camera recentered');
                        };

                        console.log('[GLBViewer] Room size:', roomWidth.toFixed(2), 'x', roomHeight.toFixed(2), 'x', roomDepth.toFixed(2));
                        console.log('[GLBViewer] Camera at (0,', eyeHeight.toFixed(2), ',', cameraZ.toFixed(2), '), looking at (0,', eyeHeight.toFixed(2), ',', targetZ.toFixed(2), ')');

                        // Notify Swift that we're loaded
                        window.webkit.messageHandlers.glbViewer.postMessage({ event: 'loaded' });

                    }, function(error) {
                        console.error('[GLBViewer] GLB parse error:', error);
                        window.webkit.messageHandlers.glbViewer.postMessage({
                            event: 'error',
                            message: 'Failed to parse 3D model'
                        });
                    });

                } catch (error) {
                    console.error('[GLBViewer] Error:', error);
                    window.webkit.messageHandlers.glbViewer.postMessage({
                        event: 'error',
                        message: error.message || 'Failed to load 3D model'
                    });
                }

                // Animation loop
                function animate() {
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }
                animate();

                // Handle resize
                window.addEventListener('resize', () => {
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                });

            </script>
        </body>
        </html>
        """
    }
}
