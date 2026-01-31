import SwiftUI
import WebKit
import UIKit
import CoreML

/// WebGL-based mesh room viewer - renders box room geometry using Three.js
/// Exports to GLTF/GLB format for universal 3D viewing
struct MeshRoomView: View {
    let roomWidth: Float
    let roomHeight: Float
    let roomDepth: Float
    let frontWallImage: UIImage
    let photoOrientation: PhotoOrientation

    // Boundary coordinates (normalized 0-1) for texturing walls
    var leftX: CGFloat = 0.12
    var rightX: CGFloat = 0.88
    var ceilingY: CGFloat = 0.15
    var floorY: CGFloat = 0.85

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var authManager: AuthenticationManager

    @State private var isLoading = true
    @State private var error: String? = nil

    // Room name for saving
    @State private var showRoomNameInput = false
    @State private var roomName = ""
    @State private var isSavingRoom = false
    @State private var showSaveAlert = false
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false

    // Brain mode (furniture detection)
    @State private var showingFurnitureFit = false
    @State private var mlModel: MLModel? = nil

    // WebView reference for GLTF export
    @State private var webView: WKWebView?

    // Model manager for saving rooms
    @StateObject private var modelManager = USDZModelManager()

    var body: some View {
        ZStack {
            // WebGL mesh viewer - OrbitControls in Three.js handles touch directly
            MeshWebGLView(
                roomWidth: roomWidth,
                roomHeight: roomHeight,
                roomDepth: roomDepth,
                frontWallImage: frontWallImage,
                photoOrientation: photoOrientation,
                leftX: leftX,
                rightX: rightX,
                ceilingY: ceilingY,
                floorY: floorY,
                webViewRef: $webView,
                onLoaded: {
                    isLoading = false
                },
                onGLBExported: { glbData in
                    saveGLBRoom(glbData: glbData)
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

            // Saving overlay
            if isSavingRoom {
                VStack(spacing: 16) {
                    ProgressView()
                        .scaleEffect(1.5)
                        .tint(.white)

                    Text("Exporting 3D model...")
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
                    mlModel: mlModel,
                    processInterval: 0.07,
                    active: true,
                    lockedOrientation: photoOrientation,
                    roomWidthMeters: roomWidth,
                    roomHeightMeters: roomHeight,
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
        .navigationTitle(String(format: "%.1f × %.1f m", roomWidth, roomHeight))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .navigationBarTrailing) {
                // Save Room button
                Button(action: {
                    showRoomNameInput = true
                }) {
                    Image(systemName: "square.and.arrow.down")
                }
                .disabled(isLoading || isSavingRoom)

                // Recenter button
                Button(action: {
                    NotificationCenter.default.post(name: NSNotification.Name("RecenterMeshCamera"), object: nil)
                }) {
                    Image(systemName: "viewfinder")
                }
                .disabled(isLoading)
            }
        }
        .onAppear {
            loadMLModel()
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
        .alert("Save Room", isPresented: $showRoomNameInput) {
            TextField("Room name", text: $roomName)
            Button("Cancel", role: .cancel) {
                roomName = ""
            }
            Button("Save") {
                requestGLBExport()
            }
            .disabled(roomName.isEmpty)
        }
        .alert("Room Saved", isPresented: $showSaveAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(saveAlertMessage)
        }
        .defersSystemGestures(on: .all)
    }

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

    // MARK: - Request GLB Export from JavaScript
    private func requestGLBExport() {
        isSavingRoom = true
        webView?.evaluateJavaScript("exportGLB()") { _, error in
            if let error = error {
                logDebug("❌ [MeshRoomView] Error requesting GLB export: \(error)")
                DispatchQueue.main.async {
                    isSavingRoom = false
                    saveAlertMessage = "Failed to export 3D model"
                    showSaveAlert = true
                }
            }
            // GLB data will come back via the message handler
        }
    }

    // MARK: - Save GLB Room
    private func saveGLBRoom(glbData: Data) {
        modelManager.saveGLBRoom(
            glbData: glbData,
            name: roomName,
            photoOrientation: photoOrientation,
            roomWidth: roomWidth,
            roomHeight: roomHeight,
            roomDepth: roomDepth
        ) { success, errorMessage in
            isSavingRoom = false
            saveWasSuccessful = success

            if success {
                saveAlertMessage = "Room '\(roomName)' saved successfully!"
                // Post notification to dismiss the sheet and refresh home view
                NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil)
            } else {
                saveAlertMessage = errorMessage ?? "Failed to save room"
            }
            showSaveAlert = true
            roomName = ""
        }
    }

    // MARK: - Take Screenshot
    private func takeScreenshot() {
        logDebug("📸 [MeshRoomView] Taking screenshot...")

        // Capture the window
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let windows = scenes.flatMap { $0.windows }
        guard let window = windows.first(where: { $0.isKeyWindow }) ?? windows.first else {
            logDebug("❌ [MeshRoomView] No window found")
            return
        }

        let format = UIGraphicsImageRendererFormat()
        format.scale = UIScreen.main.scale
        let renderer = UIGraphicsImageRenderer(bounds: window.bounds, format: format)
        let image = renderer.image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }

        logDebug("📸 [MeshRoomView] Screenshot captured, saving to Photos...")

        // Save to photos
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        logDebug("✅ [MeshRoomView] Screenshot saved to Photos")
    }

    // MARK: - Load ML Model for FurnitureFit
    private func loadMLModel() {
        guard mlModel == nil else { return }

        let candidateNames = [
            ("yoloe-11l-seg-pf", "mlmodelc"),
            ("yoloe-11l-seg-pf", "mlpackage"),
        ]

        for (name, ext) in candidateNames {
            if let url = Bundle.main.url(forResource: name, withExtension: ext) {
                do {
                    let config = MLModelConfiguration()
                    config.computeUnits = .cpuOnly
                    let model = try MLModel(contentsOf: url, configuration: config)
                    self.mlModel = model
                    logDebug("✅ [MeshRoomView] Loaded MLModel '\(name).\(ext)'")
                    break
                } catch {
                    logDebug("❌ [MeshRoomView] Failed to load \(name).\(ext): \(error)")
                }
            }
        }
    }
}

// MARK: - WebGL Mesh View (UIViewRepresentable)
struct MeshWebGLView: UIViewRepresentable {
    let roomWidth: Float
    let roomHeight: Float
    let roomDepth: Float
    let frontWallImage: UIImage
    let photoOrientation: PhotoOrientation

    // Boundary coordinates for wall texturing (normalized 0-1)
    let leftX: CGFloat
    let rightX: CGFloat
    let ceilingY: CGFloat
    let floorY: CGFloat

    @Binding var webViewRef: WKWebView?
    let onLoaded: () -> Void
    let onGLBExported: (Data) -> Void

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true

        // Add message handler for JS -> Swift communication
        config.userContentController.add(context.coordinator, name: "meshViewer")

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

        // Store reference
        DispatchQueue.main.async {
            webViewRef = webView
        }

        // Load the HTML with Three.js
        let html = generateMeshViewerHTML()
        webView.loadHTMLString(html, baseURL: nil)

        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onLoaded: onLoaded, onGLBExported: onGLBExported)
    }

    class Coordinator: NSObject, WKScriptMessageHandler, UIGestureRecognizerDelegate {
        let onLoaded: () -> Void
        let onGLBExported: (Data) -> Void
        weak var webView: WKWebView?

        init(onLoaded: @escaping () -> Void, onGLBExported: @escaping (Data) -> Void) {
            self.onLoaded = onLoaded
            self.onGLBExported = onGLBExported
            super.init()

            // Listen for recenter notification
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(recenterCamera),
                name: NSNotification.Name("RecenterMeshCamera"),
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
            case "glbExported":
                // Receive base64-encoded GLB data
                if let base64Data = body["data"] as? String,
                   let glbData = Data(base64Encoded: base64Data) {
                    logDebug("✅ [MeshViewer] Received GLB data: \(glbData.count) bytes")
                    DispatchQueue.main.async {
                        self.onGLBExported(glbData)
                    }
                } else {
                    logDebug("❌ [MeshViewer] Failed to decode GLB data")
                }
            default:
                break
            }
        }
    }

    // MARK: - Generate Three.js HTML with GLTF Export
    private func generateMeshViewerHTML() -> String {
        // Convert image to base64
        let imageData = frontWallImage.jpegData(compressionQuality: 0.85) ?? Data()
        let base64Image = imageData.base64EncodedString()

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
                import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

                // Room dimensions from Swift (in meters)
                const roomWidth = \(roomWidth);
                const roomHeight = \(roomHeight);
                const roomDepth = \(roomDepth);
                const isPortrait = \(isPortrait ? "true" : "false");

                // Boundary coordinates from Swift (normalized 0-1)
                // These define where the front wall edges are in the source image
                const leftX = \(leftX);      // Left wall boundary (0.12 = 12% from left)
                const rightX = \(rightX);    // Right wall boundary (0.88 = 88% from left)
                const ceilingY = \(ceilingY); // Ceiling boundary (0.15 = 15% from top)
                const floorY = \(floorY);    // Floor boundary (0.85 = 85% from top)

                console.log('[MeshViewer] Room dimensions:', roomWidth, 'x', roomHeight, 'x', roomDepth, 'm');
                console.log('[MeshViewer] Boundaries: L=' + leftX + ', R=' + rightX + ', T=' + ceilingY + ', B=' + floorY);

                // Scene setup
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x404040);

                // Create a group to hold the room for export
                const roomGroup = new THREE.Group();
                roomGroup.name = 'Room';
                scene.add(roomGroup);

                // Camera - start inside the room looking at front wall
                const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 100);

                // Renderer
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                renderer.outputColorSpace = THREE.SRGBColorSpace;
                document.body.appendChild(renderer.domElement);

                // Orbit controls - smooth touch interactions
                const controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;  // Quick response
                controls.rotateSpeed = 3.0;     // Fast rotation for touch
                controls.zoomSpeed = 2.5;       // Fast zoom
                controls.panSpeed = 1.5;        // Fast pan
                controls.enableZoom = true;
                controls.enablePan = true;
                controls.minDistance = 0.3;
                controls.maxDistance = Math.max(roomWidth, roomDepth) * 1.5;
                controls.maxPolarAngle = Math.PI * 0.95;
                controls.minPolarAngle = Math.PI * 0.05;

                // Touch-specific settings
                controls.touches = {
                    ONE: THREE.TOUCH.ROTATE,
                    TWO: THREE.TOUCH.DOLLY_PAN
                };

                // Set initial camera position - back center of room looking at front wall
                const targetY = roomHeight * 0.5;
                controls.target.set(0, targetY, 0);
                camera.position.set(0, targetY, roomDepth * 0.35);
                controls.update();

                // Save initial camera state for recenter
                const initialCameraPos = camera.position.clone();
                const initialTarget = controls.target.clone();

                // Recenter function
                window.recenterCamera = function() {
                    camera.position.copy(initialCameraPos);
                    controls.target.copy(initialTarget);
                    controls.update();
                    console.log('[MeshViewer] Camera recentered');
                };

                // Lighting - brighter for textured walls
                const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
                roomGroup.add(ambientLight);

                const frontLight = new THREE.DirectionalLight(0xffffff, 0.3);
                frontLight.position.set(0, roomHeight, roomDepth * 0.3);
                roomGroup.add(frontLight);

                // Load source image
                const img = new Image();
                img.onload = function() {
                    console.log('[MeshViewer] Image loaded:', img.width, 'x', img.height);
                    buildRoomWithTextures(img);
                };
                img.onerror = function() {
                    console.error('[MeshViewer] Failed to load image');
                    buildRoomGray(); // Build room with gray walls as fallback
                };
                img.src = 'data:image/jpeg;base64,\(base64Image)';

                // Create a texture from a portion of the source image
                function createTextureFromRegion(sourceImg, x, y, w, h) {
                    const canvas = document.createElement('canvas');
                    canvas.width = Math.max(1, Math.round(w));
                    canvas.height = Math.max(1, Math.round(h));
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(sourceImg, x, y, w, h, 0, 0, canvas.width, canvas.height);

                    const texture = new THREE.CanvasTexture(canvas);
                    texture.colorSpace = THREE.SRGBColorSpace;
                    texture.needsUpdate = true;
                    return texture;
                }

                function buildRoomWithTextures(sourceImg) {
                    const imgW = sourceImg.width;
                    const imgH = sourceImg.height;

                    // Calculate pixel coordinates for each region
                    const pxLeftX = leftX * imgW;
                    const pxRightX = rightX * imgW;
                    const pxCeilingY = ceilingY * imgH;
                    const pxFloorY = floorY * imgH;

                    console.log('[MeshViewer] Pixel boundaries:', pxLeftX, pxRightX, pxCeilingY, pxFloorY);

                    // Front wall: center region (leftX to rightX, ceilingY to floorY)
                    const frontTexture = createTextureFromRegion(
                        sourceImg,
                        pxLeftX, pxCeilingY,
                        pxRightX - pxLeftX, pxFloorY - pxCeilingY
                    );

                    // Left wall: left region (0 to leftX, ceilingY to floorY)
                    const leftTexture = createTextureFromRegion(
                        sourceImg,
                        0, pxCeilingY,
                        pxLeftX, pxFloorY - pxCeilingY
                    );

                    // Right wall: right region (rightX to 1, ceilingY to floorY)
                    const rightTexture = createTextureFromRegion(
                        sourceImg,
                        pxRightX, pxCeilingY,
                        imgW - pxRightX, pxFloorY - pxCeilingY
                    );

                    // Ceiling: top region (leftX to rightX, 0 to ceilingY)
                    const ceilingTexture = createTextureFromRegion(
                        sourceImg,
                        pxLeftX, 0,
                        pxRightX - pxLeftX, pxCeilingY
                    );

                    // Floor: bottom region (leftX to rightX, floorY to 1)
                    const floorTexture = createTextureFromRegion(
                        sourceImg,
                        pxLeftX, pxFloorY,
                        pxRightX - pxLeftX, imgH - pxFloorY
                    );

                    // Create materials with textures
                    const frontMaterial = new THREE.MeshBasicMaterial({
                        map: frontTexture,
                        side: THREE.DoubleSide
                    });

                    const leftMaterial = new THREE.MeshBasicMaterial({
                        map: leftTexture,
                        side: THREE.DoubleSide
                    });

                    const rightMaterial = new THREE.MeshBasicMaterial({
                        map: rightTexture,
                        side: THREE.DoubleSide
                    });

                    const ceilingMaterial = new THREE.MeshBasicMaterial({
                        map: ceilingTexture,
                        side: THREE.DoubleSide
                    });

                    const floorMaterial = new THREE.MeshBasicMaterial({
                        map: floorTexture,
                        side: THREE.DoubleSide
                    });

                    // Front wall - at z = -roomDepth/2
                    const frontGeometry = new THREE.PlaneGeometry(roomWidth, roomHeight);
                    const frontWall = new THREE.Mesh(frontGeometry, frontMaterial);
                    frontWall.position.set(0, roomHeight / 2, -roomDepth / 2);
                    frontWall.name = 'FrontWall';
                    roomGroup.add(frontWall);

                    // Floor - at y = 0
                    const floorGeometry = new THREE.PlaneGeometry(roomWidth, roomDepth);
                    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
                    floor.rotation.x = -Math.PI / 2;
                    floor.position.set(0, 0, 0);
                    floor.name = 'Floor';
                    roomGroup.add(floor);

                    // Ceiling - at y = roomHeight
                    const ceilingGeometry = new THREE.PlaneGeometry(roomWidth, roomDepth);
                    const ceiling = new THREE.Mesh(ceilingGeometry, ceilingMaterial);
                    ceiling.rotation.x = Math.PI / 2;
                    ceiling.position.set(0, roomHeight, 0);
                    ceiling.name = 'Ceiling';
                    roomGroup.add(ceiling);

                    // Left wall - at x = -roomWidth/2
                    const leftGeometry = new THREE.PlaneGeometry(roomDepth, roomHeight);
                    const leftWall = new THREE.Mesh(leftGeometry, leftMaterial);
                    leftWall.rotation.y = Math.PI / 2;
                    leftWall.position.set(-roomWidth / 2, roomHeight / 2, 0);
                    leftWall.name = 'LeftWall';
                    roomGroup.add(leftWall);

                    // Right wall - at x = roomWidth/2
                    const rightGeometry = new THREE.PlaneGeometry(roomDepth, roomHeight);
                    const rightWall = new THREE.Mesh(rightGeometry, rightMaterial);
                    rightWall.rotation.y = -Math.PI / 2;
                    rightWall.position.set(roomWidth / 2, roomHeight / 2, 0);
                    rightWall.name = 'RightWall';
                    roomGroup.add(rightWall);

                    // No back wall - open for camera to enter

                    console.log('[MeshViewer] Room built with textures');

                    // Notify Swift that we're loaded
                    window.webkit.messageHandlers.meshViewer.postMessage({ event: 'loaded' });
                }

                // Fallback: build room with gray walls
                function buildRoomGray() {
                    const wallMaterial = new THREE.MeshStandardMaterial({
                        color: 0x666666,
                        side: THREE.DoubleSide,
                        roughness: 0.8
                    });

                    const floorMaterial = new THREE.MeshStandardMaterial({
                        color: 0x444444,
                        side: THREE.DoubleSide,
                        roughness: 0.9
                    });

                    const ceilingMaterial = new THREE.MeshStandardMaterial({
                        color: 0x999999,
                        side: THREE.DoubleSide,
                        roughness: 0.7
                    });

                    // Front wall
                    const frontGeometry = new THREE.PlaneGeometry(roomWidth, roomHeight);
                    const frontWall = new THREE.Mesh(frontGeometry, wallMaterial);
                    frontWall.position.set(0, roomHeight / 2, -roomDepth / 2);
                    frontWall.name = 'FrontWall';
                    roomGroup.add(frontWall);

                    // Floor
                    const floorGeometry = new THREE.PlaneGeometry(roomWidth, roomDepth);
                    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
                    floor.rotation.x = -Math.PI / 2;
                    floor.position.set(0, 0, 0);
                    floor.name = 'Floor';
                    roomGroup.add(floor);

                    // Ceiling
                    const ceilingGeometry = new THREE.PlaneGeometry(roomWidth, roomDepth);
                    const ceiling = new THREE.Mesh(ceilingGeometry, ceilingMaterial);
                    ceiling.rotation.x = Math.PI / 2;
                    ceiling.position.set(0, roomHeight, 0);
                    ceiling.name = 'Ceiling';
                    roomGroup.add(ceiling);

                    // Left wall
                    const leftGeometry = new THREE.PlaneGeometry(roomDepth, roomHeight);
                    const leftWall = new THREE.Mesh(leftGeometry, wallMaterial);
                    leftWall.rotation.y = Math.PI / 2;
                    leftWall.position.set(-roomWidth / 2, roomHeight / 2, 0);
                    leftWall.name = 'LeftWall';
                    roomGroup.add(leftWall);

                    // Right wall
                    const rightGeometry = new THREE.PlaneGeometry(roomDepth, roomHeight);
                    const rightWall = new THREE.Mesh(rightGeometry, wallMaterial);
                    rightWall.rotation.y = -Math.PI / 2;
                    rightWall.position.set(roomWidth / 2, roomHeight / 2, 0);
                    rightWall.name = 'RightWall';
                    roomGroup.add(rightWall);

                    console.log('[MeshViewer] Room built (gray fallback)');
                    window.webkit.messageHandlers.meshViewer.postMessage({ event: 'loaded' });
                }

                // GLTF Export function - called from Swift
                window.exportGLB = function() {
                    console.log('[MeshViewer] Starting GLB export...');

                    const exporter = new GLTFExporter();

                    exporter.parse(
                        roomGroup,
                        function(glb) {
                            console.log('[MeshViewer] GLB export complete, size:', glb.byteLength);

                            // Convert ArrayBuffer to base64
                            const bytes = new Uint8Array(glb);
                            let binary = '';
                            const chunkSize = 8192;
                            for (let i = 0; i < bytes.byteLength; i += chunkSize) {
                                const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.byteLength));
                                binary += String.fromCharCode.apply(null, chunk);
                            }
                            const base64 = btoa(binary);

                            // Send to Swift
                            window.webkit.messageHandlers.meshViewer.postMessage({
                                event: 'glbExported',
                                data: base64
                            });
                        },
                        function(error) {
                            console.error('[MeshViewer] GLB export error:', error);
                            window.webkit.messageHandlers.meshViewer.postMessage({
                                event: 'exportError',
                                message: error.message || 'Export failed'
                            });
                        },
                        { binary: true }
                    );
                };

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

                console.log('[MeshViewer] Initialized');

            </script>
        </body>
        </html>
        """
    }
}
