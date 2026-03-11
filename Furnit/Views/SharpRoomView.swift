import SwiftUI
import CoreML
import Photos
import WebKit

// MARK: - Room Boundary Manager

/// Boundary manager for WebGL Gaussian splat rendering
/// Uses actual room bounds to calculate optimal camera positions
struct RoomBoundaryManager {
    let bounds: RoomBounds

    /// Room dimensions
    var width: Float { bounds.width }
    var height: Float { bounds.height }
    var depth: Float { bounds.depth }

    /// Room center
    var centerX: Float { bounds.centerX }
    var centerY: Float { bounds.centerY }
    var centerZ: Float { bounds.centerZ }

    /// Wall positions (maxZ = front wall in classic PLY, minZ = back wall)
    /// In classic PLY from SHARP, Z is negative, and the wall closest to camera
    /// is the one with the *largest* Z (least negative).
    var frontWallZ: Float { bounds.maxZ }  // closest to camera
    var backWallZ: Float { bounds.minZ }   // farthest from camera

    init(bounds: RoomBounds) {
        self.bounds = bounds
    }

    /// Default bounds when none provided
    static var defaultBounds: RoomBounds {
        RoomBounds(minX: -2, maxX: 2, minY: -1.5, maxY: 1.5, minZ: -5, maxZ: -1)
    }

    /// Match Android RoomBoundaryManager: depth-adaptive inset (18% for tiny rooms, up to 50% for deep).
    private static func backCenterInsetFraction(depth: Float) -> Float {
        let t = min(1.0, max(0.0, depth / 6.0))
        return 0.18 + 0.32 * t
    }

    /// Camera at back center with depth-adaptive inset (matches Android when room opened from list / room created).
    private let cameraPadding: Float = 0.3

    /// Calculate camera position using Android formula: back center, depth-adaptive inset, look at front wall.
    func getCameraAtBackCenter() -> (eye: SIMD3<Float>, target: SIMD3<Float>) {
        let fraction = Self.backCenterInsetFraction(depth: depth)
        let insetFromBack = max(depth * fraction, cameraPadding)
        // PLY: back wall = minZ, so camera Z = backWallZ + insetFromBack (inside room from back)
        let camZ = backWallZ + insetFromBack
        let camY = centerY + 0.4
        let eye = SIMD3<Float>(centerX, camY, camZ)
        let target = SIMD3<Float>(centerX, centerY, frontWallZ)
        logDebug("📷 [BoundaryManager] getCameraAtBackCenter depth=\(depth) fraction=\(fraction) inset=\(insetFromBack) eye=(\(eye.x),\(eye.y),\(eye.z)) target=(\(target.x),\(target.y),\(target.z))")
        return (eye: eye, target: target)
    }

    /// Calculate camera position just INSIDE the room (matches Android formula for list / created room).
    func getCameraAtBackWall(fovDegrees: Float = 60) -> (eye: SIMD3<Float>, target: SIMD3<Float>) {
        return getCameraAtBackCenter()
    }
}


// MARK: - Orbit Gesture View (like antimatter15/splat)

/// Gesture overlay: one-finger orbit, two-finger pan, pinch zoom. Uses UIKit so we can distinguish 1- vs 2-finger pan.
struct OrbitGestureView: View {
    var body: some View {
        WebGLGestureOverlayView()
    }
}

private struct WebGLGestureOverlayView: UIViewRepresentable {
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .clear
        view.isUserInteractionEnabled = true
        view.isMultipleTouchEnabled = true

        let oneFingerPan = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleOneFingerPan(_:)))
        oneFingerPan.minimumNumberOfTouches = 1
        oneFingerPan.maximumNumberOfTouches = 1
        view.addGestureRecognizer(oneFingerPan)

        let pinch = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePinch(_:)))
        view.addGestureRecognizer(pinch)

        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    final class Coordinator: NSObject {
        private var lastOneFingerLocation: CGPoint?
        private var lastPinchScale: CGFloat = 1.0

        @objc func handleOneFingerPan(_ recognizer: UIPanGestureRecognizer) {
            let location = recognizer.location(in: recognizer.view)
            switch recognizer.state {
            case .began:
                NotificationCenter.default.post(name: NSNotification.Name("WebGLOrbitGestureState"), object: nil, userInfo: ["interacting": true])
                lastOneFingerLocation = location
            case .changed:
                if let last = lastOneFingerLocation {
                    let deltaX = location.x - last.x
                    let deltaY = location.y - last.y
                    NotificationCenter.default.post(name: NSNotification.Name("WebGLOrbitGesture"), object: nil, userInfo: ["deltaX": deltaX, "deltaY": deltaY, "incremental": true])
                }
                lastOneFingerLocation = location
            case .ended, .cancelled:
                lastOneFingerLocation = nil
                NotificationCenter.default.post(name: NSNotification.Name("WebGLOrbitGestureState"), object: nil, userInfo: ["interacting": false])
            default:
                break
            }
        }

        @objc func handlePinch(_ recognizer: UIPinchGestureRecognizer) {
            switch recognizer.state {
            case .began:
                NotificationCenter.default.post(name: NSNotification.Name("WebGLOrbitGestureState"), object: nil, userInfo: ["interacting": true])
                lastPinchScale = recognizer.scale
            case .changed:
                let incrementalScale = recognizer.scale / lastPinchScale
                lastPinchScale = recognizer.scale
                NotificationCenter.default.post(name: NSNotification.Name("WebGLZoomGesture"), object: nil, userInfo: ["scale": incrementalScale, "incremental": true])
            case .ended, .cancelled:
                lastPinchScale = 1.0
                NotificationCenter.default.post(name: NSNotification.Name("WebGLOrbitGestureState"), object: nil, userInfo: ["interacting": false])
            default:
                break
            }
        }
    }
}

/// Simple 3D Gaussian Splat viewer using WebGL (antimatter15/splat)
/// Renders PLY files generated by SHARP
struct SharpRoomView: View {
    let plyURL: URL
    let roomMeasurements: RoomMeasurements?  // Room measurements for display
    let allowSave: Bool  // Show save button (true for new rooms, false for viewing from home)
    let photoOrientation: PhotoOrientation  // Source photo orientation (for UI layout)
    let savedRoomWidth: Float?  // Room width from saved metadata (for HomeView)
    let savedRoomHeight: Float?  // Room height from saved metadata (for HomeView)
    @Environment(\.dismiss) private var dismiss

    // Parsed bounds from PLY file (used when roomMeasurements is nil)
    private let parsedBounds: RoomBounds?

    // Convenience initializer for backwards compatibility
    init(plyURL: URL, roomMeasurements: RoomMeasurements? = nil, allowSave: Bool = true, photoOrientation: PhotoOrientation = .portrait, savedRoomWidth: Float? = nil, savedRoomHeight: Float? = nil) {
        self.plyURL = plyURL
        self.roomMeasurements = roomMeasurements
        self.allowSave = allowSave
        self.photoOrientation = photoOrientation
        self.savedRoomWidth = savedRoomWidth
        self.savedRoomHeight = savedRoomHeight

        // Compute viewer PLY URL (prefer 3DGS for SparkJS, then classic, then base)
        let basePath = plyURL.path
        let threeDGSPath = basePath.replacingOccurrences(of: ".ply", with: "_3dgs.ply")
        let classicPath = basePath.replacingOccurrences(of: ".ply", with: "_classic.ply")

        let viewerURL: URL
        if FileManager.default.fileExists(atPath: threeDGSPath) {
            viewerURL = URL(fileURLWithPath: threeDGSPath)
        } else if FileManager.default.fileExists(atPath: classicPath) {
            viewerURL = URL(fileURLWithPath: classicPath)
        } else {
            viewerURL = plyURL
        }

        // Always parse bounds from the PLY we're actually displaying
        self.parsedBounds = Self.parsePLYBounds(from: viewerURL)
    }

    /// Parse PLY file to extract position bounds (for HomeView where roomMeasurements isn't available)
    private static func parsePLYBounds(from url: URL) -> RoomBounds? {
        guard let data = try? Data(contentsOf: url) else {
            logDebug("PLY Parser: Failed to read file")
            return nil
        }

        // Find end_header
        guard let headerEndRange = data.range(of: Data("end_header\n".utf8)) else {
            logDebug("PLY Parser: No end_header found")
            return nil
        }

        // Parse header for vertex count
        guard let headerString = String(data: data[..<headerEndRange.lowerBound], encoding: .utf8) else {
            logDebug("PLY Parser: Failed to parse header")
            return nil
        }

        var vertexCount = 0
        for line in headerString.components(separatedBy: "\n") {
            if line.hasPrefix("element vertex ") {
                let parts = line.components(separatedBy: " ")
                if parts.count >= 3, let count = Int(parts[2]) {
                    vertexCount = count
                }
            }
        }

        guard vertexCount > 0 else {
            logDebug("PLY Parser: No vertices found")
            return nil
        }

        logDebug("PLY Parser: Found \(vertexCount) vertices")

        // Binary data starts after header
        let binaryStart = headerEndRange.upperBound

        // Each vertex: x,y,z (3 floats) + other properties
        // Stride: 3 floats (xyz) + 3 floats (scale) + 4 floats (rot) + 1 float (opacity) + 3 bytes (rgb) = 47 bytes
        let bytesPerVertex = 47
        let binaryDataLength = data.count - binaryStart

        var minX: Float = .greatestFiniteMagnitude
        var maxX: Float = -.greatestFiniteMagnitude
        var minY: Float = .greatestFiniteMagnitude
        var maxY: Float = -.greatestFiniteMagnitude
        var minZ: Float = .greatestFiniteMagnitude
        var maxZ: Float = -.greatestFiniteMagnitude

        let maxVertices = min(vertexCount, binaryDataLength / bytesPerVertex)

        // Use safe byte copying to avoid alignment issues
        for i in 0..<maxVertices {
            let byteOffset = binaryStart + i * bytesPerVertex
            guard byteOffset + 12 <= data.count else { continue }

            var x: Float = 0
            var y: Float = 0
            var z: Float = 0

            _ = withUnsafeMutableBytes(of: &x) { dest in
                data.copyBytes(to: dest, from: byteOffset..<(byteOffset + 4))
            }
            _ = withUnsafeMutableBytes(of: &y) { dest in
                data.copyBytes(to: dest, from: (byteOffset + 4)..<(byteOffset + 8))
            }
            _ = withUnsafeMutableBytes(of: &z) { dest in
                data.copyBytes(to: dest, from: (byteOffset + 8)..<(byteOffset + 12))
            }

            guard x.isFinite, y.isFinite, z.isFinite else { continue }
            let maxSane: Float = 1_000
            guard abs(x) <= maxSane, abs(y) <= maxSane, abs(z) <= maxSane else { continue }

            minX = min(minX, x)
            maxX = max(maxX, x)
            minY = min(minY, y)
            maxY = max(maxY, y)
            minZ = min(minZ, z)
            maxZ = max(maxZ, z)
        }

        guard minX < maxX else {
            logDebug("PLY Parser: Invalid bounds")
            return nil
        }
        let maxSane: Float = 1_000
        guard abs(minX) <= maxSane, abs(maxX) <= maxSane,
              abs(minY) <= maxSane, abs(maxY) <= maxSane,
              abs(minZ) <= maxSane, abs(maxZ) <= maxSane else {
            logDebug("PLY Parser: Rejecting bounds (abs > \(maxSane)) X[\(minX), \(maxX)] Y[\(minY), \(maxY)] Z[\(minZ), \(maxZ)]")
            return nil
        }
        let maxRoomSize: Float = 50
        let spanX = maxX - minX, spanY = maxY - minY, spanZ = maxZ - minZ
        guard spanX <= maxRoomSize, spanY <= maxRoomSize, spanZ <= maxRoomSize else {
            logDebug("PLY Parser: Rejecting bounds (room size > \(maxRoomSize)m) span X=\(spanX) Y=\(spanY) Z=\(spanZ)")
            return nil
        }

        logDebug("PLY Parser: Bounds X[\(minX), \(maxX)] Y[\(minY), \(maxY)] Z[\(minZ), \(maxZ)]")

        return RoomBounds(
            minX: minX, maxX: maxX,
            minY: minY, maxY: maxY,
            minZ: minZ, maxZ: maxZ
        )
    }

    /// Get effective bounds - always use parsed bounds from the displayed PLY file
    /// (roomMeasurements.actualBounds uses different coordinate system)
    private var effectiveBounds: RoomBounds? {
        parsedBounds
    }

    @State private var isLoading = true
    @State private var error: String?
    @State private var showingFurnitureFit = false
    @State private var fullSegmentationMode = false  // true = blue icon (multi-furniture), false = brain (one primary + margin)
    @ObservedObject private var yoloeService = YOLOEModelService.shared

    // JS-measured front wall dimensions (from actual splat bounds)
    @State private var jsFrontWallWidth: Float?
    @State private var jsFrontWallHeight: Float?

    // Detected furniture size (from FurnitureFit, in meters - before calibration)
    @State private var detectedFurnitureWidth: Float?
    @State private var detectedFurnitureHeight: Float?

    // User-input real furniture dimensions for room calibration
    @State private var showFurnitureDimensionsInput = false
    @State private var inputFurnitureHeight: String = ""
    @State private var realFurnitureHeight: Float?  // Confirmed real height in meters

    // Calibrated room dimensions (computed from real furniture size)
    @State private var calibratedRoomHeight: Float?
    @State private var calibratedRoomWidth: Float?

    // Room viewer settings
    @AppStorage("roomViewer.oscillation") private var oscillationEnabled: Bool = false
    @AppStorage("roomViewer.infiniteZoom") private var infiniteZoomEnabled: Bool = true

    // Save room state
    @StateObject private var modelManager = USDZModelManager()
    @State private var isSavingRoom = false
    @State private var saveProgress: Double = 0.0
    @State private var savingTimer: Timer?
    @State private var showSaveAlert = false
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false
    @State private var isDismissing = false
    @State private var showRoomNameInput = false
    @State private var roomName = ""
    @State private var showShareSheet = false
    @State private var isCapturingSnapshot = false
    @EnvironmentObject var authManager: AuthenticationManager

    /// Compute classic PLY URL (pre-rotated for antimatter15/splat)
    private var classicPlyURL: URL {
        let path = plyURL.path
        let classicPath = path.replacingOccurrences(of: ".ply", with: "_classic.ply")
        let classicURL = URL(fileURLWithPath: classicPath)
        if FileManager.default.fileExists(atPath: classicPath) {
            return classicURL
        }
        return plyURL
    }

    /// PLY file for WebGL viewer: prefer 3DGS (SparkJS compatible), then classic, then base
    private var viewerPlyURL: URL {
        let basePath = plyURL.path

        // Prefer 3DGS variant for Spark/SuperSplat-style viewers
        let threeDGSPath = basePath.replacingOccurrences(of: ".ply", with: "_3dgs.ply")
        if FileManager.default.fileExists(atPath: threeDGSPath) {
            return URL(fileURLWithPath: threeDGSPath)
        }

        let classicPath = basePath.replacingOccurrences(of: ".ply", with: "_classic.ply")
        if FileManager.default.fileExists(atPath: classicPath) {
            return URL(fileURLWithPath: classicPath)
        }

        return plyURL
    }

    var body: some View {
        ZStack {
            // WebGL view using SparkJS (use classic PLY - pre-rotated for antimatter viewer)
            AntimatterSplatView(
                plyURL: classicPlyURL,
                roomBounds: roomMeasurements?.boundingBox,
                actualBounds: nil,
                photoOrientation: photoOrientation,
                oscillationEnabled: oscillationEnabled,
                infiniteZoom: infiniteZoomEnabled,
                onLoaded: {
                    isLoading = false
                },
                onFrontWallDimensions: { w, h in
                    // Store JS-measured dimensions for nav title
                    jsFrontWallWidth = Float(w)
                    jsFrontWallHeight = Float(h)
                }
            )
            .ignoresSafeArea()
            .allowsHitTesting(false)  // Route all touches to SwiftUI overlays, not WKWebView

            // Gesture overlay for orbit control (captures drags and sends to WebGL)
            OrbitGestureView()
                .ignoresSafeArea()  // Cover full screen including safe areas
                .allowsHitTesting(!isLoading)  // Keep enabled even with FurnitureFit - touches outside bbox pass through
                .zIndex(10)  // Above WebGL view, below other overlays

            // Camera move buttons — up/down and left/right (tap to shift view)
            if !isLoading {
                ZStack(alignment: .topLeading) {
                    Color.clear
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .allowsHitTesting(false)
                    HStack(spacing: 8) {
                        // Left arrow — move camera left
                        Button(action: {
                            NotificationCenter.default.post(name: NSNotification.Name("WebGLCameraMoveLeft"), object: nil)
                        }) {
                            Image(systemName: "arrow.left")
                                .font(.system(size: 20, weight: .semibold))
                                .foregroundColor(.white)
                                .frame(width: 44, height: 44)
                                .background(Circle().fill(Color.black.opacity(0.5)))
                        }
                        .buttonStyle(.plain)
                        VStack(spacing: 8) {
                            Button(action: {
                                NotificationCenter.default.post(name: NSNotification.Name("WebGLCameraMoveUp"), object: nil)
                            }) {
                                Image(systemName: "arrow.up")
                                    .font(.system(size: 20, weight: .semibold))
                                    .foregroundColor(.white)
                                    .frame(width: 44, height: 44)
                                    .background(Circle().fill(Color.black.opacity(0.5)))
                            }
                            .buttonStyle(.plain)
                            Button(action: {
                                NotificationCenter.default.post(name: NSNotification.Name("WebGLCameraMoveDown"), object: nil)
                            }) {
                                Image(systemName: "arrow.down")
                                    .font(.system(size: 20, weight: .semibold))
                                    .foregroundColor(.white)
                                    .frame(width: 44, height: 44)
                                    .background(Circle().fill(Color.black.opacity(0.5)))
                            }
                            .buttonStyle(.plain)
                        }
                        // Right arrow — move camera right
                        Button(action: {
                            NotificationCenter.default.post(name: NSNotification.Name("WebGLCameraMoveRight"), object: nil)
                        }) {
                            Image(systemName: "arrow.right")
                                .font(.system(size: 20, weight: .semibold))
                                .foregroundColor(.white)
                                .frame(width: 44, height: 44)
                                .background(Circle().fill(Color.black.opacity(0.5)))
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(12)
                }
                .opacity(isCapturingSnapshot ? 0 : 1)
                .zIndex(12)
            }

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

            // FurnitureFit overlay (when active) - full screen
            if showingFurnitureFit {
                FurnitureFitUIView(
                    capturedImage: .constant(nil),
                    roomImage: nil,
                    mlModel: yoloeService.model,
                    processInterval: 0.07,
                    active: true,
                    lockedOrientation: photoOrientation,
                    fullSegmentationMode: fullSegmentationMode,
                    roomWidthMeters: roomMeasurements?.frontWallWidth ?? jsFrontWallWidth ?? 4.0,
                    roomHeightMeters: roomMeasurements?.frontWallHeight ?? jsFrontWallHeight ?? 3.0,
                    onFurnitureSizeEstimated: { width, height in
                        detectedFurnitureWidth = width
                        detectedFurnitureHeight = height
                    }
                )
                .ignoresSafeArea()
                .zIndex(100)
            }


            // Save progress overlay
            if isSavingRoom {
                saveRoomProgressOverlay
            }

            // Calibration input overlay (custom instead of sheet for landscape rotation)
            if showFurnitureDimensionsInput {
                ZStack {
                    // Dimmed background
                    Color.black.opacity(0.6)
                        .ignoresSafeArea()
                        .onTapGesture {
                            showFurnitureDimensionsInput = false
                        }

                    // Input card with custom number pad - rotated for landscape
                    VStack(spacing: 16) {
                        Text("Calibrate Room")
                            .font(.headline)
                            .foregroundColor(.white)

                        Text("Enter real furniture height (meters)")
                            .font(.caption)
                            .foregroundColor(.gray)

                        if let h = detectedFurnitureHeight {
                            Text(String(format: "Detected: %.2fm", h))
                                .font(.caption2)
                                .foregroundColor(.orange)
                        }

                        // Display current input
                        Text(inputFurnitureHeight.isEmpty ? "0.00" : inputFurnitureHeight)
                            .font(.system(size: 32, weight: .bold, design: .monospaced))
                            .foregroundColor(.white)
                            .frame(width: 120, height: 44)
                            .background(Color.white.opacity(0.1))
                            .cornerRadius(8)

                        // Custom number pad
                        VStack(spacing: 8) {
                            ForEach(0..<3) { row in
                                HStack(spacing: 8) {
                                    ForEach(1...3, id: \.self) { col in
                                        let digit = row * 3 + col
                                        Button(action: {
                                            appendDigit("\(digit)")
                                        }) {
                                            Text("\(digit)")
                                                .font(.title2.bold())
                                                .foregroundColor(.white)
                                                .frame(width: 50, height: 44)
                                                .background(Color.gray.opacity(0.3))
                                                .cornerRadius(8)
                                        }
                                    }
                                }
                            }
                            // Bottom row: "." 0 "⌫"
                            HStack(spacing: 8) {
                                Button(action: {
                                    if !inputFurnitureHeight.contains(".") {
                                        inputFurnitureHeight += inputFurnitureHeight.isEmpty ? "0." : "."
                                    }
                                }) {
                                    Text(".")
                                        .font(.title2.bold())
                                        .foregroundColor(.white)
                                        .frame(width: 50, height: 44)
                                        .background(Color.gray.opacity(0.3))
                                        .cornerRadius(8)
                                }
                                Button(action: {
                                    appendDigit("0")
                                }) {
                                    Text("0")
                                        .font(.title2.bold())
                                        .foregroundColor(.white)
                                        .frame(width: 50, height: 44)
                                        .background(Color.gray.opacity(0.3))
                                        .cornerRadius(8)
                                }
                                Button(action: {
                                    if !inputFurnitureHeight.isEmpty {
                                        inputFurnitureHeight.removeLast()
                                    }
                                }) {
                                    Image(systemName: "delete.left")
                                        .font(.title3)
                                        .foregroundColor(.white)
                                        .frame(width: 50, height: 44)
                                        .background(Color.gray.opacity(0.3))
                                        .cornerRadius(8)
                                }
                            }
                        }

                        HStack(spacing: 16) {
                            Button("Cancel") {
                                inputFurnitureHeight = ""
                                showFurnitureDimensionsInput = false
                            }
                            .font(.body.bold())
                            .foregroundColor(.red)
                            .frame(width: 80, height: 40)
                            .background(Color.red.opacity(0.2))
                            .cornerRadius(8)

                            Button("Apply") {
                                if let realHeight = Float(inputFurnitureHeight),
                                   let detectedHeight = detectedFurnitureHeight,
                                   detectedHeight > 0 {
                                    realFurnitureHeight = realHeight
                                    let scaleFactor = realHeight / detectedHeight

                                    // Scale the room to match real furniture size
                                    NotificationCenter.default.post(
                                        name: NSNotification.Name("WebGLScaleRoom"),
                                        object: nil,
                                        userInfo: ["factor": Double(scaleFactor)]
                                    )

                                    // Update calibrated room dimensions
                                    if let roomH = roomMeasurements?.frontWallHeight ?? jsFrontWallHeight {
                                        calibratedRoomHeight = roomH * scaleFactor
                                    }
                                    if let roomW = roomMeasurements?.frontWallWidth ?? jsFrontWallWidth {
                                        calibratedRoomWidth = roomW * scaleFactor
                                    }

                                    logDebug("📐 [Calibration] Real height: \(realHeight)m, Scale factor: \(scaleFactor)")
                                }
                                inputFurnitureHeight = ""
                                showFurnitureDimensionsInput = false
                            }
                            .font(.body.bold())
                            .foregroundColor(.green)
                            .frame(width: 80, height: 40)
                            .background(Color.green.opacity(0.2))
                            .cornerRadius(8)
                            .disabled(Float(inputFurnitureHeight) == nil || inputFurnitureHeight.isEmpty)
                        }
                    }
                    .padding(20)
                    .background(Color.black.opacity(0.95))
                    .cornerRadius(16)
                }
                .zIndex(99999)  // Above everything
            }

            // Landscape layout: normal bottom bar (device actually in landscape mode)
            // Use ZStack so only the bar receives touches; rest pass through to OrbitGestureView
            if photoOrientation == .landscape {
                ZStack(alignment: .bottom) {
                    Color.clear
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .allowsHitTesting(false)
                    HStack(spacing: 20) {
                        // Full segmentation (blue) – multi-furniture
                        Button(action: {
                            if showingFurnitureFit {
                                showingFurnitureFit = false
                            } else {
                                fullSegmentationMode = true
                                SHARPService.shared.releaseResources()
                                showingFurnitureFit = true
                            }
                        }) {
                            Image(systemName: "square.on.square.dashed")
                                .font(.system(size: 26))
                                .foregroundColor(.white)
                                .frame(width: 56, height: 56)
                                .background(Circle().fill(showingFurnitureFit && fullSegmentationMode ? Color.cyan : Color.blue).shadow(radius: 5))
                        }
                        .disabled(isLoading)

                        // Brain button (one primary + 10% margin)
                        Button(action: {
                            if showingFurnitureFit {
                                showingFurnitureFit = false
                            } else {
                                fullSegmentationMode = false
                                SHARPService.shared.releaseResources()
                                showingFurnitureFit = true
                            }
                        }) {
                            Image(systemName: "brain.head.profile")
                                .font(.system(size: 28))
                                .foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(Circle().fill(showingFurnitureFit && !fullSegmentationMode ? Color.green : Color.blue).shadow(radius: 5))
                        }
                        .disabled(isLoading)

                        // Orientation label — allow pinch/drag to pass through to OrbitGestureView
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
                        .allowsHitTesting(false)

                        // Furniture calibration (if active)
                        if showingFurnitureFit, let h = detectedFurnitureHeight {
                            Button(action: {
                                showFurnitureDimensionsInput = true
                            }) {
                                VStack(spacing: 2) {
                                    if let calibH = calibratedRoomHeight {
                                        Text(String(format: "Room: %.2fm", calibH))
                                            .font(.caption2)
                                            .foregroundColor(.green)
                                    }
                                    Text(String(format: "Furn: %.2fm", realFurnitureHeight ?? h))
                                        .font(.caption.bold())
                                        .foregroundColor(realFurnitureHeight != nil ? .green : .white)
                                    Text("Tap to calibrate")
                                        .font(.system(size: 9))
                                        .foregroundColor(.gray)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.black.opacity(0.6))
                                .cornerRadius(6)
                            }
                        }

                        Spacer()
                            .allowsHitTesting(false)

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
                .opacity(isCapturingSnapshot ? 0 : 1)
                .zIndex(99997)
            } else {
                // Portrait layout: standard bottom controls
                // NOTE: OrbitGestureView (zIndex 10) handles all camera gestures for WebGL
                // Do NOT add SimpleJoystickOverlay here - it sends to GlobalCameraController
                // which has no camera registered for WebGL rooms

                // Buttons at bottom row
                VStack {
                    Spacer()
                        .allowsHitTesting(false)

                    // Orientation label (center, above buttons) — allow pinch/drag to pass through
                    VStack(spacing: 1) {
                        Text(NSLocalizedString("orientation.heldVertically", comment: ""))
                            .font(.caption2)
                        Text(NSLocalizedString("orientation.portrait", comment: ""))
                            .font(.caption2)
                            .fontWeight(.medium)
                    }
                    .foregroundColor(.white.opacity(0.8))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.black.opacity(0.4))
                    .cornerRadius(6)
                    .padding(.bottom, 12)
                    .allowsHitTesting(false)

                    HStack(spacing: 12) {
                        // Full segmentation (blue) – multi-furniture
                        Button(action: {
                            if showingFurnitureFit {
                                showingFurnitureFit = false
                            } else {
                                fullSegmentationMode = true
                                SHARPService.shared.releaseResources()
                                showingFurnitureFit = true
                            }
                        }) {
                            Image(systemName: "square.on.square.dashed")
                                .font(.system(size: 26))
                                .foregroundColor(.white)
                                .frame(width: 56, height: 56)
                                .background(Circle().fill(showingFurnitureFit && fullSegmentationMode ? Color.cyan : Color.blue).shadow(radius: 5))
                        }
                        .disabled(isLoading)

                        // Brain button (one primary + 10% margin)
                        Button(action: {
                            if showingFurnitureFit {
                                showingFurnitureFit = false
                            } else {
                                fullSegmentationMode = false
                                SHARPService.shared.releaseResources()
                                showingFurnitureFit = true
                            }
                        }) {
                            Image(systemName: "brain.head.profile")
                                .font(.system(size: 28))
                                .foregroundColor(.white)
                                .frame(width: 60, height: 60)
                                .background(Circle().fill(showingFurnitureFit && !fullSegmentationMode ? Color.green : Color.blue).shadow(radius: 5))
                        }
                        .disabled(isLoading)

                        Spacer()
                            .allowsHitTesting(false)

                        // Furniture size display + Screenshot button (bottom-right) — buttons remain tappable
                        VStack(spacing: 8) {
                            // Show detected furniture size - tap to input real size for calibration
                            if showingFurnitureFit, let h = detectedFurnitureHeight {
                                Button(action: {
                                    showFurnitureDimensionsInput = true
                                }) {
                                    VStack(spacing: 2) {
                                        if let calibH = calibratedRoomHeight {
                                            // Show calibrated room dimensions
                                            Text(String(format: "Room: %.2fm", calibH))
                                                .font(.caption2)
                                                .foregroundColor(.green)
                                        }
                                        Text(String(format: "Furn: %.2fm", realFurnitureHeight ?? h))
                                            .font(.caption.bold())
                                            .foregroundColor(realFurnitureHeight != nil ? .green : .white)
                                        Text("Tap to calibrate")
                                            .font(.system(size: 9))
                                            .foregroundColor(.gray)
                                    }
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.black.opacity(0.6))
                                    .cornerRadius(6)
                                }
                            }

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
                        .padding(.trailing, 16)
                    }
                    .padding(.leading, 16)
                    .padding(.bottom, 20)
                }
                .opacity(isCapturingSnapshot ? 0 : 1)
                .zIndex(99998)  // Higher than joystick (99997) to allow button taps
            }
        }
        .background(Color.gray)
        .navigationBarHidden(isCapturingSnapshot)
        .navigationTitle(navigationTitleWithDimensions)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .navigationBarTrailing) {
                // Share button - only for authorized users
                if authManager.canShare {
                    Button(action: {
                        showShareSheet = true
                    }) {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .disabled(isLoading)
                }

                // Save button (only for new rooms, not when viewing from home)
                if allowSave {
                    Button(action: {
                        showRoomNameInput = true
                    }) {
                        Image(systemName: "square.and.arrow.down")
                    }
                    .disabled(isLoading)
                }

                // Recenter button
                Button(action: {
                    NotificationCenter.default.post(name: NSNotification.Name("RecenterWebGLCamera"), object: nil)
                }) {
                    Image(systemName: "viewfinder")
                }
                .disabled(isLoading)
            }
        }
        .sheet(isPresented: $showShareSheet) {
            // Share the 3DGS format (SuperSplat compatible)
            let threeDGSPath = plyURL.path.replacingOccurrences(of: ".ply", with: "_3dgs.ply")
            let shareURL = FileManager.default.fileExists(atPath: threeDGSPath)
                ? URL(fileURLWithPath: threeDGSPath)
                : plyURL
            ShareSheet(activityItems: [shareURL])
        }
        .onAppear {
            yoloeService.ensureModelLoaded()
            // Lock orientation based on photo orientation
            if photoOrientation == .landscape {
                OrientationLockManager.shared.lockToLandscape()
            } else {
                OrientationLockManager.shared.lockToPortrait()
            }
            logDebug("📐 [SharpRoomView] photoOrientation = \(photoOrientation)")
        }
        .onDisappear {
            // Unlock orientation when leaving the view
            OrientationLockManager.shared.unlock()
        }
        // Room name input alert
        .alert("Save Room", isPresented: $showRoomNameInput) {
            TextField("Room name", text: $roomName)
            Button("Cancel", role: .cancel) {
                roomName = ""
            }
            Button("Save") {
                startSavingRoom()
            }
            .disabled(roomName.isEmpty)
        } message: {
            Text(NSLocalizedString("roomViewer.enterName", comment: ""))
        }
        // Save result alert
        .alert("Room Save", isPresented: $showSaveAlert) {
            Button("OK", role: .cancel) {
                if saveWasSuccessful {
                    // Show dismissing indicator
                    isDismissing = true
                    // Release resources in background, then dismiss
                    DispatchQueue.global(qos: .userInitiated).async {
                        // Release SHARP model memory
                        DispatchQueue.main.async {
                            SHARPService.shared.releaseResources()
                        }
                        // Small delay to let cleanup complete
                        Thread.sleep(forTimeInterval: 0.1)
                        DispatchQueue.main.async {
                            // Dismiss entire sheet and go back to home view
                            NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil)
                        }
                    }
                }
            }
        } message: {
            Text(saveAlertMessage)
        }
        // Dismissing overlay
        .overlay {
            if isDismissing {
                ZStack {
                    Color.black.opacity(0.7)
                        .ignoresSafeArea()
                    VStack(spacing: 16) {
                        ProgressView()
                            .scaleEffect(1.5)
                            .tint(.white)
                        Text("Going back...")
                            .foregroundColor(.white)
                            .font(.headline)
                    }
                }
            }
        }
        .defersSystemGestures(on: .all)
    }

    // MARK: - Number Pad Helper

    private func appendDigit(_ digit: String) {
        // Limit to reasonable length (e.g., "12.34")
        if inputFurnitureHeight.count >= 5 { return }
        // Limit decimal places to 2
        if let dotIndex = inputFurnitureHeight.firstIndex(of: ".") {
            let decimals = inputFurnitureHeight.distance(from: dotIndex, to: inputFurnitureHeight.endIndex) - 1
            if decimals >= 2 { return }
        }
        inputFurnitureHeight += digit
    }

    // MARK: - Screenshot

    private func takeScreenshot() {
        logDebug("📸 Taking screenshot...")
        isCapturingSnapshot = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
            let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
            let windows = scenes.flatMap { $0.windows }
            guard let window = windows.first(where: { $0.isKeyWindow }) ?? windows.first else {
                isCapturingSnapshot = false
                logDebug("❌ No window found")
                return
            }
            let format = UIGraphicsImageRendererFormat()
            format.scale = UIScreen.main.scale
            let renderer = UIGraphicsImageRenderer(bounds: window.bounds, format: format)
            let image = renderer.image { _ in
                window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
            }
            logDebug("📸 Screenshot captured, saving to Photos...")
            UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
            logDebug("✅ Screenshot saved to Photos")
            DispatchQueue.main.async {
                isCapturingSnapshot = false
            }
        }
    }

    // MARK: - YOLOE model loaded via YOLOEModelService (ODR)

    // MARK: - Navigation Title with Dimensions
    private var navigationTitleWithDimensions: String {
        // Show front wall dimensions (width × height)
        // Priority: roomMeasurements -> saved metadata -> JS-measured -> default
        // Use saved dimensions first since they're available immediately
        let wallWidth = roomMeasurements?.frontWallWidth
            ?? savedRoomWidth
            ?? jsFrontWallWidth
            ?? 4.0
        let wallHeight = roomMeasurements?.frontWallHeight
            ?? savedRoomHeight
            ?? jsFrontWallHeight
            ?? 3.0

        return String(format: "%.1f × %.1f m", wallWidth, wallHeight)
    }

    // MARK: - Save Room Progress Overlay
    private var saveRoomProgressOverlay: some View {
        ZStack {
            Color.black.opacity(0.9)
                .ignoresSafeArea()

            VStack(spacing: 24) {
                ZStack {
                    Circle()
                        .fill(Color.green.opacity(0.2))
                        .frame(width: 100, height: 100)

                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 40))
                        .foregroundColor(.green)
                }

                Text("Saving Room...")
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)

                ProgressView(value: saveProgress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .green))
                    .frame(width: 200)

                Text("\(Int(saveProgress * 100))%")
                    .font(.headline)
                    .foregroundColor(.gray)

                Button("Cancel") {
                    cancelSavingRoom()
                }
                .foregroundColor(.red)
                .padding(.top, 20)
            }
        }
    }

    // MARK: - Save Room Functions
    private func startSavingRoom() {
        guard !roomName.isEmpty else { return }

        let savedName = roomName
        logDebug("💾 [SharpRoomView] Starting room save: \(savedName)")

        withAnimation(.easeIn(duration: 0.3)) {
            isSavingRoom = true
            saveProgress = 0.0
        }

        var saveStarted = false
        var saveCompleted = false
        var saveSuccess = false
        var saveError: String?

        savingTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { timer in
            if !saveStarted || (saveStarted && saveCompleted) {
                self.saveProgress += 0.015
            }

            if self.saveProgress >= 0.6 && !saveStarted {
                saveStarted = true
                let width = self.jsFrontWallWidth
                let height = self.jsFrontWallHeight
                // Save the classic PLY file (pre-transformed for correct viewing)
                self.modelManager.savePLY(from: self.classicPlyURL, name: savedName, photoOrientation: self.photoOrientation, roomWidth: width, roomHeight: height) { success, error in
                    DispatchQueue.main.async {
                        saveCompleted = true
                        saveSuccess = success
                        saveError = error
                        logDebug(success ? "✅ [SharpRoomView] Room saved" : "❌ [SharpRoomView] Save failed: \(error ?? "unknown")")
                    }
                }
            }

            if self.saveProgress >= 1.0 && saveCompleted {
                timer.invalidate()
                self.savingTimer = nil

                withAnimation(.easeOut(duration: 0.3)) {
                    self.isSavingRoom = false
                }

                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    if saveSuccess {
                        self.saveAlertMessage = "Room '\(savedName)' saved successfully!"
                        self.saveWasSuccessful = true
                    } else {
                        self.saveAlertMessage = "Failed to save: \(saveError ?? "Unknown error")"
                        self.saveWasSuccessful = false
                    }
                    self.showSaveAlert = true
                    self.roomName = ""
                }
            }
        }
    }

    private func cancelSavingRoom() {
        savingTimer?.invalidate()
        savingTimer = nil

        withAnimation(.easeOut(duration: 0.2)) {
            isSavingRoom = false
            saveProgress = 0.0
        }

        roomName = ""
        logDebug("❌ [SharpRoomView] Room save cancelled")
    }

}

// MARK: - Antimatter15 WebGL Splat View

/// Custom URL scheme handler to serve HTML and PLY files to WKWebView
class LocalFileSchemeHandler: NSObject, WKURLSchemeHandler {
    var plyURL: URL?
    var htmlContent: String = ""
    var loadStartTime: CFAbsoluteTime = 0

    func webView(_ webView: WKWebView, start urlSchemeTask: WKURLSchemeTask) {
        guard let requestURL = urlSchemeTask.request.url else {
            urlSchemeTask.didFailWithError(NSError(domain: "LocalFileSchemeHandler", code: -1, userInfo: nil))
            return
        }

        let path = requestURL.path
        logDebug("LocalFileSchemeHandler: Request for \(path)")

        // Serve the HTML page
        if path == "/" || path == "/index.html" || path.isEmpty {
            let data = htmlContent.data(using: .utf8) ?? Data()
            let response = HTTPURLResponse(
                url: requestURL,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: [
                    "Content-Type": "text/html; charset=utf-8",
                    "Content-Length": "\(data.count)"
                ]
            )!
            urlSchemeTask.didReceive(response)
            urlSchemeTask.didReceive(data)
            urlSchemeTask.didFinish()
            logDebug("LocalFileSchemeHandler: Served HTML (\(data.count) bytes)")
            return
        }

        // Serve the PLY file
        if path == "/room.ply", let plyURL = plyURL {
            do {
                let readStart = CFAbsoluteTimeGetCurrent()
                let data = try Data(contentsOf: plyURL)
                let readTime = (CFAbsoluteTimeGetCurrent() - readStart) * 1000

                let response = HTTPURLResponse(
                    url: requestURL,
                    statusCode: 200,
                    httpVersion: "HTTP/1.1",
                    headerFields: [
                        "Content-Type": "application/octet-stream",
                        "Content-Length": "\(data.count)",
                        "Access-Control-Allow-Origin": "*"
                    ]
                )!
                urlSchemeTask.didReceive(response)
                urlSchemeTask.didReceive(data)
                urlSchemeTask.didFinish()

                let sizeMB = Double(data.count) / (1024 * 1024)
                logDebug("⏱️ [WebGL] PLY file read: \(String(format: "%.0f", readTime))ms (\(String(format: "%.1f", sizeMB)) MB)")
            } catch {
                logDebug("LocalFileSchemeHandler: Failed to read PLY: \(error)")
                urlSchemeTask.didFailWithError(error)
            }
            return
        }

        // Serve the local main.js (modified antimatter15/splat)
        if path == "/main.js" {
            // Try bundle resource
            if let jsURL = Bundle.main.url(forResource: "splat_main", withExtension: "js"),
               let data = try? Data(contentsOf: jsURL) {
                let response = HTTPURLResponse(
                    url: requestURL,
                    statusCode: 200,
                    httpVersion: "HTTP/1.1",
                    headerFields: [
                        "Content-Type": "application/javascript; charset=utf-8",
                        "Content-Length": "\(data.count)"
                    ]
                )!
                urlSchemeTask.didReceive(response)
                urlSchemeTask.didReceive(data)
                urlSchemeTask.didFinish()
                logDebug("LocalFileSchemeHandler: Served main.js from bundle (\(data.count) bytes)")
                return
            }

            logDebug("LocalFileSchemeHandler: main.js not found in bundle!")
        }

        // 404 for other paths
        let errorResponse = HTTPURLResponse(
            url: requestURL,
            statusCode: 404,
            httpVersion: "HTTP/1.1",
            headerFields: nil
        )!
        urlSchemeTask.didReceive(errorResponse)
        urlSchemeTask.didReceive(Data())
        urlSchemeTask.didFinish()
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: WKURLSchemeTask) {
        // Nothing to clean up
    }
}

/// WebGL-based 3D Gaussian Splat viewer using antimatter15/splat
/// Renders PLY files in a WKWebView using WebGL
struct AntimatterSplatView: UIViewRepresentable {
    let plyURL: URL
    let roomBounds: SIMD3<Float>?  // Room dimensions for camera positioning
    let actualBounds: RoomBounds?  // Actual min/max bounds for precise framing
    let photoOrientation: PhotoOrientation  // For orientation-based room rotation
    let oscillationEnabled: Bool  // Auto-orbit setting from user preferences
    let infiniteZoom: Bool  // If true, no zoom/distance limits; can pass through walls
    let onLoaded: () -> Void
    var onFrontWallDimensions: ((Double, Double) -> Void)? = nil

    func makeCoordinator() -> Coordinator {
        Coordinator(onLoaded: onLoaded, onFrontWallDimensions: onFrontWallDimensions)
    }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true

        // Enable JavaScript
        let prefs = WKWebpagePreferences()
        prefs.allowsContentJavaScript = true
        config.defaultWebpagePreferences = prefs

        // Register custom URL scheme handler
        let schemeHandler = LocalFileSchemeHandler()
        schemeHandler.plyURL = plyURL
        schemeHandler.htmlContent = generateSplatViewerHTML(bounds: roomBounds, actualBounds: actualBounds, orientation: self.photoOrientation, oscillation: oscillationEnabled, infiniteZoom: infiniteZoom)
        config.setURLSchemeHandler(schemeHandler, forURLScheme: "splat")

        // Add message handlers for communication from JS
        config.userContentController.add(context.coordinator, name: "splatLoaded")
        config.userContentController.add(context.coordinator, name: "frontWallDimensions")
        config.userContentController.add(context.coordinator, name: "cameraPose")
        config.userContentController.add(context.coordinator, name: "jsLog")

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        webView.scrollView.isScrollEnabled = false
        webView.scrollView.bounces = false
        webView.scrollView.bouncesZoom = false
        webView.scrollView.minimumZoomScale = 1.0
        webView.scrollView.maximumZoomScale = 1.0
        webView.scrollView.contentInsetAdjustmentBehavior = .never

        // Disable all scroll view gesture recognizers to let Three.js handle touches
        for gestureRecognizer in webView.scrollView.gestureRecognizers ?? [] {
            gestureRecognizer.isEnabled = false
        }

        webView.isOpaque = false
        webView.backgroundColor = .gray
        webView.isUserInteractionEnabled = true
        webView.isMultipleTouchEnabled = true

        // Store reference
        context.coordinator.schemeHandler = schemeHandler
        context.coordinator.webView = webView
        context.coordinator.lastInfiniteZoom = infiniteZoom

        // Load from custom scheme (HTML will auto-load room.ply)
        if let url = URL(string: "splat://local/") {
            webView.load(URLRequest(url: url))
            logDebug("AntimatterSplatView: Loading from custom scheme")
        }

        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        // Reload viewer when Infinite Zoom changes so the setting takes effect without leaving the room
        if context.coordinator.lastInfiniteZoom != infiniteZoom {
            context.coordinator.lastInfiniteZoom = infiniteZoom
            context.coordinator.schemeHandler?.htmlContent = generateSplatViewerHTML(bounds: roomBounds, actualBounds: actualBounds, orientation: photoOrientation, oscillation: oscillationEnabled, infiniteZoom: infiniteZoom)
            uiView.reload()
        }
    }

    private func generateSplatViewerHTML(bounds: SIMD3<Float>?, actualBounds: RoomBounds?, orientation: PhotoOrientation, oscillation: Bool, infiniteZoom: Bool) -> String {
        // Camera and framing come from Box3 in the viewer only. No Swift bounds injection.
        logDebug("📐 [WebGL] Framing from Box3 only (no Swift bounds)")

        // SparkJS + THREE.js based Gaussian Splat viewer
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <title>3D Room</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                html, body {
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    background: #808080;
                    touch-action: none;
                    -webkit-touch-callout: none;
                    -webkit-user-select: none;
                }
                canvas {
                    width: 100%;
                    height: 100%;
                    display: block;
                    touch-action: none;
                }
            </style>
            <script type="importmap">
            {
                "imports": {
                    "three": "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.170.0/three.module.min.js",
                    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/",
                    "@sparkjsdev/spark": "https://sparkjs.dev/releases/spark/0.1.10/spark.module.js"
                }
            }
            </script>
        </head>
        <body>
            <script type="module">
                import * as THREE from 'three';
                import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
                import { SplatMesh, SparkRenderer } from '@sparkjsdev/spark';

                // Photo orientation from Swift
                const isPortrait = \(orientation == .portrait ? "true" : "false");
                console.log('[WebGL] isPortrait =', isPortrait);

                // Oscillation setting from Swift
                const OSCILLATION_ENABLED = \(oscillation ? "true" : "false");
                console.log('[WebGL] oscillation =', OSCILLATION_ENABLED);

                // Infinite zoom: no distance/room bounds clamping (can pass through walls into void)
                const INFINITE_ZOOM = \(infiniteZoom ? "true" : "false");
                console.log('[WebGL] infiniteZoom =', INFINITE_ZOOM);

                // Log JS errors
                window.addEventListener('error', function(e) {
                    console.log('[JS Error] ' + e.message + ' at ' + e.filename + ':' + e.lineno);
                });

                const jsStartTime = performance.now();

                // Scene setup
                // Use neutral gray background to blend better with room colors
                // (black makes holes/sparse areas very obvious)
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x808080);

                // Camera - start facing origin, autoFrameRoom will reposition using Box3. Infinite zoom: smaller near for close-up.
                const cameraNear = INFINITE_ZOOM ? 0.001 : 0.1;
                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, cameraNear, 1000);
                camera.position.set(0, 0, 3);  // Closer initial position
                camera.lookAt(0, 0, 0);        // Explicitly look at origin
                camera.up.set(0, 1, 0);        // Normal Y-up

                // THREE.js Renderer (for SparkRenderer)
                const renderer = new THREE.WebGLRenderer({ antialias: false });  // antialias: false per SparkJS docs
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(window.devicePixelRatio);
                document.body.appendChild(renderer.domElement);

                // SparkRenderer with settings tuned to reduce visible holes
                // - Increased maxStdDev allows larger gaussians to fill gaps
                // - preBlurAmount adds soft edges to blend sparse areas
                // - falloff controls opacity falloff (lower = more spread)
                const spark = new SparkRenderer({
                    renderer: renderer,
                    maxStdDev: 3.0,           // Larger gaussians to fill gaps
                    preBlurAmount: 0.5,       // Soft blur to blend edges
                    blurAmount: 0.3,          // Slightly more post-blur
                    falloff: 0.8,             // Gentler opacity falloff
                    focalAdjustment: 1.5      // Slightly less aggressive focal adjustment
                });
                camera.add(spark);  // Add SparkRenderer as child of camera

                // Battery optimization: only render when needed
                let needsRender = true;  // Force initial render

                // Orbit controls for touch/mouse
                const controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.08;  // Smooth deceleration
                controls.rotateSpeed = 1.5;     // Faster rotation
                controls.zoomSpeed = 2.0;       // Faster zoom
                controls.screenSpacePanning = false;
                // Portrait: allow zoom into wall (0.001); landscape 0.01 (match Android). Infinite zoom: 0 = no minimum, can pass through target.
                controls.minDistance = INFINITE_ZOOM ? 0 : (isPortrait ? 0.001 : 0.01);
                controls.maxDistance = INFINITE_ZOOM ? 1e6 : 100;
                // Default target, autoFrameRoom will update using Box3
                controls.target.set(0, 0, 0);

                // No full 360° spin - we'll do back-and-forth oscillation instead
                controls.autoRotate = false;

                // Request render when controls change (handles damping animation too)
                controls.addEventListener('change', function() {
                    needsRender = true;
                });

                // --- Back-and-forth orbit state ---
                let autoOrbitEnabled = OSCILLATION_ENABLED;  // Read from Swift setting
                let autoOrbitTime = 0;             // time accumulator
                let autoOrbitBaseAngle = 0;        // center angle around target
                let autoOrbitRadius = 5;           // distance from target to camera
                const clock = new THREE.Clock();

                // Global interaction flag (used by auto-orbit)
                window._userInteracting = false;

                // Called from Swift AND by OrbitControls
                window.setUserInteracting = function(flag) {
                    window._userInteracting = !!flag;
                };

                // Toggle auto-orbit from Swift
                window.setAutoOrbit = function(enabled) {
                    autoOrbitEnabled = enabled;
                    console.log('Auto-orbit:', enabled ? 'enabled' : 'disabled');
                    needsRender = true;
                };

                // Let OrbitControls also toggle interaction
                controls.addEventListener('start', () => { window._userInteracting = true; });
                controls.addEventListener('end',   () => { window._userInteracting = false; });

                // Unlimited spin around object (train-style orbit)
                controls.minAzimuthAngle = -Infinity;
                controls.maxAzimuthAngle =  Infinity;
                // Relax polar limits so zoom-into-wall works (match Android)
                controls.minPolarAngle = 0.001;
                controls.maxPolarAngle = Math.PI - 0.001;

                // Camera clamping disabled for train-style free orbit
                // Re-enable if needed for room boundary constraints
                /*
                controls.addEventListener('change', () => {
                    if (!roomBoundsForClamping) return;
                    const margin = 0.1;
                    const b = roomBoundsForClamping;
                    camera.position.x = Math.max(b.minX + margin, Math.min(b.maxX - margin, camera.position.x));
                    camera.position.z = Math.max(b.minZ + margin, Math.min(b.maxZ - margin, camera.position.z));
                    camera.position.y = Math.max(b.minY - margin, Math.min(b.maxY + margin, camera.position.y));
                });
                */

                // Save initial camera state for recenter (will be updated by autoFrameRoom)
                let initialCameraPosition = camera.position.clone();
                let initialControlsTarget = controls.target.clone();

                // Recenter function
                window.recenterCamera = function() {
                    camera.position.copy(initialCameraPosition);
                    controls.target.copy(initialControlsTarget);
                    controls.update();
                    needsRender = true;
                };

                // Scale room function - called from Swift when user calibrates with real furniture
                window.scaleRoom = function(factor) {
                    if (splatMesh) {
                        splatMesh.scale.set(factor, factor, factor);
                        console.log('Room scaled by factor:', factor);
                        needsRender = true;
                        // Re-frame after scaling
                        setTimeout(autoFrameRoom, 100);
                    }
                };

                // Function to update initial position (called after auto-frame)
                window.updateInitialPosition = function() {
                    initialCameraPosition = camera.position.clone();
                    initialControlsTarget = controls.target.clone();
                    console.log('Updated initial position for recenter:', initialCameraPosition);
                };

                // Load PLY splat using SparkJS
                const plyURL = 'splat://local/room.ply';
                console.log('Loading splat from:', plyURL);

                let splatMesh = null;

                try {
                    splatMesh = new SplatMesh({
                        url: plyURL,
                        maxSh: 0  // Disable spherical harmonics for cleaner look
                    });
                    scene.add(splatMesh);

                    // Classic PLY rotation - flip 180° around X
                    splatMesh.rotation.x = Math.PI;
                    // For portrait photos, also rotate 90° around Z
                    if (isPortrait) {
                        splatMesh.rotation.z = Math.PI / 2;
                        console.log('SplatMesh: rotated 180° X + 90° Z (portrait)');
                    } else {
                        console.log('SplatMesh: rotated 180° X only (landscape)');
                    }

                    let autoFrameRoomRetryCount = 0;
                    const AUTO_FRAME_MAX_RETRIES = 60;

                    function autoFrameRoom() {
                        const jsLog = function(msg) { if (window.webkit?.messageHandlers?.jsLog) window.webkit.messageHandlers.jsLog.postMessage(msg); };
                        try {
                            if (!splatMesh) {
                                autoFrameRoomRetryCount++;
                                if (autoFrameRoomRetryCount <= 3) {
                                    jsLog('No splatMesh yet, retry ' + autoFrameRoomRetryCount);
                                    setTimeout(autoFrameRoom, 300);
                                } else {
                                    jsLog('Giving up: no splatMesh after ' + autoFrameRoomRetryCount + ' retries — using default camera');
                                    camera.position.set(0, 0, 4);
                                    controls.target.set(0, 0, 0);
                                    controls.update();
                                    needsRender = true;
                                }
                                return;
                            }
                            // SplatMesh is not a standard Three.js Mesh — setFromObject() sees no geometry and returns (0,0,0).
                            // Use SparkJS getBoundingBox() which uses actual splat positions. Must wait until initialized.
                            if (!splatMesh.isInitialized) {
                                autoFrameRoomRetryCount++;
                                if (autoFrameRoomRetryCount <= 3 || autoFrameRoomRetryCount % 15 === 0) {
                                    jsLog('SplatMesh not initialized yet, retry ' + autoFrameRoomRetryCount);
                                }
                                if (autoFrameRoomRetryCount < AUTO_FRAME_MAX_RETRIES) {
                                    setTimeout(autoFrameRoom, 200);
                                } else {
                                    jsLog('Giving up: SplatMesh never initialized — using default camera');
                                    camera.position.set(0, 0, 4);
                                    controls.target.set(0, 0, 0);
                                    controls.update();
                                    needsRender = true;
                                }
                                return;
                            }
                            splatMesh.updateMatrixWorld(true);
                            let box;
                            try {
                                const localBox = splatMesh.getBoundingBox(true);
                                box = localBox.clone().applyMatrix4(splatMesh.matrixWorld);
                            } catch (e) {
                                jsLog('getBoundingBox failed: ' + e.message);
                                autoFrameRoomRetryCount++;
                                if (autoFrameRoomRetryCount < AUTO_FRAME_MAX_RETRIES) {
                                    setTimeout(autoFrameRoom, 300);
                                } else {
                                    camera.position.set(0, 0, 4);
                                    controls.target.set(0, 0, 0);
                                    controls.update();
                                    needsRender = true;
                                }
                                return;
                            }
                            const size = box.getSize(new THREE.Vector3());
                            if (size.length() < 0.01) {
                                autoFrameRoomRetryCount++;
                                if (autoFrameRoomRetryCount < AUTO_FRAME_MAX_RETRIES) {
                                    jsLog('Box3 size still zero after getBoundingBox, retry ' + autoFrameRoomRetryCount);
                                    setTimeout(autoFrameRoom, 300);
                                } else {
                                    jsLog('Giving up: Box3 stayed zero — using default camera');
                                    camera.position.set(0, 0, 4);
                                    controls.target.set(0, 0, 0);
                                    controls.update();
                                    needsRender = true;
                                }
                                return;
                            }

                            autoFrameRoomRetryCount = 0;
                            const center = box.getCenter(new THREE.Vector3());
                            // After 90° Z rotation, axes mapping depends on photo orientation
                            // Portrait: width = X (narrower), height = Y (taller)
                            // Landscape: width = Y, height = X
                            let roomWidth, roomHeight;
                            if (isPortrait) {
                                roomWidth  = size.x;
                                roomHeight = size.y;
                            } else {
                                roomWidth  = size.y;
                                roomHeight = size.x;
                            }
                            let roomDepth  = size.z;

                            const box3Msg = 'Box3 raw size: ' + roomWidth.toFixed(2) + ' x ' + roomHeight.toFixed(2) + ' (isPortrait: ' + isPortrait + ')';
                            console.log(box3Msg);
                            if (window.webkit?.messageHandlers?.jsLog) window.webkit.messageHandlers.jsLog.postMessage(box3Msg);

                            // Cap to realistic room dimensions (fog makes bounds too large)
                            const maxRealisticWidth = isPortrait ? 5.0 : 8.0;
                            const maxRealisticHeight = isPortrait ? 3.5 : 3.2;
                            if (roomWidth > maxRealisticWidth) {
                                console.log('Capping width from', roomWidth.toFixed(2), 'to', maxRealisticWidth);
                                roomWidth = maxRealisticWidth;
                            }
                            if (roomHeight > maxRealisticHeight) {
                                console.log('Capping height from', roomHeight.toFixed(2), 'to', maxRealisticHeight);
                                roomHeight = maxRealisticHeight;
                            }

                            console.log('Box3 capped size:', roomWidth.toFixed(2), roomHeight.toFixed(2), roomDepth.toFixed(2));
                            console.log('Box3 center:', center.x.toFixed(2), center.y.toFixed(2), center.z.toFixed(2));

                            // 2) Raw bounds from Box3
                            let rawMinX = box.min.x;
                            let rawMaxX = box.max.x;
                            let rawMinY = box.min.y;
                            let rawMaxY = box.max.y;
                            let rawMinZ = box.min.z;
                            let rawMaxZ = box.max.z;

                            // 3) Shrink bounds to ignore foggy outer 15% on front/sides; allow camera right up to back wall
                            const fogFactor = 0.15;
                            const shrinkX = roomWidth  * fogFactor * 0.5;
                            const shrinkY = roomHeight * fogFactor * 0.5;
                            const shrinkZ = roomDepth  * fogFactor * 0.5;
                            const backWallInset = 0.02;

                            const minX = rawMinX + shrinkX;
                            const maxX = rawMaxX - shrinkX;
                            const minY = rawMinY + shrinkY;
                            const maxY = rawMaxY - shrinkY;
                            const minZ = rawMinZ + shrinkZ;
                            const maxZ = rawMaxZ - backWallInset;

                            const innerCenterX = (minX + maxX) / 2;
                            const innerCenterY = (minY + maxY) / 2;
                            const innerCenterZ = (minZ + maxZ) / 2;

                            // Store tightened bounds for joystick clamping
                            roomBoundsForClamping = {
                                minX, maxX,
                                minY, maxY,
                                minZ, maxZ,
                                centerX: innerCenterX,
                                centerY: innerCenterY,
                                centerZ: innerCenterZ
                            };

                            roomRadius = Math.max(
                                maxX - minX,
                                maxY - minY,
                                Math.abs(maxZ - minZ)
                            ) * 0.5;

                            // Store for oscillation camera path
                            window.roomViewParams = {
                                centerX: innerCenterX,
                                centerY: innerCenterY,
                                centerZ: innerCenterZ,
                                radius: roomRadius * 1.2
                            };

                            console.log('Inner bounds:', JSON.stringify(roomBoundsForClamping));

                            // 4) Send wall dimensions back to Swift UI
                            if (window.webkit?.messageHandlers?.frontWallDimensions) {
                                window.webkit.messageHandlers.frontWallDimensions.postMessage({
                                    width: roomWidth,
                                    height: roomHeight
                                });
                            }

                            // 5) Start: right in front of FRONT WALL (the wall at far distance), looking up at it.
                            // Front wall = far wall = maxZ in scene. Camera just in front of it (maxZ + dist), looking up at it.
                            const frontWallZ = maxZ;
                            const distInFront = 0.025;
                            // Look straight at wall (eye level): target at same height as camera
                            const newCamPos = new THREE.Vector3(
                                innerCenterX,
                                innerCenterY,
                                frontWallZ + distInFront
                            );
                            const newTarget = new THREE.Vector3(
                                innerCenterX,
                                innerCenterY,
                                frontWallZ
                            );

                            const msg1 = '[StartPos] Camera close to curtain, looking straight — distInFront=' + distInFront + ' minZ=' + minZ.toFixed(3) + ' maxZ=' + maxZ.toFixed(3) + ' isPortrait=' + isPortrait;
                            const msg2 = '[StartPos] pos=(' + newCamPos.x.toFixed(3) + ',' + newCamPos.y.toFixed(3) + ',' + newCamPos.z.toFixed(3) + ') tgt=(' + newTarget.x.toFixed(3) + ',' + newTarget.y.toFixed(3) + ',' + newTarget.z.toFixed(3) + ')';
                            console.log(msg1);
                            console.log(msg2);
                            if (window.webkit?.messageHandlers?.jsLog) {
                                window.webkit.messageHandlers.jsLog.postMessage(msg1);
                                window.webkit.messageHandlers.jsLog.postMessage(msg2);
                            }

                            camera.position.copy(newCamPos);
                            controls.target.copy(newTarget);
                            controls.update();

                            // 6) Update recenter base
                            initialCameraPosition.copy(camera.position);
                            initialControlsTarget.copy(controls.target);

                            // 7) Setup base orbit parameters around this view
                            autoOrbitRadius = camera.position.distanceTo(controls.target);
                            autoOrbitBaseAngle = Math.atan2(
                                camera.position.x - controls.target.x,
                                camera.position.z - controls.target.z
                            );
                            console.log('Auto-orbit base radius:', autoOrbitRadius.toFixed(2),
                                        'baseAngle:', autoOrbitBaseAngle.toFixed(2));

                            // Report camera pose to Swift for debugging
                            if (window.webkit?.messageHandlers?.cameraPose) {
                                window.webkit.messageHandlers.cameraPose.postMessage({
                                    ex: camera.position.x,
                                    ey: camera.position.y,
                                    ez: camera.position.z,
                                    tx: controls.target.x,
                                    ty: controls.target.y,
                                    tz: controls.target.z
                                });
                            }

                            // Force render after camera positioning (critical when auto-orbit is off)
                            needsRender = true;

                            // Also schedule a few more renders to ensure scene is visible
                            setTimeout(() => { needsRender = true; }, 100);
                            setTimeout(() => { needsRender = true; }, 300);

                            // Send dimensions again after delays to ensure Swift receives them
                            // (in case first message was missed during loading)
                            function sendDimensionsToSwift() {
                                if (window.webkit?.messageHandlers?.frontWallDimensions) {
                                    window.webkit.messageHandlers.frontWallDimensions.postMessage({
                                        width: roomWidth,
                                        height: roomHeight
                                    });
                                    console.log('Sent dimensions to Swift:', roomWidth.toFixed(2), 'x', roomHeight.toFixed(2));
                                }
                            }
                            setTimeout(sendDimensionsToSwift, 500);
                            setTimeout(sendDimensionsToSwift, 1500);
                            setTimeout(sendDimensionsToSwift, 3000);

                            console.log('=== Camera positioned using Box3 bounds ===');
                        } catch (err) {
                            console.error('autoFrameRoom error:', err);
                        }
                    }
                    console.log('Scheduling autoFrameRoom in 500ms...');
                    setTimeout(autoFrameRoom, 500);

                    // Force immediate first render to show something while splat loads
                    needsRender = true;

                } catch (err) {
                    console.error('Failed to load splat:', err);
                }

                // Handle resize
                window.addEventListener('resize', () => {
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                });

                // Room bounds for camera clamping (will be set by autoFrameRoom)
                let roomBoundsForClamping = null;
                let roomRadius = null;

                // Joystick movement handler with boundary clamping
                window.moveCamera = function(dx, dy) {
                    // Stop auto orbit when user starts walking
                    autoOrbitEnabled = false;

                    const moveSpeed = 0.03;  // Increased for better walk feel

                    // Calculate new position
                    let newX = camera.position.x + dx * moveSpeed;
                    let newZ = camera.position.z + dy * moveSpeed;

                    // Clamp to room bounds; tiny margin at back wall so user can get right in front of it
                    if (roomBoundsForClamping) {
                        const marginSide = 0.05;
                        const marginBack = 0.02;
                        newX = Math.max(roomBoundsForClamping.minX + marginSide,
                               Math.min(roomBoundsForClamping.maxX - marginSide, newX));
                        newZ = Math.max(roomBoundsForClamping.minZ + marginSide,
                               Math.min(roomBoundsForClamping.maxZ - marginBack, newZ));
                    }

                    // Apply clamped movement
                    const actualDx = newX - camera.position.x;
                    const actualDz = newZ - camera.position.z;

                    camera.position.x = newX;
                    camera.position.z = newZ;
                    controls.target.x += actualDx;
                    controls.target.z += actualDz;
                    needsRender = true;  // Request render after camera move
                };

                // Move camera (and target) up/down in world Y — called from Swift "camera up" button
                window.moveCameraUp = function(dy) {
                    autoOrbitEnabled = false;
                    if (typeof dy !== 'number' || !isFinite(dy)) return;
                    camera.position.y += dy;
                    controls.target.y += dy;
                    if (!INFINITE_ZOOM && roomBoundsForClamping) {
                        const m = 0.05;
                        camera.position.y = Math.max(roomBoundsForClamping.minY + m, Math.min(roomBoundsForClamping.maxY - m, camera.position.y));
                        controls.target.y = Math.max(roomBoundsForClamping.minY + m, Math.min(roomBoundsForClamping.maxY - m, controls.target.y));
                    }
                    controls.update();
                    needsRender = true;
                };

                // Two-finger pan / pan pad: shift room in screen direction (camera and target move together)
                // Swift sends screen deltas: +X = right, +Y = down. Axes swapped so drag-up = room-up when view/camera is rotated.
                window.panCamera = function(deltaX, deltaY) {
                    if (typeof deltaX !== 'number' || typeof deltaY !== 'number' || !isFinite(deltaX) || !isFinite(deltaY)) return;
                    autoOrbitEnabled = false;
                    const panSpeed = 0.06;
                    const forward = new THREE.Vector3();
                    camera.getWorldDirection(forward);
                    const right = new THREE.Vector3().crossVectors(camera.up, forward).normalize();
                    const upNorm = camera.up.clone().normalize();
                    // Map screen dx -> world up, screen dy -> world right (fixes perpendicular pan when camera/view rotated)
                    const move = right.multiplyScalar(-deltaY * panSpeed).add(upNorm.multiplyScalar(deltaX * panSpeed));
                    camera.position.add(move);
                    controls.target.add(move);
                    if (!INFINITE_ZOOM && roomBoundsForClamping) {
                        const m = 0.05;
                        camera.position.x = Math.max(roomBoundsForClamping.minX + m, Math.min(roomBoundsForClamping.maxX - m, camera.position.x));
                        camera.position.y = Math.max(roomBoundsForClamping.minY + m, Math.min(roomBoundsForClamping.maxY - m, camera.position.y));
                        camera.position.z = Math.max(roomBoundsForClamping.minZ + m, Math.min(roomBoundsForClamping.maxZ - m, camera.position.z));
                        controls.target.x = Math.max(roomBoundsForClamping.minX + m, Math.min(roomBoundsForClamping.maxX - m, controls.target.x));
                        controls.target.y = Math.max(roomBoundsForClamping.minY + m, Math.min(roomBoundsForClamping.maxY - m, controls.target.y));
                        controls.target.z = Math.max(roomBoundsForClamping.minZ + m, Math.min(roomBoundsForClamping.maxZ - m, controls.target.z));
                    }
                    controls.update();
                    needsRender = true;
                };

                // Orbit rotation handler (for Swift gesture overlay)
                window.orbitCamera = function(deltaX, deltaY) {
                    // Stop auto orbit when user drags
                    autoOrbitEnabled = false;

                    // Rotate around target using spherical coordinates
                    const rotateSpeed = 0.012;  // Increased for faster response

                    // Get current spherical position relative to target
                    const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
                    const spherical = new THREE.Spherical().setFromVector3(offset);

                    // Apply rotation
                    spherical.theta -= deltaX * rotateSpeed;  // Horizontal rotation
                    spherical.phi += deltaY * rotateSpeed;    // Vertical rotation

                    // Clamp phi to avoid flipping
                    spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

                    // Convert back to cartesian
                    offset.setFromSpherical(spherical);
                    camera.position.copy(controls.target).add(offset);

                    controls.update();
                    needsRender = true;  // Request render after orbit
                };

                // Zoom handler (for Swift pinch gesture) — incremental scale from Swift
                // Pinch out (scale > 1) = zoom in (closer). Pinch in (scale < 1) = zoom out (further).
                window.zoomCamera = function(scale) {
                    autoOrbitEnabled = false;
                    if (typeof scale !== 'number' || scale <= 0 || !isFinite(scale)) return;
                    const zoomSensitivity = 4.0;

                    if (INFINITE_ZOOM) {
                        // Dolly: move camera and target along view direction so we can pass through curtain
                        const forward = new THREE.Vector3().subVectors(controls.target, camera.position).normalize();
                        const step = (scale - 1) * 0.22 * zoomSensitivity;  // scale>1 -> move forward, scale<1 -> move back
                        camera.position.addScaledVector(forward, step);
                        controls.target.addScaledVector(forward, step);
                    } else {
                        const amplifiedScale = 1 + (scale - 1) * zoomSensitivity;
                        const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
                        offset.multiplyScalar(1 / amplifiedScale);
                        let dist = offset.length();
                        if (dist < 0.01) offset.setLength(0.01);
                        if (dist > 50) offset.setLength(50);
                        camera.position.copy(controls.target).add(offset);
                    }

                    if (!INFINITE_ZOOM && roomBoundsForClamping) {
                        const marginSide = 0.05;
                        const marginBack = 0.02;
                        const minX = roomBoundsForClamping.minX + marginSide;
                        const maxX = roomBoundsForClamping.maxX - marginSide;
                        const minY = roomBoundsForClamping.minY + marginSide;
                        const maxY = roomBoundsForClamping.maxY - marginSide;
                        const minZ = roomBoundsForClamping.minZ + marginSide;
                        const maxZInside = roomBoundsForClamping.maxZ - marginBack;
                        camera.position.x = Math.max(minX, Math.min(maxX, camera.position.x));
                        camera.position.y = Math.max(minY, Math.min(maxY, camera.position.y));
                        // If camera is in front of the front wall (z > maxZ), don't clamp z from above — otherwise
                        // zoom-out would snap the camera through the wall and we'd see the back of the wall.
                        if (camera.position.z <= roomBoundsForClamping.maxZ) {
                            camera.position.z = Math.max(minZ, Math.min(maxZInside, camera.position.z));
                        } else {
                            camera.position.z = Math.max(minZ, camera.position.z);
                        }
                    }
                    controls.update();
                    needsRender = true;
                };

                // Animation loop with battery optimization
                // - Warm-up period: render continuously for first 5 seconds (SparkJS needs this)
                // - After warm-up: only renders when needed
                // - Throttles to 30fps when idle auto-orbit is running
                let loadNotified = false;
                let lastRenderTime = 0;
                const IDLE_FPS = 30;     // Lower FPS for auto-orbit (saves battery)
                const IDLE_FRAME_TIME = 1000 / IDLE_FPS;
                const WARMUP_DURATION = 5000;  // 5 seconds warm-up for SparkJS to fully load
                const animationStartTime = performance.now();

                // Request render on next frame (called when something changes)
                window.requestRender = function() {
                    needsRender = true;
                };

                function animate(currentTime) {
                    requestAnimationFrame(animate);

                    const dt = clock.getDelta();
                    let shouldRender = needsRender;
                    needsRender = false;

                    // Warm-up period: always render for first 5 seconds
                    // SparkJS progressively loads splat data and needs continuous rendering
                    const elapsed = performance.now() - animationStartTime;
                    const inWarmup = elapsed < WARMUP_DURATION;
                    if (inWarmup) {
                        shouldRender = true;
                    }

                    // Back-and-forth orbit when not interacting (uses base angle from autoFrameRoom)
                    if (autoOrbitEnabled && !window._userInteracting && autoOrbitRadius > 0.1) {
                        autoOrbitTime += dt;

                        const speed = 0.35;
                        const t = controls.target;

                        if (isPortrait) {
                            // Portrait: circular arc oscillation ±30°
                            const amplitude = Math.PI / 6;
                            const angle = autoOrbitBaseAngle + amplitude * Math.sin(autoOrbitTime * speed);

                            camera.position.x = t.x + autoOrbitRadius * Math.sin(angle);
                            camera.position.z = t.z + autoOrbitRadius * Math.cos(angle);
                        } else {
                            // Landscape: horizontal left-right sweep only
                            const sweepAmount = autoOrbitRadius * 0.3 * Math.sin(autoOrbitTime * speed);

                            camera.position.x = initialCameraPosition.x + sweepAmount;
                            camera.position.z = initialCameraPosition.z;
                        }

                        // Throttle auto-orbit to 30fps to save battery
                        if (currentTime - lastRenderTime >= IDLE_FRAME_TIME) {
                            shouldRender = true;
                        }
                    }

                    // Always render during user interaction (60fps)
                    if (window._userInteracting) {
                        shouldRender = true;
                    }

                    // Skip render if nothing changed (huge battery savings)
                    // But always render during warm-up period
                    if (!shouldRender) {
                        return;
                    }

                    lastRenderTime = currentTime;
                    controls.update();

                    // Use SparkRenderer's update method for optimized Gaussian rendering
                    spark.update({ scene });
                    renderer.render(scene, camera);

                    // Notify Swift when loaded
                    if (!loadNotified && splatMesh) {
                        loadNotified = true;
                        const totalTime = performance.now() - jsStartTime;
                        if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.splatLoaded) {
                            window.webkit.messageHandlers.splatLoaded.postMessage({ loaded: true, timeMs: totalTime });
                        }
                        console.log('Splat loaded in', totalTime, 'ms');
                    }
                }
                animate(0);
            </script>
        </body>
        </html>
        """
    }

    class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        var webView: WKWebView?
        var schemeHandler: LocalFileSchemeHandler?
        var lastInfiniteZoom: Bool?
        let onLoaded: () -> Void
        let onFrontWallDimensions: ((Double, Double) -> Void)?
        private var hasNotifiedLoaded = false
        private var loadStartTime: CFAbsoluteTime = 0

        private var joystickTimer: Timer?

        init(onLoaded: @escaping () -> Void, onFrontWallDimensions: ((Double, Double) -> Void)? = nil) {
            self.onLoaded = onLoaded
            self.onFrontWallDimensions = onFrontWallDimensions
            self.loadStartTime = CFAbsoluteTimeGetCurrent()
            logDebug("⏱️ [WebGL] Starting load...")
            super.init()

            // Listen for recenter notification
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(recenterCamera),
                name: NSNotification.Name("RecenterWebGLCamera"),
                object: nil
            )

            // Listen for joystick movement
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleJoystickMove(_:)),
                name: NSNotification.Name("WebGLJoystickMove"),
                object: nil
            )

            // Listen for orbit gesture
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleOrbitGesture(_:)),
                name: NSNotification.Name("WebGLOrbitGesture"),
                object: nil
            )

            // Listen for zoom gesture
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleZoomGesture(_:)),
                name: NSNotification.Name("WebGLZoomGesture"),
                object: nil
            )

            // Listen for gesture state (to pause/resume auto-orbit)
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleOrbitGestureState(_:)),
                name: NSNotification.Name("WebGLOrbitGestureState"),
                object: nil
            )

            // Listen for room scale (calibration from real furniture dimensions)
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleScaleRoom(_:)),
                name: NSNotification.Name("WebGLScaleRoom"),
                object: nil
            )

            // Listen for camera move up/down (from overlay buttons)
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleCameraMoveUp(_:)),
                name: NSNotification.Name("WebGLCameraMoveUp"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleCameraMoveDown(_:)),
                name: NSNotification.Name("WebGLCameraMoveDown"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleCameraMoveLeft(_:)),
                name: NSNotification.Name("WebGLCameraMoveLeft"),
                object: nil
            )
            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleCameraMoveRight(_:)),
                name: NSNotification.Name("WebGLCameraMoveRight"),
                object: nil
            )
        }

        deinit {
            joystickTimer?.invalidate()
            NotificationCenter.default.removeObserver(self)
        }

        @objc private func recenterCamera() {
            logDebug("🎯 [WebGL] Recentering camera")
            let js = "if (typeof recenterCamera === 'function') recenterCamera();"
            webView?.evaluateJavaScript(js) { _, error in
                if let error = error {
                    logDebug("❌ [WebGL] Recenter JS error: \(error)")
                }
            }
        }

        @objc private func handleJoystickMove(_ notification: Notification) {
            guard let userInfo = notification.userInfo,
                  let offset = userInfo["offset"] as? CGSize else { return }

            let dx = Float(offset.width)
            let dy = Float(-offset.height)  // Invert Y for intuitive control

            // Skip if no movement
            if abs(dx) < 0.01 && abs(dy) < 0.01 { return }

            // Call SparkJS moveCamera function
            let js = "if (typeof moveCamera === 'function') moveCamera(\(dx), \(dy));"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleOrbitGesture(_ notification: Notification) {
            guard let userInfo = notification.userInfo,
                  let deltaX = userInfo["deltaX"] as? CGFloat,
                  let deltaY = userInfo["deltaY"] as? CGFloat else { return }

            // OrbitGestureView now sends incremental values directly
            let js = "if (typeof orbitCamera === 'function') orbitCamera(\(deltaX), \(deltaY));"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleZoomGesture(_ notification: Notification) {
            guard let userInfo = notification.userInfo,
                  let scale = userInfo["scale"] as? CGFloat else { return }

            // OrbitGestureView sends incremental scale; zoom strength is controlled in JS (amplifiedScale multiplier)
            let js = "if (typeof zoomCamera === 'function') zoomCamera(\(scale));"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleOrbitGestureState(_ notification: Notification) {
            guard let userInfo = notification.userInfo,
                  let interacting = userInfo["interacting"] as? Bool else { return }

            let js = "if (typeof setUserInteracting === 'function') setUserInteracting(\(interacting ? "true" : "false"));"
            logDebug("🎮 [WebGL] setUserInteracting(\(interacting))")
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleCameraMoveUp(_ notification: Notification) {
            let step = 0.2
            let js = "if (typeof moveCameraUp === 'function') moveCameraUp(\(step));"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleCameraMoveDown(_ notification: Notification) {
            let step = -0.2
            let js = "if (typeof moveCameraUp === 'function') moveCameraUp(\(step));"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleCameraMoveLeft(_ notification: Notification) {
            let step: Float = -8
            let js = "if (typeof moveCamera === 'function') moveCamera(\(step), 0);"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleCameraMoveRight(_ notification: Notification) {
            let step: Float = 8
            let js = "if (typeof moveCamera === 'function') moveCamera(\(step), 0);"
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        @objc private func handleScaleRoom(_ notification: Notification) {
            guard let userInfo = notification.userInfo,
                  let factor = userInfo["factor"] as? Double else { return }

            let js = "if (typeof scaleRoom === 'function') scaleRoom(\(factor));"
            logDebug("📐 [WebGL] scaleRoom(\(factor))")
            webView?.evaluateJavaScript(js, completionHandler: nil)
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            let elapsed = (CFAbsoluteTimeGetCurrent() - loadStartTime) * 1000
            logDebug("⏱️ [WebGL] HTML page loaded: \(String(format: "%.0f", elapsed))ms")

            DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
                if !self.hasNotifiedLoaded {
                    self.hasNotifiedLoaded = true
                    let total = (CFAbsoluteTimeGetCurrent() - self.loadStartTime) * 1000
                    logDebug("⏱️ [WebGL] Total load (timeout): \(String(format: "%.0f", total))ms")
                    self.onLoaded()
                }
            }
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            logDebug("❌ [WebGL] Navigation failed: \(error)")
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            logDebug("❌ [WebGL] Provisional navigation failed: \(error)")
        }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            if message.name == "splatLoaded" {
                if let body = message.body as? [String: Any], body["loaded"] != nil {
                    let elapsed = (CFAbsoluteTimeGetCurrent() - loadStartTime) * 1000
                    logDebug("⏱️ [WebGL] Splat rendered: \(String(format: "%.0f", elapsed))ms total")

                    if !hasNotifiedLoaded {
                        hasNotifiedLoaded = true
                        DispatchQueue.main.async {
                            self.onLoaded()
                        }
                    }
                }
            } else if message.name == "frontWallDimensions" {
                if let body = message.body as? [String: Any],
                   let w = body["width"] as? Double,
                   let h = body["height"] as? Double {
                    logDebug("📏 [WebGL] Front wall from JS: \(String(format: "%.2f", w)) × \(String(format: "%.2f", h))")
                    DispatchQueue.main.async {
                        self.onFrontWallDimensions?(w, h)
                    }
                }
            } else if message.name == "cameraPose" {
                if let body = message.body as? [String: Any],
                   let ex = body["ex"] as? Double,
                   let ey = body["ey"] as? Double,
                   let ez = body["ez"] as? Double,
                   let tx = body["tx"] as? Double,
                   let ty = body["ty"] as? Double,
                   let tz = body["tz"] as? Double {
                    logDebug("🎥 [WebGL] Camera pose JS -> Swift: eye=(\(String(format: "%.2f", ex)), \(String(format: "%.2f", ey)), \(String(format: "%.2f", ez))) target=(\(String(format: "%.2f", tx)), \(String(format: "%.2f", ty)), \(String(format: "%.2f", tz)))")
                }
            } else if message.name == "jsLog", let text = message.body as? String {
                logDebug("📜 [JS] \(text)")
            }
        }
    }
}

// MARK: - Share Sheet

/// UIActivityViewController wrapper for sharing files
struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: activityItems,
            applicationActivities: nil
        )
        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - Preview

#Preview {
    NavigationStack {
        // Preview with a sample URL (won't actually load)
        SharpRoomView(plyURL: URL(fileURLWithPath: "/sample.ply"))
    }
}

