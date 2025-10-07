import Foundation
import UIKit
import SceneKit
import SwiftUI

// MARK: - Diagnostics & Logging

private enum DollLog {
    static var enabled: Bool = true  // flip to false to silence logs in release builds

    @inline(__always)
    static func p(_ msg: String) {
        guard enabled else { return }
        print("[DOLLHOUSE] \(msg)")
    }

    @inline(__always)
    static func bytesString(_ count: Int64?) -> String {
        guard let c = count else { return "n/a" }
        if c < 1024 { return "\(c) B" }
        let kb = Double(c) / 1024.0
        if kb < 1024 { return String(format: "%.1f KB", kb) }
        let mb = kb / 1024.0
        return String(format: "%.2f MB", mb)
    }

    @inline(__always)
    static func deg(_ radians: Float) -> String {
        String(format: "%.1f°", radians * 180.0 / .pi)
    }
}

private func fileSize(at url: URL) -> Int64? {
    (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
}

private func timeMark() -> CFAbsoluteTime { CFAbsoluteTimeGetCurrent() }

private func timeElapsedString(since t0: CFAbsoluteTime) -> String {
    let dt = CFAbsoluteTimeGetCurrent() - t0
    return String(format: "%.3f s", dt)
}

// MARK: - 2.5D Dollhouse Creator (SceneKit build → SceneKit USDZ export)

final class DollhouseCreator {

    // Delete any previous dollhouse_* files in Documents
    static func clearAllDollhouseFiles() {
        guard let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return
        }
        do {
            let files = try FileManager.default.contentsOfDirectory(at: documentsPath, includingPropertiesForKeys: nil)
            for file in files where file.lastPathComponent.contains("dollhouse_") {
                try? FileManager.default.removeItem(at: file)
                DollLog.p("🗑️ Deleted existing: \(file.lastPathComponent)")
            }
        } catch {
            DollLog.p("⚠️ Error clearing: \(error)")
        }
    }

    /// Build a USDZ from exactly 6 photos: [front, right, back, left, floor, ceiling]
    /// Returns the final USDZ URL in Documents, or nil on failure.
    static func createDollhouseFile(from photos: [UIImage], fileName: String) -> URL? {
        DollLog.p("========== DOLLHOUSE CREATION START ==========")
        let t0 = timeMark()
        clearAllDollhouseFiles()

        guard photos.count >= 6 else {
            DollLog.p("❌ ERROR: Need 6 photos (got \(photos.count))")
            return nil
        }
        guard let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            DollLog.p("❌ ERROR: No documents directory")
            return nil
        }

        // Temp working dir for textures & temp usdz
        let tempDir = documentsPath.appendingPathComponent("temp_textures_\(UUID().uuidString)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
            DollLog.p("📁 Created temp dir: \(tempDir.lastPathComponent)")
        } catch {
            DollLog.p("❌ Failed to create temp dir: \(error)")
            return nil
        }

        // Save 6 textures (PNG) to temp directory in fixed order
        var imagePaths: [URL] = []
        for (index, photo) in photos.prefix(6).enumerated() {
            let img = normalizeOrientation(photo) ?? photo
            DollLog.p(String(format: "🖼️  Photo[%d] size: %.0fx%.0f px, scale: %.2f, orientation fixed: %@",
                             index, img.size.width, img.size.height, img.scale, (photo.imageOrientation == .up ? "no" : "yes")))

            let imagePath = tempDir.appendingPathComponent("texture_\(index).png")
            if let png = img.pngData() {
                do {
                    try png.write(to: imagePath, options: .atomic)
                    let sz = fileSize(at: imagePath)
                    DollLog.p("💾 Saved texture[\(index)]: \(imagePath.lastPathComponent) (\(DollLog.bytesString(sz)))")
                    if !FileManager.default.fileExists(atPath: imagePath.path) {
                        DollLog.p("❌ File missing right after write: \(imagePath.path)")
                    }
                    imagePaths.append(imagePath)
                } catch {
                    DollLog.p("❌ Failed to write texture[\(index)]: \(error)")
                }
            } else {
                DollLog.p("❌ pngData() nil for texture[\(index)]")
            }
        }

        guard imagePaths.count == 6 else {
            DollLog.p("❌ ERROR: Failed to save all 6 textures (got \(imagePaths.count))")
            try? FileManager.default.removeItem(at: tempDir)
            return nil
        }

        // Room dimensions (meters)
        let roomWidth: Float  = 4
        let roomHeight: Float = 3
        let roomDepth: Float  = 4
        DollLog.p("📐 Room dimensions: W:\(roomWidth) H:\(roomHeight) D:\(roomDepth)")

        // Build SceneKit scene of inward-facing planes (correct rotations)
        let scene = SCNScene()

        // FRONT (z = -depth/2), inward normal +Z → no rotation needed
        DollLog.p("🟦 Creating FRONT wall…")
        let frontWall = createWall(width: CGFloat(roomWidth), height: CGFloat(roomHeight), imagePath: imagePaths[0])
        frontWall.position = SCNVector3(0, 0, -roomDepth * 0.5)
        frontWall.eulerAngles = SCNVector3(0, 0, 0)
        frontWall.name = "FrontWall"
        scene.rootNode.addChildNode(frontWall)
        logNode("FrontWall", node: frontWall)
        
        // RIGHT (x = +width/2) — inward normal should be -X  → yaw = -π/2
        

        // LEFT (x = -width/2) — inward normal should be +X  → yaw = +π/2
        


        // RIGHT (x = +width/2), inward normal −X → yaw +π/2
        DollLog.p("🟩 Creating RIGHT wall…")
        let rightWall = createWall(width: CGFloat(roomDepth), height: CGFloat(roomHeight), imagePath: imagePaths[1])
        rightWall.position = SCNVector3(roomWidth * 0.5, 0, 0)
        rightWall.eulerAngles = SCNVector3(0, -Float.pi / 2, 0)
        rightWall.name = "RightWall"
        scene.rootNode.addChildNode(rightWall)
        logNode("RightWall", node: rightWall)

        // BACK (z = +depth/2), inward normal −Z → yaw π
        DollLog.p("🟨 Creating BACK wall…")
        let backWall = createWall(width: CGFloat(roomWidth), height: CGFloat(roomHeight), imagePath: imagePaths[2])
        backWall.position = SCNVector3(0, 0, roomDepth * 0.5)
        backWall.eulerAngles = SCNVector3(0, Float.pi, 0)
        backWall.name = "BackWall"
        scene.rootNode.addChildNode(backWall)
        logNode("BackWall", node: backWall)

        // LEFT (x = −width/2), inward normal +X → yaw −π/2
        DollLog.p("🟧 Creating LEFT wall…")
        let leftWall = createWall(width: CGFloat(roomDepth), height: CGFloat(roomHeight), imagePath: imagePaths[3])
        leftWall.position = SCNVector3(-roomWidth * 0.5, 0, 0)
        leftWall.eulerAngles = SCNVector3(0, +Float.pi / 2, 0)
        leftWall.name = "LeftWall"
        scene.rootNode.addChildNode(leftWall)
        logNode("LeftWall", node: leftWall)

        // FLOOR (y = −height/2), inward normal +Y → pitch −π/2
        DollLog.p("🟫 Creating FLOOR…")
        let floor = createFloor(width: CGFloat(roomWidth), depth: CGFloat(roomDepth), imagePath: imagePaths[4])
        floor.position = SCNVector3(0, -roomHeight * 0.5, 0)
        floor.name = "Floor"
        scene.rootNode.addChildNode(floor)
        logNode("Floor", node: floor)

        // CEILING (y = +height/2), inward normal −Y → pitch +π/2
        DollLog.p("⬜ Creating CEILING…")
        let ceiling = createCeiling(width: CGFloat(roomWidth), depth: CGFloat(roomDepth), imagePath: imagePaths[5])
        ceiling.position = SCNVector3(0, roomHeight * 0.5, 0)
        ceiling.name = "Ceiling"
        scene.rootNode.addChildNode(ceiling)
        logNode("Ceiling", node: ceiling)

        // Camera (optional)
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.fieldOfView = 60
        cameraNode.position = SCNVector3(-3, 1, 4)
        cameraNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(cameraNode)

        // Export: SceneKit -> USDZ directly (iOS 12+), with destination URL & embed textures
        let tempUsdzPath = tempDir.appendingPathComponent(fileName)
        DollLog.p("📦 Exporting USDZ to temp (SCNScene.write) with destination & embed options…")
        let tExport = timeMark()
        if #available(iOS 12.0, *) {

            // Use String keys to avoid SDK symbols that may be missing.
            // Keys:
            //  - "SCNSceneExportDestinationURL" : NSURL
            //  - "SCNSceneExportEmbedTextures"  : Bool
            let exportOptions: [String: Any] = [
                "SCNSceneExportDestinationURL": tempDir as NSURL, // where to localize textures
                "SCNSceneExportEmbedTextures": true               // embed textures inside USDZ
            ]

            let ok = scene.write(
                to: tempUsdzPath,
                options: exportOptions,
                delegate: nil,
                progressHandler: { (progress, error, stop) in
                    if let err = error {
                        DollLog.p("⚠️ Export progress error: \(err)")
                        stop.pointee = true
                    }
                    // DollLog.p(String(format: "… export: %.0f%%", progress * 100))
                }
            )
            DollLog.p("⏱️ Export time: \(timeElapsedString(since: tExport))")
            if !ok {
                DollLog.p("❌ SceneKit failed to write USDZ")
                try? FileManager.default.removeItem(at: tempDir)
                return nil
            }
            DollLog.p("✅ USDZ created at temp: \(tempUsdzPath.lastPathComponent) (\(DollLog.bytesString(fileSize(at: tempUsdzPath))))")

            // Move to final location
            let finalURL = documentsPath.appendingPathComponent(fileName)
            if FileManager.default.fileExists(atPath: finalURL.path) {
                DollLog.p("🔁 Overwriting existing: \(finalURL.lastPathComponent)")
                try? FileManager.default.removeItem(at: finalURL)
            }
            do {
                try FileManager.default.moveItem(at: tempUsdzPath, to: finalURL)
                DollLog.p("✅ Moved USDZ → \(finalURL.lastPathComponent) (\(DollLog.bytesString(fileSize(at: finalURL))))")
                try? FileManager.default.removeItem(at: tempDir)
                DollLog.p("🗑️ Cleaned up temp dir")
                DollLog.p("✅ DONE in \(timeElapsedString(since: t0))")
                return finalURL
            } catch {
                DollLog.p("❌ Export/move error: \(error)")
                try? FileManager.default.removeItem(at: tempDir)
                return nil
            }
        } else {
            DollLog.p("❌ USDZ export requires iOS 12+")
            try? FileManager.default.removeItem(at: tempDir)
            return nil
        }
    }
}

// MARK: - SceneKit materials & nodes

private func makeMaterial(from imagePath: URL) -> SCNMaterial {
    let m = SCNMaterial()

    // Force-load bitmap from disk (more reliable than passing URL directly)
    if let img = UIImage(contentsOfFile: imagePath.path) {
        m.diffuse.contents = img
        DollLog.p(String(format: "🧩 Material: loaded image '%@' (%.0fx%.0f px)",
                         imagePath.lastPathComponent, img.size.width, img.size.height))
    } else {
        DollLog.p("❌ Material: failed to load image at \(imagePath.path); using fallback color")
        m.diffuse.contents = UIColor.darkGray
    }

    // Useful texture params for safety
    m.diffuse.mipFilter = .linear
    m.diffuse.wrapS = .clamp
    m.diffuse.wrapT = .clamp

    // Double-sided in case a wall faces away for any reason
    m.isDoubleSided = true
    m.lightingModel = .constant

    // If your photos appear upside down, uncomment to flip vertically:
    // m.diffuse.contentsTransform = SCNMatrix4MakeScale(1, -1, 1)

    return m
}

private func createWall(width: CGFloat, height: CGFloat, imagePath: URL) -> SCNNode {
    let plane = SCNPlane(width: width, height: height)
    plane.firstMaterial = makeMaterial(from: imagePath)
    let node = SCNNode(geometry: plane)
    DollLog.p(String(format: "🧱 Wall plane W:%.2f H:%.2f", width, height))
    return node
}

private func createFloor(width: CGFloat, depth: CGFloat, imagePath: URL) -> SCNNode {
    let plane = SCNPlane(width: width, height: depth)
    plane.firstMaterial = makeMaterial(from: imagePath)
    let node = SCNNode(geometry: plane)
    node.eulerAngles = SCNVector3(-Float.pi / 2, 0, 0) // lay it flat
    DollLog.p(String(format: "🪵 Floor plane W:%.2f D:%.2f", width, depth))
    return node
}

private func createCeiling(width: CGFloat, depth: CGFloat, imagePath: URL) -> SCNNode {
    let plane = SCNPlane(width: width, height: depth)
    plane.firstMaterial = makeMaterial(from: imagePath)
    let node = SCNNode(geometry: plane)
    node.eulerAngles = SCNVector3(+Float.pi / 2, 0, 0) // upside flat
    DollLog.p(String(format: "🧱 Ceiling plane W:%.2f D:%.2f", width, depth))
    return node
}

private func logNode(_ label: String, node: SCNNode) {
    DollLog.p(String(format: "📍 %@ pos(%.2f, %.2f, %.2f) euler(yaw,pitch,roll)=(%@, %@, %@)",
                     label,
                     node.position.x, node.position.y, node.position.z,
                     DollLog.deg(node.eulerAngles.y),
                     DollLog.deg(node.eulerAngles.x),
                     DollLog.deg(node.eulerAngles.z)))
}

// MARK: - Utilities

/// Normalize UIImage EXIF/orientation into upright baked image
private func normalizeOrientation(_ image: UIImage) -> UIImage? {
    if image.imageOrientation == .up {
        return image
    }
    let size = image.size
    UIGraphicsBeginImageContextWithOptions(size, true, image.scale)
    image.draw(in: CGRect(origin: .zero, size: size))
    let normalized = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    return normalized
}

// MARK: - Scanner View (unchanged UI; added logging in process flow)

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
        "FRONT WALL - Stand at back, capture front wall",
        "RIGHT WALL - Turn right, capture right wall",
        "BACK WALL - Turn around, capture back wall",
        "LEFT WALL - Turn left, capture left wall",
        "FLOOR - Point camera down at floor",
        "CEILING - Point camera up at ceiling"
    ]

    var body: some View {
        NavigationView {
            ZStack {
                LinearGradient(gradient: Gradient(colors: [.black, .blue.opacity(0.3)]), startPoint: .top, endPoint: .bottom)
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
                    Button("Cancel") { isPresented = false }
                        .foregroundColor(.white)
                }
            }
            .preferredColorScheme(.dark)
        }
        .sheet(isPresented: $showingCamera, onDismiss: {
            if let image = currentImage {
                DollLog.p(String(format: "📸 Captured image size: %.0fx%.0f px", image.size.width, image.size.height))
                capturedPhotos.append(image)
                currentImage = nil
                currentPhotoIndex += 1
                DollLog.p("➡️ Progress: \(currentPhotoIndex)/6 photos")
                if currentPhotoIndex >= 6 { processPhotos() }
            } else {
                DollLog.p("⚠️ Camera dismissed without image")
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

            Text(photoInstructions[currentPhotoIndex])
                .font(.title3)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
                .padding()
            .background(Color.black.opacity(0.5))
            .cornerRadius(16)

            Spacer()

            Button(action: { showingCamera = true }) {
                HStack {
                    Image(systemName: "camera.fill")
                    Text("Take Photo")
                }
                .font(.headline)
                .foregroundColor(.white)
                .padding(.horizontal, 40)
                .padding(.vertical, 16)
                .background(LinearGradient(gradient: Gradient(colors: [.blue, .purple]), startPoint: .leading, endPoint: .trailing))
                .cornerRadius(25)
            }
            .padding(.bottom, 50)

            if !capturedPhotos.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(capturedPhotos.enumerated()), id: \.offset) { _, photo in
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
            Button(action: { isPresented = false }) {
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
                DollLog.p("🔁 Resetting scanner after error: \(message)")
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
        DollLog.p("🧮 Starting processing of 6 photos…")
        DispatchQueue.global(qos: .userInitiated).async {
            let fileName = "dollhouse_\(UInt64(Date().timeIntervalSince1970)).usdz"
            let url = DollhouseCreator.createDollhouseFile(from: capturedPhotos, fileName: fileName)
            DispatchQueue.main.async {
                generatedModelURL = url
                if let url = url {
                    DollLog.p("🎉 Generated USDZ at: \(url.lastPathComponent)")
                    saveToCollection(usdzURL: url, fileName: fileName)
                } else {
                    DollLog.p("❌ Failed to create USDZ")
                    errorMessage = "Failed to create room"
                }
                isProcessing = false
                scanComplete = true
            }
        }
    }

    private func saveToCollection(usdzURL: URL, fileName: String) {
        guard FileManager.default.fileExists(atPath: usdzURL.path) else {
            DollLog.p("⚠️ saveToCollection: file not found at \(usdzURL.path)")
            return
        }
        DollLog.p("📣 Posting didAddModelNotification for \(fileName)")
        NotificationCenter.default.post(name: USDZModelManager.didAddModelNotification, object: nil, userInfo: ["fileName": fileName])
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            DollLog.p("🔄 Refreshing model manager…")
            modelManager.refreshModels()
        }
    }

    private func resetScanner() {
        DollLog.p("🧹 Reset scanner state")
        capturedPhotos = []
        currentPhotoIndex = 0
        scanComplete = false
        generatedModelURL = nil
        errorMessage = nil
    }
}
