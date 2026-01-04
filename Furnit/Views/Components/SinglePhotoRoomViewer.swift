import SwiftUI
import SceneKit
import Accelerate
import CoreML
import CoreVideo

// MARK: - Room Boundary Detection View with DRAGGABLE boundaries
struct RoomBoundaryDetectionView: View {
    let originalImage: UIImage
    @Binding var savedBoundaries: RoomStructure?

    @Environment(\.dismiss) var dismiss
    
    // Boundary positions (as percentages of image dimensions)
    @State private var floorY: CGFloat = 0.85
    @State private var ceilingY: CGFloat = 0.15
    @State private var leftX: CGFloat = 0.12
    @State private var rightX: CGFloat = 0.88
    @State private var vanishingX: CGFloat = 0.5
    @State private var vanishingY: CGFloat = 0.45
    
    // Custom magenta color
    private let magentaColor = Color(red: 1.0, green: 0.0, blue: 1.0)
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Interactive adjustment view - image is already fixed
                GeometryReader { geometry in
                    ZStack {
                        // Background image
                        Image(uiImage: originalImage)
                            .resizable()
                            .scaledToFit()
                            .frame(width: geometry.size.width)
                            .background(Color.red) // ✅ Debug: should not see red if image loads

                        // Overlay with draggable boundaries
                        BoundaryLinesCanvas(
                            imageSize: originalImage.size,
                            floorY: floorY,
                            ceilingY: ceilingY,
                            leftX: leftX,
                            rightX: rightX,
                            vanishingX: vanishingX,
                            vanishingY: vanishingY
                        )
                        .frame(width: geometry.size.width)
                        
                        // Draggable handles
                        DraggableHandlesOverlay(
                            geometry: geometry,
                            imageSize: originalImage.size,
                            floorY: $floorY,
                            ceilingY: $ceilingY,
                            leftX: $leftX,
                            rightX: $rightX,
                            vanishingX: $vanishingX,
                            vanishingY: $vanishingY,
                            magentaColor: magentaColor
                        )
                    }
                }
                    
                    // Adjustment instructions
                    VStack(spacing: 12) {
                        Text(L10n.Boundary.instructions)
                            .font(.headline)
                            .padding(.top, 8)

                        HStack(spacing: 16) {
                            Label(L10n.Boundary.floor, systemImage: "arrow.down")
                                .foregroundColor(.green)
                                .font(.caption)
                            Label(L10n.Boundary.ceiling, systemImage: "arrow.up")
                                .foregroundColor(.cyan)
                                .font(.caption)
                            Label(L10n.Boundary.walls, systemImage: "arrow.left.and.right")
                                .foregroundColor(.red)
                                .font(.caption)
                            Label(L10n.Boundary.vanish, systemImage: "scope")
                                .foregroundColor(magentaColor)
                                .font(.caption)
                        }
                        .padding(.horizontal)

                        HStack(spacing: 20) {
                            Button(L10n.Common.reset) {
                                withAnimation {
                                    floorY = 0.85
                                    ceilingY = 0.15
                                    leftX = 0.12
                                    rightX = 0.88
                                    vanishingX = 0.5
                                    vanishingY = 0.45
                                }
                            }
                            .buttonStyle(.bordered)

                            Button(L10n.Common.done) {
                                var boundaries = RoomStructure()
                                boundaries.floorY = floorY
                                boundaries.ceilingY = ceilingY
                                boundaries.leftX = leftX
                                boundaries.rightX = rightX
                                boundaries.vanishingX = vanishingX
                                boundaries.vanishingY = vanishingY

                                print("✅✅✅ [RoomBoundary] DONE BUTTON PRESSED! Setting savedBoundaries ✅✅✅")
                                savedBoundaries = boundaries
                                print("   savedBoundaries is now: \(String(describing: savedBoundaries))")
                                logDebug("✅ Saved adjusted boundaries:")
                                logDebug("   Floor: \(floorY), Ceiling: \(ceilingY)")
                                logDebug("   Left: \(leftX), Right: \(rightX)")
                                logDebug("   VP: (\(vanishingX), \(vanishingY))")

                                dismiss()
                            }
                            .buttonStyle(.borderedProminent)
                        }
                        .padding(.bottom, 8)
                    }
                    .background(.ultraThinMaterial)
            }
            .navigationTitle(L10n.Boundary.title)
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                logDebug("🖼️ [BoundaryView] View appeared with image size: \(originalImage.size)")
                logDebug("   Image scale: \(originalImage.scale), orientation: \(originalImage.imageOrientation.rawValue)")
            }
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(L10n.Common.back) {
                        // Dismiss without saving changes
                        dismiss()
                    }
                }
            }
        }
    }
    
    func drawBoundariesOnImage() async -> UIImage {
        // Use originalImage (already orientation-fixed)
        let sourceImage = originalImage

        // ✅ OPTIMIZATION: Downscale large images to prevent memory crashes
        // Using vImage from Accelerate framework for GPU/NEON acceleration
        let maxDimension: CGFloat = 1600  // Max 1600px - balances quality & memory
        let originalWidth = sourceImage.size.width
        let originalHeight = sourceImage.size.height
        let scaleFactor = min(maxDimension / max(originalWidth, originalHeight), 1.0)

        let workingImage: UIImage
        if scaleFactor < 1.0 {
            logDebug("🚀 [BoundaryView] Downscaling \(Int(originalWidth))x\(Int(originalHeight)) → \(Int(originalWidth * scaleFactor))x\(Int(originalHeight * scaleFactor))")
            workingImage = downscaleWithAccelerate(sourceImage, scale: scaleFactor) ?? sourceImage
        } else {
            workingImage = sourceImage
        }

        let width = workingImage.size.width
        let height = workingImage.size.height

        // Use autoreleasepool to free memory immediately after rendering
        return autoreleasepool {
            let renderer = UIGraphicsImageRenderer(size: workingImage.size)
            return renderer.image { context in
                // Draw working image (downscaled if needed)
                workingImage.draw(at: .zero)
                
                let cgContext = context.cgContext
                
                // Draw floor boundary in GREEN
                cgContext.setStrokeColor(UIColor.green.cgColor)
                cgContext.setLineWidth(15.0)
                let floorYPos = floorY * height
                cgContext.move(to: CGPoint(x: 0, y: floorYPos))
                cgContext.addLine(to: CGPoint(x: width, y: floorYPos))
                cgContext.strokePath()
                
                let floorLabel = "FLOOR"
                let floorAttrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: 60),
                    .foregroundColor: UIColor.green,
                    .strokeColor: UIColor.black,
                    .strokeWidth: -4.0
                ]
                floorLabel.draw(at: CGPoint(x: 50, y: floorYPos - 80), withAttributes: floorAttrs)
                
                // Draw ceiling boundary in CYAN
                cgContext.setStrokeColor(UIColor.cyan.cgColor)
                cgContext.setLineWidth(15.0)
                let ceilingYPos = ceilingY * height
                cgContext.move(to: CGPoint(x: 0, y: ceilingYPos))
                cgContext.addLine(to: CGPoint(x: width, y: ceilingYPos))
                cgContext.strokePath()
                
                let ceilingLabel = "CEILING"
                let ceilingAttrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: 60),
                    .foregroundColor: UIColor.cyan,
                    .strokeColor: UIColor.black,
                    .strokeWidth: -4.0
                ]
                ceilingLabel.draw(at: CGPoint(x: 50, y: ceilingYPos + 30), withAttributes: ceilingAttrs)
                
                // Draw left wall in RED
                cgContext.setStrokeColor(UIColor.red.cgColor)
                cgContext.setLineWidth(12.0)
                let leftXPos = leftX * width
                cgContext.move(to: CGPoint(x: leftXPos, y: 0))
                cgContext.addLine(to: CGPoint(x: leftXPos, y: height))
                cgContext.strokePath()
                
                let leftLabel = "LEFT"
                let leftAttrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: 50),
                    .foregroundColor: UIColor.red,
                    .strokeColor: UIColor.white,
                    .strokeWidth: -3.0
                ]
                leftLabel.draw(at: CGPoint(x: leftXPos + 30, y: height / 2), withAttributes: leftAttrs)
                
                // Draw right wall in YELLOW
                cgContext.setStrokeColor(UIColor.yellow.cgColor)
                cgContext.setLineWidth(12.0)
                let rightXPos = rightX * width
                cgContext.move(to: CGPoint(x: rightXPos, y: 0))
                cgContext.addLine(to: CGPoint(x: rightXPos, y: height))
                cgContext.strokePath()
                
                let rightLabel = "RIGHT"
                let rightAttrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: 50),
                    .foregroundColor: UIColor.yellow,
                    .strokeColor: UIColor.black,
                    .strokeWidth: -3.0
                ]
                rightLabel.draw(at: CGPoint(x: rightXPos - 150, y: height / 2), withAttributes: rightAttrs)
                
                // Draw vanishing point in MAGENTA
                cgContext.setFillColor(UIColor.magenta.cgColor)
                let vpX = vanishingX * width
                let vpY = vanishingY * height
                let vpRadius: CGFloat = 40
                let vpRect = CGRect(x: vpX - vpRadius, y: vpY - vpRadius, width: vpRadius * 2, height: vpRadius * 2)
                cgContext.fillEllipse(in: vpRect)
                
                // Crosshair
                cgContext.setStrokeColor(UIColor.white.cgColor)
                cgContext.setLineWidth(5.0)
                cgContext.move(to: CGPoint(x: vpX - 80, y: vpY))
                cgContext.addLine(to: CGPoint(x: vpX + 80, y: vpY))
                cgContext.move(to: CGPoint(x: vpX, y: vpY - 80))
                cgContext.addLine(to: CGPoint(x: vpX, y: vpY + 80))
                cgContext.strokePath()
                
                let vpLabel = "VP"
                let vpAttrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: 40),
                    .foregroundColor: UIColor.magenta,
                    .strokeColor: UIColor.white,
                    .strokeWidth: -3.0
                ]
                vpLabel.draw(at: CGPoint(x: vpX - 30, y: vpY - 100), withAttributes: vpAttrs)
            }
        } // autoreleasepool
    }

    // ✅ vImage-accelerated downscaling (uses GPU/NEON SIMD for speed)
    private func downscaleWithAccelerate(_ image: UIImage, scale: CGFloat) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        let newWidth = Int(CGFloat(cgImage.width) * scale)
        let newHeight = Int(CGFloat(cgImage.height) * scale)

        // Use vImage for hardware-accelerated scaling
        var format = vImage_CGImageFormat(
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            colorSpace: nil,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            version: 0,
            decode: nil,
            renderingIntent: .defaultIntent
        )

        var sourceBuffer = vImage_Buffer()
        var error = vImageBuffer_InitWithCGImage(&sourceBuffer, &format, nil, cgImage, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else {
            logDebug("❌ vImage source buffer init failed: \(error)")
            return nil
        }
        defer { free(sourceBuffer.data) }

        var destBuffer = vImage_Buffer()
        error = vImageBuffer_Init(&destBuffer, vImagePixelCount(newHeight), vImagePixelCount(newWidth), 32, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else {
            logDebug("❌ vImage dest buffer init failed: \(error)")
            return nil
        }
        defer { free(destBuffer.data) }

        // High-quality Lanczos scaling
        error = vImageScale_ARGB8888(&sourceBuffer, &destBuffer, nil, vImage_Flags(kvImageHighQualityResampling))
        guard error == kvImageNoError else {
            logDebug("❌ vImage scale failed: \(error)")
            return nil
        }

        guard let scaledCGImage = vImageCreateCGImageFromBuffer(&destBuffer, &format, nil, nil, vImage_Flags(kvImageNoFlags), &error)?.takeRetainedValue() else {
            logDebug("❌ vImage CGImage creation failed: \(error)")
            return nil
        }

        logDebug("✅ [vImage] Downscaled to \(newWidth)x\(newHeight)")
        return UIImage(cgImage: scaledCGImage)
    }
}

// MARK: - Boundary Lines Canvas (fixed: no top-level `let` in ViewBuilder)
struct BoundaryLinesCanvas: View {
    let imageSize: CGSize
    let floorY: CGFloat
    let ceilingY: CGFloat
    let leftX: CGFloat
    let rightX: CGFloat
    let vanishingX: CGFloat
    let vanishingY: CGFloat

    var body: some View {
        GeometryReader { geometry in
            BoundaryLinesCanvasInner(
                calc: calculateImageBounds(size: geometry.size),
                floorY: floorY,
                ceilingY: ceilingY,
                leftX: leftX,
                rightX: rightX,
                vanishingX: vanishingX,
                vanishingY: vanishingY
            )
        }
    }

    private func calculateImageBounds(size: CGSize) -> (imageWidth: CGFloat, imageHeight: CGFloat, offsetX: CGFloat, offsetY: CGFloat) {
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = size.width / size.height

        var imageWidth: CGFloat
        var imageHeight: CGFloat
        var offsetX: CGFloat = 0
        var offsetY: CGFloat = 0

        if imageAspect > viewAspect {
            imageWidth = size.width
            imageHeight = size.width / imageAspect
            offsetY = (size.height - imageHeight) / 2
        } else {
            imageHeight = size.height
            imageWidth = size.height * imageAspect
            offsetX = (size.width - imageWidth) / 2
        }

        return (imageWidth, imageHeight, offsetX, offsetY)
    }
}

private struct BoundaryLinesCanvasInner: View {
    let calc: (imageWidth: CGFloat, imageHeight: CGFloat, offsetX: CGFloat, offsetY: CGFloat)
    let floorY: CGFloat
    let ceilingY: CGFloat
    let leftX: CGFloat
    let rightX: CGFloat
    let vanishingX: CGFloat
    let vanishingY: CGFloat

    var body: some View {
        ZStack {
            // Floor line (GREEN)
            Path { path in
                let y = calc.offsetY + floorY * calc.imageHeight
                path.move(to: CGPoint(x: calc.offsetX, y: y))
                path.addLine(to: CGPoint(x: calc.offsetX + calc.imageWidth, y: y))
            }
            .stroke(Color.green, lineWidth: 8)

            // Ceiling line (CYAN)
            Path { path in
                let y = calc.offsetY + ceilingY * calc.imageHeight
                path.move(to: CGPoint(x: calc.offsetX, y: y))
                path.addLine(to: CGPoint(x: calc.offsetX + calc.imageWidth, y: y))
            }
            .stroke(Color.cyan, lineWidth: 8)

            // Left wall line (RED)
            Path { path in
                let x = calc.offsetX + leftX * calc.imageWidth
                path.move(to: CGPoint(x: x, y: calc.offsetY))
                path.addLine(to: CGPoint(x: x, y: calc.offsetY + calc.imageHeight))
            }
            .stroke(Color.red, lineWidth: 6)

            // Right wall line (YELLOW)
            Path { path in
                let x = calc.offsetX + rightX * calc.imageWidth
                path.move(to: CGPoint(x: x, y: calc.offsetY))
                path.addLine(to: CGPoint(x: x, y: calc.offsetY + calc.imageHeight))
            }
            .stroke(Color.yellow, lineWidth: 6)

            // Vanishing point (MAGENTA)
            Circle()
                .fill(Color(red: 1.0, green: 0.0, blue: 1.0))
                .frame(width: 30, height: 30)
                .position(
                    x: calc.offsetX + vanishingX * calc.imageWidth,
                    y: calc.offsetY + vanishingY * calc.imageHeight
                )

            // Crosshair
            Path { path in
                let vpX = calc.offsetX + vanishingX * calc.imageWidth
                let vpY = calc.offsetY + vanishingY * calc.imageHeight
                path.move(to: CGPoint(x: vpX - 30, y: vpY))
                path.addLine(to: CGPoint(x: vpX + 30, y: vpY))
                path.move(to: CGPoint(x: vpX, y: vpY - 30))
                path.addLine(to: CGPoint(x: vpX, y: vpY + 30))
            }
            .stroke(Color.white, lineWidth: 3)
        }
    }
}



// MARK: - Draggable Handles Overlay (fixed: no top-level lets in ViewBuilder)
struct DraggableHandlesOverlay: View {
    let geometry: GeometryProxy
    let imageSize: CGSize
    @Binding var floorY: CGFloat
    @Binding var ceilingY: CGFloat
    @Binding var leftX: CGFloat
    @Binding var rightX: CGFloat
    @Binding var vanishingX: CGFloat
    @Binding var vanishingY: CGFloat
    let magentaColor: Color

    var body: some View {
        DraggableHandlesOverlayInner(
            calc: computeBounds(),
            floorY: $floorY,
            ceilingY: $ceilingY,
            leftX: $leftX,
            rightX: $rightX,
            vanishingX: $vanishingX,
            vanishingY: $vanishingY,
            magentaColor: magentaColor
        )
    }

    private func computeBounds() -> (imageWidth: CGFloat, imageHeight: CGFloat, offsetX: CGFloat, offsetY: CGFloat) {
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = geometry.size.width / geometry.size.height

        var imageWidth: CGFloat
        var imageHeight: CGFloat
        var offsetX: CGFloat = 0
        var offsetY: CGFloat = 0

        if imageAspect > viewAspect {
            imageWidth = geometry.size.width
            imageHeight = geometry.size.width / imageAspect
            offsetY = (geometry.size.height - imageHeight) / 2
        } else {
            imageHeight = geometry.size.height
            imageWidth = geometry.size.height * imageAspect
            offsetX = (geometry.size.width - imageWidth) / 2
        }

        return (imageWidth, imageHeight, offsetX, offsetY)
    }
}

private struct DraggableHandlesOverlayInner: View {
    let calc: (imageWidth: CGFloat, imageHeight: CGFloat, offsetX: CGFloat, offsetY: CGFloat)
    @Binding var floorY: CGFloat
    @Binding var ceilingY: CGFloat
    @Binding var leftX: CGFloat
    @Binding var rightX: CGFloat
    @Binding var vanishingX: CGFloat
    @Binding var vanishingY: CGFloat
    let magentaColor: Color

    var body: some View {
        ZStack {
            // Floor handle (GREEN)
            DraggableHandle(color: .green, icon: "arrow.down.circle.fill")
                .position(
                    x: calc.offsetX + calc.imageWidth / 2,
                    y: calc.offsetY + floorY * calc.imageHeight
                )
                .gesture(
                    DragGesture().onChanged { value in
                        let newY = value.location.y - calc.offsetY
                        floorY = min(max(newY / calc.imageHeight, 0.5), 0.95)
                    }
                )

            // Ceiling handle (CYAN)
            DraggableHandle(color: .cyan, icon: "arrow.up.circle.fill")
                .position(
                    x: calc.offsetX + calc.imageWidth / 2,
                    y: calc.offsetY + ceilingY * calc.imageHeight
                )
                .gesture(
                    DragGesture().onChanged { value in
                        let newY = value.location.y - calc.offsetY
                        ceilingY = min(max(newY / calc.imageHeight, 0.05), 0.5)
                    }
                )

            // Left wall handle (RED)
            DraggableHandle(color: .red, icon: "arrow.left.circle.fill")
                .position(
                    x: calc.offsetX + leftX * calc.imageWidth,
                    y: calc.offsetY + calc.imageHeight / 2
                )
                .gesture(
                    DragGesture().onChanged { value in
                        let newX = value.location.x - calc.offsetX
                        leftX = min(max(newX / calc.imageWidth, 0.02), 0.4)
                    }
                )

            // Right wall handle (YELLOW)
            DraggableHandle(color: .yellow, icon: "arrow.right.circle.fill")
                .position(
                    x: calc.offsetX + rightX * calc.imageWidth,
                    y: calc.offsetY + calc.imageHeight / 2
                )
                .gesture(
                    DragGesture().onChanged { value in
                        let newX = value.location.x - calc.offsetX
                        rightX = min(max(newX / calc.imageWidth, 0.6), 0.98)
                    }
                )

            // Vanishing point handle (MAGENTA)
            DraggableHandle(color: magentaColor, icon: "scope", size: 50)
                .position(
                    x: calc.offsetX + vanishingX * calc.imageWidth,
                    y: calc.offsetY + vanishingY * calc.imageHeight
                )
                .gesture(
                    DragGesture().onChanged { value in
                        let newX = value.location.x - calc.offsetX
                        let newY = value.location.y - calc.offsetY
                        vanishingX = min(max(newX / calc.imageWidth, 0.1), 0.9)
                        vanishingY = min(max(newY / calc.imageHeight, 0.1), 0.9)
                    }
                )
        }
    }
}


// MARK: - Draggable Handle Component
struct DraggableHandle: View {
    let color: Color
    let icon: String
    var size: CGFloat = 44
    
    var body: some View {
        Image(systemName: icon)
            .font(.system(size: size))
            .foregroundColor(color)
            .shadow(color: .black.opacity(0.5), radius: 3, x: 0, y: 2)
            .shadow(color: color.opacity(0.5), radius: 10, x: 0, y: 0)
    }
}

import SwiftUI
import RoomPlan

// MARK: - Render Mode
enum RoomRenderMode: String, CaseIterable {
    case metalSplat = "Metal Splats"
    case sceneKit = "SceneKit Room"
    case yoloeRoom = "YOLOE Room"
}

struct SinglePhotoRoomView: View {
    @StateObject private var reconstructor = SinglePhotoRoomReconstructor()
    @State private var selectedImage: UIImage?
    @State private var showImagePicker = false
    @State private var adjustedBoundaries: RoomStructure?
    @State private var navigateToViewer = false
    @State private var fixedImage: UIImage? // ✅ Store fixed image separately

    // Render mode toggle - default to Metal splats
    @AppStorage("roomRenderMode") private var renderMode: String = RoomRenderMode.metalSplat.rawValue

    // Identifiable wrapper for reliable sheet(item:) presentation
    @State private var fixedImageItem: IdentifiedImage?

    struct IdentifiedImage: Identifiable {
        let id = UUID()
        let image: UIImage
    }

    // Read dimensions from settings
    @AppStorage("singlePhotoRoom.width") private var roomWidth: Double = 4.0
    @AppStorage("singlePhotoRoom.depth") private var roomDepth: Double = 4.5
    @AppStorage("singlePhotoRoom.height") private var roomHeight: Double = 2.8
    
    var body: some View {
        // ✅ Simple photo selection screen only
        VStack(spacing: 20) {
            Text(L10n.PhotoRoom.createTitle)
                .font(.title2)
                .fontWeight(.bold)
                .padding(.top, 40)

            Text(L10n.PhotoRoom.createSubtitle)
                .font(.subheadline)
                .foregroundColor(.secondary)

            Button(action: {
                logDebug("🖼️ [View] Select photo button tapped")
                showImagePicker = true
            }) {
                VStack(spacing: 16) {
                    Image(systemName: "photo.on.rectangle")
                        .font(.system(size: 50))
                        .foregroundColor(.orange)

                    VStack(spacing: 4) {
                        Text(L10n.PhotoRoom.quickPhoto)
                            .font(.headline)
                        Text(L10n.PhotoRoom.quickPhotoSubtitle)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(24)
                .background(Color.orange.opacity(0.1))
                .cornerRadius(16)
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.orange, lineWidth: 2)
                )
            }
            .padding(.horizontal)

            Spacer()
        }
        .navigationTitle(L10n.PhotoRoom.title)
        .sheet(isPresented: $showImagePicker) {
            PhotoPickerView(selectedImage: $selectedImage)
        }
        .onChange(of: selectedImage) { oldValue, newValue in
            guard let image = newValue else { return }
            logDebug("✅ [View] Image selected, preparing boundary sheet...")
            // Capture image and dismiss picker first
            fixedImage = image
            // Ensure the picker is dismissed before presenting another sheet
            if showImagePicker {
                showImagePicker = false
            }
            // Present boundary sheet on next runloop after dismissal
            DispatchQueue.main.async {
                fixedImageItem = IdentifiedImage(image: image)
            }
        }
        .sheet(item: $fixedImageItem) { item in
            RoomBoundaryDetectionView(
                originalImage: item.image,
                savedBoundaries: $adjustedBoundaries
            )
            .onAppear {
                logDebug("✅ [Sheet] Opening RoomBoundaryDetectionView with image: \(item.image.size)")
            }
        }
        .onAppear {
            logDebug("👁️ [View] SinglePhotoRoomView appeared")
            // Dimensions are now managed by @AppStorage
        }
        // ✅ Watch for boundary changes and rebuild automatically, then navigate to viewer
        .onChange(of: adjustedBoundaries) { oldValue, newValue in
            if let boundaries = newValue, let image = fixedImage {
                print("🔄🔄🔄 [View] Done pressed - boundaries adjusted, calling processPhotoWithBoundaries 🔄🔄🔄")
                logDebug("🔄 [View] Boundaries adjusted, rebuilding room...")
                Task {
                    // Use dimensions from settings
                    var updatedDimensions = reconstructor.estimatedDimensions ?? SinglePhotoRoomReconstructor.RoomDimensions()
                    updatedDimensions.width = Float(roomWidth)
                    updatedDimensions.depth = Float(roomDepth)
                    updatedDimensions.height = Float(roomHeight)
                    
                    await MainActor.run {
                        reconstructor.estimatedDimensions = updatedDimensions
                    }
                    
                    await reconstructor.processPhotoWithBoundaries(image, boundaries: boundaries)
                    // Auto-navigate to viewer once ready
                    let useMetalSplats = renderMode == RoomRenderMode.metalSplat.rawValue
                    let canNavigate = useMetalSplats
                        ? !reconstructor.rawSplats.isEmpty
                        : reconstructor.generatedRoomScene != nil
                    if canNavigate {
                        await MainActor.run {
                            navigateToViewer = true
                        }
                    }
                }
            }
        }
        // Programmatic navigation using the modern API (iOS 16+).  When
        // `navigateToViewer` is set to true, a destination is pushed onto
        // the navigation stack.  We wrap the destination in a `Group` to
        // handle the optional room scene gracefully.
        .navigationDestination(isPresented: $navigateToViewer) {
            Group {
                if renderMode == RoomRenderMode.metalSplat.rawValue {
                    // Pure Metal splat renderer
                    StandaloneSplatViewer(splats: reconstructor.rawSplats)
                } else if renderMode == RoomRenderMode.yoloeRoom.rawValue {
                    // YOLOE detection-based room
                    if let image = fixedImage {
                        YOLOERoomViewer(image: image)
                    }
                } else if let scene = reconstructor.generatedRoomScene {
                    // SceneKit room with planes
                    SceneKitViewer(scene: scene)
                }
            }
        }
    }
    
    private func confidenceColor(_ confidence: Float) -> Color {
        switch confidence {
        case 0.7...1.0: return .green
        case 0.4..<0.7: return .orange
        default: return .red
        }
    }
}

// MARK: - Photo Picker View
struct PhotoPickerView: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.dismiss) var dismiss
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        logDebug("📱 [PhotoPicker] Creating UIImagePickerController")
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator { Coordinator(self) }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: PhotoPickerView
        
        init(_ parent: PhotoPickerView) {
            self.parent = parent
            logDebug("📱 [PhotoPicker] Coordinator initialized")
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            logDebug("📱 [PhotoPicker] Image picked from library")
            if let image = info[.originalImage] as? UIImage {
                logDebug("✅ [PhotoPicker] Got UIImage: \(image.size)")
                parent.selectedImage = image
            } else {
                logDebug("❌ [PhotoPicker] Failed to get UIImage")
            }
            parent.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            logDebug("❌ [PhotoPicker] User cancelled")
            parent.dismiss()
        }
    }
}

// MARK: - YOLOE Room Viewer
/// Constructs a 3D room from YOLOE furniture detections
struct YOLOERoomViewer: View {
    let image: UIImage
    @Environment(\.dismiss) private var dismiss
    @StateObject private var viewModel = YOLOERoomViewModel()

    var body: some View {
        ZStack {
            if viewModel.isLoading {
                VStack(spacing: 16) {
                    ProgressView()
                        .scaleEffect(1.5)
                    Text(viewModel.statusMessage)
                        .foregroundColor(.secondary)
                }
            } else if let scene = viewModel.scene {
                SceneKitViewer(scene: scene)
            } else if let error = viewModel.errorMessage {
                VStack(spacing: 16) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle)
                        .foregroundColor(.orange)
                    Text(error)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                    Button("Try Again") {
                        viewModel.processImage(image)
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding()
            }
        }
        .navigationBarBackButtonHidden(true)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button(action: { dismiss() }) {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                        Text("Back")
                    }
                }
            }
        }
        .onAppear {
            viewModel.processImage(image)
        }
    }
}

// MARK: - YOLOE Room ViewModel
@MainActor
class YOLOERoomViewModel: ObservableObject {
    @Published var isLoading = false
    @Published var statusMessage = "Loading model..."
    @Published var scene: SCNScene?
    @Published var errorMessage: String?
    @Published var detections: [YOLOEDetection] = []

    private var mlModel: MLModel?

    struct YOLOEDetection {
        let classId: Int
        let className: String
        let confidence: Float
        let bbox: CGRect  // Normalized 0-1
    }

    // Class names loaded from classes.json
    private lazy var classNames: [Int: String] = {
        guard let url = Bundle.main.url(forResource: "classes", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            return [:]
        }
        var result: [Int: String] = [:]
        for (key, value) in dict {
            if let id = Int(key) { result[id] = value }
        }
        return result
    }()

    func processImage(_ image: UIImage) {
        isLoading = true
        errorMessage = nil
        statusMessage = "Loading model..."

        Task {
            do {
                // Load model
                guard let modelUrl = Bundle.main.url(forResource: "yoloe-11l-seg-pf", withExtension: "mlmodelc") else {
                    throw NSError(domain: "YOLOERoom", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model not found"])
                }

                let config = MLModelConfiguration()
                config.computeUnits = .cpuOnly
                mlModel = try MLModel(contentsOf: modelUrl, configuration: config)

                statusMessage = "Running detection..."

                // Run inference
                let dets = try runInference(image: image)
                self.detections = dets

                statusMessage = "Building room..."

                // Build 3D scene
                let scene = buildScene(from: dets, imageSize: image.size)
                self.scene = scene
                self.isLoading = false

                print("YOLOE Room: Found \(dets.count) furniture items")

            } catch {
                self.errorMessage = error.localizedDescription
                self.isLoading = false
            }
        }
    }

    private func runInference(image: UIImage) throws -> [YOLOEDetection] {
        guard let model = mlModel else {
            throw NSError(domain: "YOLOERoom", code: 2, userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }

        // Resize to 1280x1280
        guard let resized = resizeImage(image, to: CGSize(width: 1280, height: 1280)),
              let pixelBuffer = resized.pixelBuffer() else {
            throw NSError(domain: "YOLOERoom", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to prepare image"])
        }

        // Convert to MLMultiArray
        guard let inputArray = pixelBufferToMLMultiArray(pixelBuffer) else {
            throw NSError(domain: "YOLOERoom", code: 4, userInfo: [NSLocalizedDescriptionKey: "Failed to convert image"])
        }

        // Run inference
        let inputProvider = try MLDictionaryFeatureProvider(dictionary: ["image": inputArray])
        let output = try model.prediction(from: inputProvider)

        // Parse detections
        guard let detArray = output.featureValue(for: "var_2497")?.multiArrayValue else {
            throw NSError(domain: "YOLOERoom", code: 5, userInfo: [NSLocalizedDescriptionKey: "No detection output"])
        }

        return parseDetections(detArray, confidenceThreshold: 0.25)
    }

    private func parseDetections(_ detArray: MLMultiArray, confidenceThreshold: Float) -> [YOLOEDetection] {
        let numFeatures = detArray.shape[1].intValue
        let numAnchors = detArray.shape[2].intValue
        let numClasses = numFeatures - 4 - 32

        guard numFeatures >= 36, numAnchors > 0, numClasses > 0 else { return [] }

        // Copy to float buffer
        let totalCount = detArray.count
        var detBuf = [Float](repeating: 0, count: totalCount)

        if detArray.dataType == .float32 {
            memcpy(&detBuf, detArray.dataPointer, totalCount * MemoryLayout<Float>.size)
        } else if detArray.dataType == .float16 {
            let src = detArray.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
            for i in 0..<totalCount {
                detBuf[i] = Float(float16ToFloat32(src[i]))
            }
        }

        let stride = numAnchors
        var results: [YOLOEDetection] = []

        for anchor in 0..<numAnchors {
            let x = detBuf[0 * stride + anchor]
            let y = detBuf[1 * stride + anchor]
            let w = detBuf[2 * stride + anchor]
            let h = detBuf[3 * stride + anchor]

            guard x.isFinite, y.isFinite, w.isFinite, h.isFinite, w > 0, h > 0 else { continue }

            // Find max class score
            var maxScore: Float = 0
            var maxIdx = 0
            for c in 0..<numClasses {
                let score = detBuf[(4 + c) * stride + anchor]
                if score > maxScore {
                    maxScore = score
                    maxIdx = c
                }
            }

            guard maxScore > confidenceThreshold else { continue }

            // Normalize bbox to 0-1
            let bbox = CGRect(
                x: CGFloat((x - w/2) / 1280),
                y: CGFloat((y - h/2) / 1280),
                width: CGFloat(w / 1280),
                height: CGFloat(h / 1280)
            )

            let className = classNames[maxIdx] ?? "object_\(maxIdx)"
            results.append(YOLOEDetection(classId: maxIdx, className: className, confidence: maxScore, bbox: bbox))
        }

        // Apply simple NMS
        return applyNMS(results, iouThreshold: 0.5)
    }

    private func applyNMS(_ detections: [YOLOEDetection], iouThreshold: Float) -> [YOLOEDetection] {
        var sorted = detections.sorted { $0.confidence > $1.confidence }
        var kept: [YOLOEDetection] = []

        while !sorted.isEmpty {
            let current = sorted.removeFirst()
            kept.append(current)

            sorted.removeAll { det in
                iou(current.bbox, det.bbox) > CGFloat(iouThreshold)
            }
        }
        return kept
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        if intersection.isNull { return 0 }
        let interArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return unionArea > 0 ? interArea / unionArea : 0
    }

    private func buildScene(from detections: [YOLOEDetection], imageSize: CGSize) -> SCNScene {
        let scene = SCNScene()

        // Room dimensions from settings
        let roomWidth: Float = Float(UserDefaults.standard.double(forKey: "singlePhotoRoom.width"))
        let roomDepth: Float = Float(UserDefaults.standard.double(forKey: "singlePhotoRoom.depth"))
        let roomHeight: Float = Float(UserDefaults.standard.double(forKey: "singlePhotoRoom.height"))

        let w = roomWidth > 0 ? roomWidth : 4.0
        let d = roomDepth > 0 ? roomDepth : 4.5
        let h = roomHeight > 0 ? roomHeight : 2.8

        // Add floor
        let floor = SCNBox(width: CGFloat(w), height: 0.02, length: CGFloat(d), chamferRadius: 0)
        floor.firstMaterial?.diffuse.contents = UIColor(white: 0.9, alpha: 1)
        let floorNode = SCNNode(geometry: floor)
        floorNode.position = SCNVector3(0, 0, 0)
        scene.rootNode.addChildNode(floorNode)

        // Add walls (back wall)
        let backWall = SCNBox(width: CGFloat(w), height: CGFloat(h), length: 0.02, chamferRadius: 0)
        backWall.firstMaterial?.diffuse.contents = UIColor(white: 0.95, alpha: 1)
        let backNode = SCNNode(geometry: backWall)
        backNode.position = SCNVector3(0, h/2, -d/2)
        scene.rootNode.addChildNode(backNode)

        // Left wall
        let leftWall = SCNBox(width: 0.02, height: CGFloat(h), length: CGFloat(d), chamferRadius: 0)
        leftWall.firstMaterial?.diffuse.contents = UIColor(white: 0.92, alpha: 1)
        let leftNode = SCNNode(geometry: leftWall)
        leftNode.position = SCNVector3(-w/2, h/2, 0)
        scene.rootNode.addChildNode(leftNode)

        // Right wall
        let rightWall = SCNBox(width: 0.02, height: CGFloat(h), length: CGFloat(d), chamferRadius: 0)
        rightWall.firstMaterial?.diffuse.contents = UIColor(white: 0.92, alpha: 1)
        let rightNode = SCNNode(geometry: rightWall)
        rightNode.position = SCNVector3(w/2, h/2, 0)
        scene.rootNode.addChildNode(rightNode)

        // Add furniture boxes from detections
        let colors: [UIColor] = [.systemBlue, .systemGreen, .systemOrange, .systemPurple, .systemPink, .systemTeal]

        for (i, det) in detections.enumerated() {
            // Map 2D bbox to 3D position
            // bbox.x = horizontal position (0=left, 1=right)
            // bbox.y = vertical position (0=top, 1=bottom)
            // Items lower in image (higher Y) are CLOSER to camera (positive Z)

            let centerX = Float(det.bbox.midX - 0.5) * w  // Left-right mapping

            // Depth mapping: bottom of image = front of room, top = back
            // bbox.maxY gives the bottom of the detection (feet of furniture)
            let depthFactor = Float(det.bbox.maxY)  // 0=top (far), 1=bottom (near)
            let centerZ = (depthFactor - 0.5) * d  // Maps to [-d/2, +d/2], positive = closer

            // Estimate furniture size from bbox (smaller if further away)
            let perspectiveScale = 0.5 + depthFactor * 0.5  // Items at bottom appear larger
            let furnitureWidth = Float(det.bbox.width) * w * 0.6 * perspectiveScale
            let furnitureDepth = Float(det.bbox.height) * d * 0.4 * perspectiveScale
            let furnitureHeight = min(furnitureWidth, furnitureDepth) * 0.8 + 0.4  // Heuristic height

            // Create furniture box
            let box = SCNBox(
                width: CGFloat(furnitureWidth),
                height: CGFloat(furnitureHeight),
                length: CGFloat(furnitureDepth),
                chamferRadius: 0.02
            )

            let color = colors[i % colors.count]
            box.firstMaterial?.diffuse.contents = color.withAlphaComponent(0.7)

            let node = SCNNode(geometry: box)
            node.position = SCNVector3(centerX, furnitureHeight/2, centerZ)
            node.name = det.className

            // Add label
            let text = SCNText(string: det.className, extrusionDepth: 0.01)
            text.font = UIFont.systemFont(ofSize: 0.1)
            text.firstMaterial?.diffuse.contents = UIColor.white
            let textNode = SCNNode(geometry: text)
            textNode.position = SCNVector3(-furnitureWidth/2, furnitureHeight + 0.05, 0)
            textNode.scale = SCNVector3(0.5, 0.5, 0.5)
            node.addChildNode(textNode)

            scene.rootNode.addChildNode(node)
        }

        // Add lighting
        let ambientLight = SCNLight()
        ambientLight.type = .ambient
        ambientLight.intensity = 500
        let ambientNode = SCNNode()
        ambientNode.light = ambientLight
        scene.rootNode.addChildNode(ambientNode)

        let directionalLight = SCNLight()
        directionalLight.type = .directional
        directionalLight.intensity = 800
        let lightNode = SCNNode()
        lightNode.light = directionalLight
        lightNode.position = SCNVector3(0, h, d)
        lightNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(lightNode)

        // Add camera
        let camera = SCNCamera()
        camera.zNear = 0.1
        camera.zFar = 100
        camera.fieldOfView = 60
        let cameraNode = SCNNode()
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, h * 0.6, d * 1.5)
        cameraNode.look(at: SCNVector3(0, h * 0.3, 0))
        scene.rootNode.addChildNode(cameraNode)

        return scene
    }

    // MARK: - Helper Functions

    private func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized
    }

    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width == 1280, height == 1280 else { return nil }
        guard let array = try? MLMultiArray(shape: [1, 3, 1280, 1280], dataType: .float32) else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let src = baseAddress.assumingMemoryBound(to: UInt8.self)
        let pixelCount = width * height

        let rPtr = array.dataPointer.assumingMemoryBound(to: Float32.self)
        let gPtr = rPtr.advanced(by: pixelCount)
        let bPtr = gPtr.advanced(by: pixelCount)

        for y in 0..<height {
            let row = src.advanced(by: y * bytesPerRow)
            for x in 0..<width {
                let pixel = row.advanced(by: x * 4)
                let idx = y * width + x
                rPtr[idx] = Float32(pixel[2]) / 255.0  // R
                gPtr[idx] = Float32(pixel[1]) / 255.0  // G
                bPtr[idx] = Float32(pixel[0]) / 255.0  // B
            }
        }
        return array
    }

    private func float16ToFloat32(_ value: UInt16) -> Float {
        let sign = (value & 0x8000) >> 15
        let exp = (value & 0x7C00) >> 10
        let frac = value & 0x03FF

        if exp == 0 {
            if frac == 0 { return sign == 0 ? 0.0 : -0.0 }
            // Subnormal
            var f = Float(frac) / 1024.0
            f *= pow(2.0, -14.0)
            return sign == 0 ? f : -f
        } else if exp == 31 {
            if frac == 0 { return sign == 0 ? .infinity : -.infinity }
            return .nan
        }

        let f = (1.0 + Float(frac) / 1024.0) * pow(2.0, Float(Int(exp) - 15))
        return sign == 0 ? f : -f
    }
}

// MARK: - UIImage Extension for PixelBuffer
extension UIImage {
    func pixelBuffer() -> CVPixelBuffer? {
        let width = Int(size.width)
        let height = Int(size.height)

        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width, height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        guard let cgImage = self.cgImage else { return nil }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        return buffer
    }
}

// MARK: - SceneKit Viewer
struct SceneKitViewer: View {
    let scene: SCNScene
    @Environment(\.dismiss) private var dismiss
    @State private var showControls = true
    @State private var cameraNode: SCNNode?

    // Save room state
    @StateObject private var modelManager = USDZModelManager()
    @State private var isSavingRoom = false
    @State private var saveProgress: Double = 0.0
    @State private var savingTimer: Timer?
    @State private var showSaveAlert = false
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false
    @State private var showRoomNameInput = false
    @State private var roomName = ""

    var body: some View {
        ZStack {
            SceneView(
                scene: scene,
                pointOfView: cameraNode,
                options: [.allowsCameraControl, .autoenablesDefaultLighting]
            )
            .onAppear {
                logDebug("🎬 [Viewer] SceneKit viewer appeared")
                logDebug("   - Scene nodes: \(scene.rootNode.childNodes.count)")
                setupCamera()
            }
            .onDisappear {
                GlobalCameraController.shared.clearCamera()
            }

            // Save progress overlay
            if isSavingRoom {
                saveRoomProgressOverlay
            }

            // ✅ GLOBAL JOYSTICK - uses GlobalCameraController
            SimpleJoystickOverlay()
                .zIndex(99997)

            if showControls {
                VStack {
                    Spacer()
                    HStack {
                        Image(systemName: "hand.draw")
                        Text(L10n.RoomViewer.controls)
                            .font(.caption)
                    }
                    .padding(8)
                    .background(.ultraThinMaterial)
                    .cornerRadius(8)
                    .padding()
                    .padding(.bottom, 100) // Move above joystick
                }
                .onAppear {
                    logDebug("ℹ️ [Viewer] Controls hint displayed")
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                        withAnimation { showControls = false }
                    }
                }
            }
        }
        .navigationTitle(L10n.RoomViewer.title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            // Help button
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showControls.toggle()
                    logDebug("ℹ️ [Viewer] Controls hint toggled: \(showControls)")
                } label: {
                    Image(systemName: "questionmark.circle")
                }
            }

            // Save Room button
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showRoomNameInput = true
                } label: {
                    Image(systemName: "square.and.arrow.down")
                }
            }
        }
        // Room name input alert
        .alert(L10n.RoomViewer.saveRoom, isPresented: $showRoomNameInput) {
            TextField(L10n.RoomViewer.roomName, text: $roomName)
            Button(L10n.Common.cancel, role: .cancel) {
                roomName = ""
            }
            Button(L10n.Common.save) {
                startSavingRoom()
            }
            .disabled(roomName.isEmpty)
        } message: {
            Text(L10n.RoomViewer.enterName)
        }
        // Save result alert
        .alert(L10n.RoomViewer.roomSaveTitle, isPresented: $showSaveAlert) {
            Button(L10n.Common.ok, role: .cancel) {
                if saveWasSuccessful {
                    // Post notification to dismiss the entire photo room sheet
                    NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil)
                }
            }
        } message: {
            Text(saveAlertMessage)
        }
    }
    
    // MARK: - Save Room Progress Overlay
    private var saveRoomProgressOverlay: some View {
        ZStack {
            Color.black.opacity(0.9)
                .ignoresSafeArea()
            
            VStack(spacing: 24) {
                // Save icon with animation
                ZStack {
                    Circle()
                        .fill(Color.green.opacity(0.2))
                        .frame(width: 100, height: 100)
                    
                    Image(systemName: "square.and.arrow.down.fill")
                        .font(.system(size: 40))
                        .foregroundColor(.green)
                        .rotationEffect(.degrees(saveProgress < 0.5 ? 0 : 360))
                        .animation(.linear(duration: 1).repeatForever(autoreverses: false), value: saveProgress)
                }
                
                VStack(spacing: 12) {
                    Text(L10n.RoomViewer.savingRoom)
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)

                    Text(saveProgressMessage)
                        .font(.subheadline)
                        .foregroundColor(.gray)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 40)
                }
                
                // Progress bar
                VStack(spacing: 8) {
                    ProgressView(value: saveProgress, total: 1.0)
                        .progressViewStyle(LinearProgressViewStyle(tint: .green))
                        .frame(width: 250)
                    
                    Text("\(Int(saveProgress * 100))%")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                
                // Cancel button
                Button(action: {
                    cancelSavingRoom()
                }) {
                    Text(L10n.Common.cancel)
                        .font(.body)
                        .foregroundColor(.white)
                        .padding(.horizontal, 32)
                        .padding(.vertical, 12)
                        .background(Color.red.opacity(0.8))
                        .cornerRadius(25)
                }
                .padding(.top, 8)
            }
        }
        .transition(.opacity)
    }
    
    private var saveProgressMessage: String {
        if saveProgress < 0.3 {
            return L10n.RoomViewer.preparingModel
        } else if saveProgress < 0.6 {
            return L10n.RoomViewer.exportingUSDZ
        } else if saveProgress < 0.9 {
            return L10n.RoomViewer.savingToLibrary
        } else {
            return L10n.RoomViewer.almostDone
        }
    }
    
    // MARK: - Save Room Functions
    private func startSavingRoom() {
        guard !roomName.isEmpty else {
            return
        }
        
        let savedName = roomName  // ✅ Capture the name BEFORE clearing
        logDebug("💾 [Viewer] Starting room save process: \(savedName)")
        
        withAnimation(.easeIn(duration: 0.3)) {
            isSavingRoom = true
            saveProgress = 0.0
        }
        
        var saveStarted = false
        var saveCompleted = false
        var saveSuccess = false
        var saveError: String?
        
        // Progress timer
        savingTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { timer in
            // Only advance progress if not waiting for save completion
            if !saveStarted || (saveStarted && saveCompleted) {
                self.saveProgress += 0.015
            }
            
            if self.saveProgress >= 0.3 && self.saveProgress < 0.32 {
                logDebug("📦 [Viewer] Preparing model...")
            } else if self.saveProgress >= 0.6 && !saveStarted {
                logDebug("📄 [Viewer] Exporting USDZ...")
                saveStarted = true
                
                // ✅ Actually save the room with completion handler
                self.modelManager.saveRoom(scene: scene, name: savedName) { success, error in
                    DispatchQueue.main.async {
                        saveCompleted = true
                        saveSuccess = success
                        saveError = error
                        logDebug(success ? "✅ [Viewer] Room saved successfully" : "❌ [Viewer] Failed to save: \(error ?? "unknown")")
                    }
                }
            } else if self.saveProgress >= 0.9 && self.saveProgress < 0.92 {
                logDebug("💾 [Viewer] Finalizing...")
            }
            
            // ✅ Only finish when BOTH progress complete AND save completed
            if self.saveProgress >= 1.0 && saveCompleted {
                timer.invalidate()
                self.savingTimer = nil
                
                withAnimation(.easeOut(duration: 0.3)) {
                    self.isSavingRoom = false
                }
                
                // Show result based on actual save status
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    if saveSuccess {
                        self.saveAlertMessage = L10n.RoomViewer.saveSuccess(savedName)
                        self.saveWasSuccessful = true
                        self.showSaveAlert = true
                        self.roomName = ""
                        logDebug("✅ [Viewer] Save complete!")
                    } else {
                        self.saveAlertMessage = L10n.RoomViewer.saveFailed(saveError ?? "Unknown error")
                        self.saveWasSuccessful = false
                        self.showSaveAlert = true
                        logDebug("❌ [Viewer] Save failed!")
                    }
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
        logDebug("❌ [Viewer] Room save cancelled")
    }

    // ✅ Setup camera position like vintage room (outside, looking at front wall)
    private func setupCamera() {
        logDebug("📷 [SceneKitViewer] Setting up camera position...")

        // Calculate scene bounds
        var minBounds = SCNVector3(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
        var maxBounds = SCNVector3(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)

        scene.rootNode.enumerateChildNodes { node, _ in
            let (localMin, localMax) = node.boundingBox
            let worldMin = node.convertPosition(localMin, to: nil)
            let worldMax = node.convertPosition(localMax, to: nil)

            minBounds.x = min(minBounds.x, worldMin.x, worldMax.x)
            minBounds.y = min(minBounds.y, worldMin.y, worldMax.y)
            minBounds.z = min(minBounds.z, worldMin.z, worldMax.z)

            maxBounds.x = max(maxBounds.x, worldMin.x, worldMax.x)
            maxBounds.y = max(maxBounds.y, worldMin.y, worldMax.y)
            maxBounds.z = max(maxBounds.z, worldMin.z, worldMax.z)
        }

        let roomSize = SCNVector3(
            maxBounds.x - minBounds.x,
            maxBounds.y - minBounds.y,
            maxBounds.z - minBounds.z
        )
        let roomCenter = SCNVector3(
            (minBounds.x + maxBounds.x) / 2,
            (minBounds.y + maxBounds.y) / 2,
            (minBounds.z + maxBounds.z) / 2
        )

        logDebug("   📦 Room bounds: min(\(minBounds)), max(\(maxBounds))")
        logDebug("   📏 Room size: \(roomSize.x) x \(roomSize.y) x \(roomSize.z)")
        logDebug("   🎯 Room center: \(roomCenter)")

        // ✅ Camera positioning: OUTSIDE the room (beyond max Z), looking at FRONT wall (min Z)
        // Same strategy as RealityKitBoundaryManager.getOptimalCameraPosition()
        let camX = roomCenter.x  // Center X
        let camY = roomCenter.y  // Center height
        let camZ = maxBounds.z + (roomSize.z * 0.3)  // OUTSIDE room, beyond back

        let lookAtX = roomCenter.x  // Center X
        let lookAtY = roomCenter.y  // Center height
        let lookAtZ = minBounds.z   // FRONT wall (where photo is)

        logDebug("   📷 Camera position: (\(camX), \(camY), \(camZ))")
        logDebug("   👁️ Looking at: (\(lookAtX), \(lookAtY), \(lookAtZ))")

        // Create camera node
        let camera = SCNCamera()
        camera.automaticallyAdjustsZRange = true
        camera.fieldOfView = 60

        let camNode = SCNNode()
        camNode.camera = camera
        camNode.position = SCNVector3(camX, camY, camZ)

        // Point camera at front wall (without constraint for joystick movement)
        camNode.look(at: SCNVector3(lookAtX, lookAtY, lookAtZ))

        scene.rootNode.addChildNode(camNode)
        cameraNode = camNode

        // ✅ Register with GlobalCameraController for joystick movement
        GlobalCameraController.shared.registerSceneKitCamera(camNode)

        logDebug("   ✅ Camera setup complete and registered with GlobalCameraController")
    }
}

