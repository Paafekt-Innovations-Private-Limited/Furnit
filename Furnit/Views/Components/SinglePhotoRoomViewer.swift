import SwiftUI
import SceneKit
import Accelerate
import CoreML

// MARK: - Room Boundary Detection View with DRAGGABLE boundaries
struct RoomBoundaryDetectionView: View {
    let originalImage: UIImage
    @Binding var savedBoundaries: RoomStructure?
    // Optional: pass reconstructor for in-view processing
    @ObservedObject var reconstructor: SinglePhotoRoomReconstructor
    var roomDimensions: SinglePhotoRoomReconstructor.RoomDimensions?
    var onProcessingComplete: (() -> Void)?

    @Environment(\.dismiss) var dismiss

    // Boundary positions (as percentages of image dimensions)
    @State private var floorY: CGFloat = 0.85
    @State private var ceilingY: CGFloat = 0.15
    @State private var leftX: CGFloat = 0.12
    @State private var rightX: CGFloat = 0.88
    @State private var vanishingX: CGFloat = 0.5
    @State private var vanishingY: CGFloat = 0.45

    // Processing state for progress overlay
    @State private var isProcessingInView = false

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

                                logDebug("✅ Saved adjusted boundaries:")
                                logDebug("   Floor: \(floorY), Ceiling: \(ceilingY)")
                                logDebug("   Left: \(leftX), Right: \(rightX)")
                                logDebug("   VP: (\(vanishingX), \(vanishingY))")

                                // Process within this view with progress overlay
                                isProcessingInView = true
                                Task {
                                    let startTime = Date()
                                    let minimumDisplayTime: TimeInterval = 2.0 // Show progress for at least 2 seconds

                                    // Apply dimensions if provided
                                    if let dims = roomDimensions {
                                        await MainActor.run {
                                            reconstructor.estimatedDimensions = dims
                                        }
                                    }
                                    await reconstructor.processPhotoWithBoundaries(originalImage, boundaries: boundaries)

                                    // Ensure progress is shown for minimum time
                                    let elapsed = Date().timeIntervalSince(startTime)
                                    if elapsed < minimumDisplayTime {
                                        try? await Task.sleep(nanoseconds: UInt64((minimumDisplayTime - elapsed) * 1_000_000_000))
                                    }

                                    await MainActor.run {
                                        savedBoundaries = boundaries
                                        isProcessingInView = false
                                        onProcessingComplete?()
                                        dismiss()
                                    }
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(isProcessingInView)
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
                    .disabled(isProcessingInView)
                }
            }
            // Progress overlay when processing
            .overlay {
                if isProcessingInView {
                    ZStack {
                        Color.black.opacity(0.7)
                            .ignoresSafeArea()

                        VStack(spacing: 16) {
                            ZStack {
                                Circle()
                                    .stroke(Color.orange.opacity(0.3), lineWidth: 8)
                                    .frame(width: 60, height: 60)
                                Circle()
                                    .trim(from: 0, to: CGFloat(reconstructor.progress))
                                    .stroke(Color.orange, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                                    .frame(width: 60, height: 60)
                                    .rotationEffect(.degrees(-90))
                                    .animation(.easeInOut(duration: 0.3), value: reconstructor.progress)
                                Image(systemName: "cube.fill")
                                    .font(.system(size: 24))
                                    .foregroundColor(.orange)
                            }

                            Text(reconstructor.statusMessage)
                                .font(.headline)
                                .foregroundColor(.white)

                            Text("\(Int(reconstructor.progress * 100))%")
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundColor(.orange)

                            Text(NSLocalizedString("photoRoom.buildingRoom", comment: "Building your 3D room"))
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                        .padding(32)
                        .background(Color(.systemBackground).opacity(0.95))
                        .cornerRadius(16)
                        .shadow(radius: 10)
                    }
                }
            }
        }
        .interactiveDismissDisabled(isProcessingInView)
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

struct SinglePhotoRoomView: View {
    @StateObject private var reconstructor = SinglePhotoRoomReconstructor()
    @ObservedObject private var sharpService = SHARPService.shared
    @State private var selectedImage: UIImage?
    @State private var showImagePicker = false
    @State private var showCameraCapture = false  // Show camera capture view
    @State private var captureOrientation: CaptureOrientation = .landscape  // Camera orientation selection
    @State private var adjustedBoundaries: RoomStructure?
    @State private var navigateToViewer = false
    @State private var fixedImage: UIImage? // ✅ Store fixed image separately

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
    @State private var showGenerationSuccess = false
    @State private var generatedPLYURL: URL?
    @State private var generatedRoomMeasurements: RoomMeasurements?
    @State private var navigateToSplatViewer = false
    @State private var showMethodPicker = false  // Show method choice after photo selection
    @State private var showRoomBoundaries = false  // Show boundary adjustment sheet
    @State private var selectedOrientation: PhotoOrientation = .portrait  // User-selected orientation
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            VStack {
                if let image = selectedImage, showMethodPicker {
                    // Show image preview and method selection
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxHeight: 250)
                        .cornerRadius(12)
                        .padding()
                        .onAppear { logDebug("🖼️ [View] Displaying selected image with method picker") }

                    VStack(spacing: 4) {
                        Text(NSLocalizedString("photoRoom.howToCreate", comment: ""))
                            .font(.headline)
                        Text(NSLocalizedString("photoRoom.tapOption", comment: ""))
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top, 8)
                    .padding(.bottom, 8)

                    // Method 1: SHARP (AI-powered) - Single photo to 3D
                    Button(action: {
                        logDebug("🤖 [View] SHARP method selected")
                        logDebug("📸 User selected pic type: \(selectedOrientation == .portrait ? "Portrait" : "Landscape")")
                        showMethodPicker = false
                        startSHARPGeneration(image: image)
                    }) {
                        HStack(spacing: 16) {
                            Image(systemName: "wand.and.stars")
                                .font(.system(size: 30))
                                .foregroundColor(.purple)
                                .frame(width: 50)

                            VStack(alignment: .leading, spacing: 4) {
                                Text(NSLocalizedString("photoRoom.title", comment: ""))
                                    .font(.headline)
                                    .foregroundColor(.primary)
                                Text(NSLocalizedString("photoRoom.aiPowered", comment: ""))
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                            Image(systemName: "chevron.right")
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color.purple.opacity(0.1))
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.purple, lineWidth: 2)
                        )
                    }
                    .padding(.horizontal)

                    // Method 2: Manual Boundaries
                    Button(action: {
                        logDebug("🏠 [View] Manual boundaries method selected")
                        logDebug("📸 User selected pic type: \(selectedOrientation == .portrait ? "Portrait" : "Landscape")")
                        showMethodPicker = false
                        fixedImageItem = IdentifiedImage(image: image)
                    }) {
                        HStack(spacing: 16) {
                            Image(systemName: "square.resize")
                                .font(.system(size: 30))
                                .foregroundColor(.orange)
                                .frame(width: 50)

                            VStack(alignment: .leading, spacing: 4) {
                                Text(NSLocalizedString("photoRoom.manualSetup", comment: ""))
                                    .font(.headline)
                                    .foregroundColor(.primary)
                                Text(NSLocalizedString("photoRoom.manualSetupDesc", comment: ""))
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                            Image(systemName: "chevron.right")
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color.orange.opacity(0.1))
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.orange, lineWidth: 2)
                        )
                    }
                    .padding(.horizontal)

                    // Change photo button
                    Button("Choose Different Photo") {
                        selectedImage = nil
                        showMethodPicker = false
                        showImagePicker = true
                    }
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .padding(.top, 16)

                    Spacer()

                } else {
                    // Photo Selection (initial state)
                    VStack(spacing: 20) {
                        Text(NSLocalizedString("photoRoom.createTitle", comment: ""))
                            .font(.title2)
                            .fontWeight(.bold)
                            .padding(.top, 40)

                        Text(NSLocalizedString("photoRoom.createSubtitle", comment: ""))
                            .font(.subheadline)
                            .foregroundColor(.secondary)

                        // Camera button - NEW
                        Button(action: {
                            logDebug("📷 [View] Camera button tapped")
                            showCameraCapture = true
                        }) {
                            VStack(spacing: 16) {
                                Image(systemName: "camera.fill")
                                    .font(.system(size: 50))
                                    .foregroundColor(.blue)

                                VStack(spacing: 4) {
                                    Text(NSLocalizedString("camera.takePhoto", comment: ""))
                                        .font(.headline)
                                    Text(NSLocalizedString("camera.chooseOrientationShort", comment: ""))
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .padding(24)
                            .background(Color.blue.opacity(0.1))
                            .cornerRadius(16)
                            .overlay(
                                RoundedRectangle(cornerRadius: 16)
                                    .stroke(Color.blue, lineWidth: 2)
                            )
                        }
                        .padding(.horizontal)

                        // Divider with "or"
                        HStack {
                            Rectangle()
                                .fill(Color.secondary.opacity(0.3))
                                .frame(height: 1)
                            Text(NSLocalizedString("common.or", comment: ""))
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .padding(.horizontal, 8)
                            Rectangle()
                                .fill(Color.secondary.opacity(0.3))
                                .frame(height: 1)
                        }
                        .padding(.horizontal, 32)

                        // Photo library button
                        Button(action: {
                            logDebug("🖼️ [View] Select photo button tapped")
                            showImagePicker = true
                        }) {
                            VStack(spacing: 16) {
                                Image(systemName: "photo.on.rectangle")
                                    .font(.system(size: 50))
                                    .foregroundColor(.green)

                                VStack(spacing: 4) {
                                    Text(NSLocalizedString("photoRoom.selectPhoto", comment: ""))
                                        .font(.headline)
                                    Text(NSLocalizedString("photoRoom.fromLibrary", comment: ""))
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .padding(24)
                            .background(Color.green.opacity(0.1))
                            .cornerRadius(16)
                            .overlay(
                                RoundedRectangle(cornerRadius: 16)
                                    .stroke(Color.green, lineWidth: 2)
                            )
                        }
                        .padding(.horizontal)

                        // Warning about screenshots
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.subheadline)
                            Text(NSLocalizedString("photoRoom.screenshotWarning", comment: ""))
                                .font(.subheadline)
                        }
                        .foregroundColor(.red)
                        .padding(.top, 12)

                        Spacer()
                    }
                }
            }

            // Progress overlay for ODR downloading
            if sharpService.isDownloadingResources {
                VStack(spacing: 16) {
                    ZStack {
                        Circle()
                            .stroke(Color.purple.opacity(0.3), lineWidth: 8)
                            .frame(width: 60, height: 60)
                        Circle()
                            .trim(from: 0, to: sharpService.downloadProgress)
                            .stroke(Color.purple, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                            .frame(width: 60, height: 60)
                            .rotationEffect(.degrees(-90))
                        Image(systemName: "arrow.down.circle.fill")
                            .font(.system(size: 24))
                            .foregroundColor(.purple)
                    }

                    Text(sharpService.statusMessage)
                        .font(.headline)
                        .foregroundColor(.primary)

                    Text("\(Int(sharpService.downloadProgress * 100))%")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.purple)

                    Text("One-time download (~1.2 GB)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(32)
                .background(Color(.systemBackground).opacity(0.95))
                .cornerRadius(16)
                .shadow(radius: 10)
            }

            // Progress overlay for model loading
            if sharpService.isLoadingModel && !sharpService.isDownloadingResources {
                VStack(spacing: 16) {
                    ProgressView()
                        .scaleEffect(1.5)
                        .tint(.purple)

                    Text(sharpService.statusMessage)
                        .font(.headline)
                        .foregroundColor(.primary)

                    ProgressView(value: Double(sharpService.progress))
                        .progressViewStyle(LinearProgressViewStyle(tint: .purple))
                        .frame(width: 200)
                }
                .padding(32)
                .background(Color(.systemBackground).opacity(0.95))
                .cornerRadius(16)
                .shadow(radius: 10)
            }

            // Progress overlay for on-device SHARP generation
            if case .processing = sharpService.status {
                GenerationProgressOverlay(
                    status: sharpService.status,
                    uploadProgress: sharpService.progress,
                    downloadProgress: sharpService.progress,
                    statusMessage: sharpService.statusMessage,
                    onCancel: { sharpService.cancelGeneration() }
                )
            }

            // Progress overlay for manual room reconstruction
            if reconstructor.isProcessing {
                VStack(spacing: 16) {
                    ZStack {
                        Circle()
                            .stroke(Color.orange.opacity(0.3), lineWidth: 8)
                            .frame(width: 60, height: 60)
                        Circle()
                            .trim(from: 0, to: CGFloat(reconstructor.progress))
                            .stroke(Color.orange, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                            .frame(width: 60, height: 60)
                            .rotationEffect(.degrees(-90))
                            .animation(.easeInOut(duration: 0.3), value: reconstructor.progress)
                        Image(systemName: "cube.fill")
                            .font(.system(size: 24))
                            .foregroundColor(.orange)
                    }

                    Text(reconstructor.statusMessage)
                        .font(.headline)
                        .foregroundColor(.primary)

                    Text("\(Int(reconstructor.progress * 100))%")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.orange)

                    Text(NSLocalizedString("photoRoom.buildingRoom", comment: "Building your 3D room"))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(32)
                .background(Color(.systemBackground).opacity(0.95))
                .cornerRadius(16)
                .shadow(radius: 10)
            }
        }
        .navigationTitle("Photo to 3D Room")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showImagePicker) {
            PhotoPickerView(selectedImage: $selectedImage)
                .onDisappear {
                    logDebug("📱 [View] Image picker dismissed")
                    if selectedImage != nil {
                        logDebug("✅ [View] Image selected, showing method picker...")
                        showMethodPicker = true
                    } else {
                        logDebug("⚠️ [View] No image selected")
                    }
                }
        }
        .sheet(isPresented: $showCameraCapture) {
            CameraCaptureView(selectedImage: $selectedImage, selectedOrientation: $captureOrientation)
                .onDisappear {
                    logDebug("📷 [View] Camera capture dismissed")
                    if selectedImage != nil {
                        logDebug("✅ [View] Photo captured with orientation: \(captureOrientation.rawValue), showing method picker...")
                        showMethodPicker = true
                    } else {
                        logDebug("⚠️ [View] No photo captured")
                    }
                }
        }
        .onChange(of: selectedImage) { oldValue, newValue in
            guard let image = newValue else { return }
            logDebug("✅ [View] Image selected")
            // Store the fixed image for later use
            fixedImage = image
            // Auto-detect orientation and pre-select it (user can override)
            let detectedOrientation = PhotoOrientation.detect(from: image)
            selectedOrientation = detectedOrientation
            logDebug("📐 [View] Auto-detected orientation: \(detectedOrientation.rawValue)")
        }
        .sheet(item: $fixedImageItem) { item in
            RoomBoundaryDetectionView(
                originalImage: item.image,
                savedBoundaries: $adjustedBoundaries,
                reconstructor: reconstructor,
                roomDimensions: SinglePhotoRoomReconstructor.RoomDimensions(
                    width: Float(roomWidth),
                    depth: Float(roomDepth),
                    height: Float(roomHeight)
                ),
                onProcessingComplete: {
                    // Navigate to viewer when processing is complete
                    if reconstructor.generatedRoomScene != nil {
                        navigateToViewer = true
                    }
                }
            )
            .onAppear {
                logDebug("✅ [Sheet] Opening RoomBoundaryDetectionView with image: \(item.image.size)")
            }
        }
        .onAppear {
            logDebug("👁️ [View] SinglePhotoRoomView appeared")
            // Reset navigation state on appear to clear any stale state
            if navigateToSplatViewer && generatedPLYURL == nil {
                logDebug("   Resetting stale navigateToSplatViewer state")
                navigateToSplatViewer = false
            }
            // Dimensions are now managed by @AppStorage
        }
        // ✅ Watch for boundary changes - processing is now done in the sheet
        // This handler just navigates if the scene is ready (from sheet processing)
        .onChange(of: adjustedBoundaries) { oldValue, newValue in
            logDebug("📋 [View] adjustedBoundaries onChange triggered")
            logDebug("   oldValue: \(oldValue != nil ? "set" : "nil")")
            logDebug("   newValue: \(newValue != nil ? "set" : "nil")")
            // Processing is now handled in RoomBoundaryDetectionView with progress overlay
            // Navigation is triggered by onProcessingComplete callback
            if newValue != nil && reconstructor.generatedRoomScene != nil {
                logDebug("✅ [View] Scene ready from sheet processing, navigating to viewer")
                navigateToViewer = true
            }
        }
        // Programmatic navigation using the modern API (iOS 16+).  When
        // `navigateToViewer` is set to true, a destination is pushed onto
        // the navigation stack.  We wrap the destination in a `Group` to
        // handle the optional room scene gracefully.
        .navigationDestination(isPresented: $navigateToViewer) {
            Group {
                if let scene = reconstructor.generatedRoomScene {
                    SceneKitViewer(scene: scene)
                }
            }
        }
        // Navigate to SharpRoomView when PLY is generated (used for both orientations)
        .navigationDestination(isPresented: $navigateToSplatViewer) {
            // Only navigate if we have a valid PLY URL - otherwise show empty view
            // (navigation should be prevented by the onChange guard below)
            if let plyURL = generatedPLYURL {
                SharpRoomView(
                    plyURL: plyURL,
                    roomMeasurements: generatedRoomMeasurements,
                    photoOrientation: selectedOrientation
                )
                .onAppear {
                    logDebug("🚀 [Navigation] SharpRoomView appeared")
                    logDebug("   orientation = \(selectedOrientation.rawValue)")
                    logDebug("   plyURL = \(plyURL.lastPathComponent)")
                }
            } else {
                // Fallback - should not happen if navigation is properly guarded
                Color.clear.onAppear {
                    logDebug("⚠️ [Navigation] navigateToSplatViewer=true but generatedPLYURL is nil - resetting")
                    navigateToSplatViewer = false
                }
            }
        }
        // Guard: only allow navigation when URL is set
        .onChange(of: navigateToSplatViewer) { oldValue, newValue in
            if newValue && generatedPLYURL == nil {
                logDebug("⚠️ [Navigation] Blocking navigation - generatedPLYURL is nil")
                navigateToSplatViewer = false
            }
        }
        // Success alert for API-generated PLY file
        .alert("3D Model Generated", isPresented: $showGenerationSuccess) {
            Button("Done") {
                // Dismiss the sheet and notify home to refresh
                NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil)
            }
        } message: {
            if let url = generatedPLYURL {
                let fileName = url.lastPathComponent
                Text("Successfully downloaded \(fileName). View it in your models list.")
            } else {
                Text("Your 3D model has been saved successfully.")
            }
        }
        // Handle generation errors
        .alert("Generation Failed", isPresented: Binding(
            get: {
                if case .failed = sharpService.status { return true }
                return false
            },
            set: { _ in }
        )) {
            Button("OK", role: .cancel) {
                selectedImage = nil
            }
            Button("Retry") {
                if let image = selectedImage {
                    startSHARPGeneration(image: image)
                }
            }
        } message: {
            if case .failed(let errorMessage) = sharpService.status {
                Text(errorMessage)
            } else {
                Text("An error occurred while generating your 3D model.")
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

    private func startSHARPGeneration(image: UIImage) {
        let orientation = selectedOrientation  // Capture current selection
        logDebug("🤖 [View] Starting on-device SHARP generation with orientation: \(orientation.rawValue)")

        // Clear previous generation state to prevent using stale data on failure
        generatedPLYURL = nil
        generatedRoomMeasurements = nil
        navigateToSplatViewer = false  // Reset navigation state

        Task {
            do {
                let fileURL: URL
                let measurements: RoomMeasurements?

                // Use SHARPService for both orientations
                fileURL = try await sharpService.generateGaussians(from: image)
                measurements = sharpService.roomMeasurements

                logDebug("✅ [View] PLY file generated: \(fileURL.path)")
                await MainActor.run {
                    generatedPLYURL = fileURL
                    generatedRoomMeasurements = measurements
                    navigateToSplatViewer = true
                }
            } catch {
                logDebug("❌ [View] Generation failed: \(error)")
            }
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

// MARK: - Photo Orientation Enum
enum CaptureOrientation: String, CaseIterable {
    case portrait = "Portrait"
    case landscape = "Landscape"
    case wideAngle = "Wide Angle"

    var icon: String {
        switch self {
        case .portrait: return "rectangle.portrait"
        case .landscape: return "rectangle"
        case .wideAngle: return "camera.filters"
        }
    }

    var description: String {
        switch self {
        case .portrait: return NSLocalizedString("camera.portrait.desc", comment: "Best for narrow rooms")
        case .landscape: return NSLocalizedString("camera.landscape.desc", comment: "Best for wide rooms")
        case .wideAngle: return NSLocalizedString("camera.wideAngle.desc", comment: "Ultra-wide 0.5x lens")
        }
    }

    var localizationKey: String {
        switch self {
        case .portrait: return "camera.portrait"
        case .landscape: return "camera.landscape"
        case .wideAngle: return "camera.wideAngle"
        }
    }
}

// MARK: - Camera Capture View with Orientation Selection
struct CameraCaptureView: View {
    @Binding var selectedImage: UIImage?
    @Binding var selectedOrientation: CaptureOrientation
    @Environment(\.dismiss) var dismiss

    @State private var showCamera = false
    @State private var capturedImage: UIImage?
    @State private var showWideAngleGuide = false
    @State private var showPhotoPicker = false
    @State private var showWideAngleCamera = false

    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // Orientation Selection Header
                VStack(spacing: 8) {
                    Image(systemName: "camera.viewfinder")
                        .font(.system(size: 50))
                        .foregroundColor(.blue)

                    Text(NSLocalizedString("camera.chooseOrientation", comment: "Choose Photo Orientation"))
                        .font(.title2)
                        .fontWeight(.bold)

                    Text(NSLocalizedString("camera.orientationHint", comment: "Select how you want to capture your room"))
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                .padding(.top, 20)

                // Orientation Options
                VStack(spacing: 12) {
                    ForEach(CaptureOrientation.allCases, id: \.self) { orientation in
                        OrientationOptionButton(
                            orientation: orientation,
                            isSelected: selectedOrientation == orientation,
                            action: { selectedOrientation = orientation }
                        )
                    }
                }
                .padding(.horizontal)

                // Wide angle info banner when wide angle is selected
                if selectedOrientation == .wideAngle {
                    HStack(spacing: 12) {
                        Image(systemName: "info.circle.fill")
                            .foregroundColor(.orange)
                        Text(NSLocalizedString("camera.wideAngle.info", comment: "Uses ultra-wide 0.5x lens"))
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(10)
                    .padding(.horizontal)
                }

                Spacer()

                // Capture Button - different action for panoramic
                if selectedOrientation == .wideAngle {
                    VStack(spacing: 12) {
                        // Capture with ultra-wide camera
                        Button(action: {
                            logDebug("📷 [Camera] Opening wide-angle camera")
                            showWideAngleCamera = true
                        }) {
                            HStack(spacing: 12) {
                                Image(systemName: "camera.filters")
                                    .font(.title2)
                                Text(NSLocalizedString("camera.captureWideAngle", comment: "Capture Wide Photo"))
                                    .font(.headline)
                            }
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(Color.orange)
                            .cornerRadius(12)
                        }

                        // Select from library button
                        Button(action: {
                            logDebug("📷 [Camera] Opening photo picker for wide-angle selection")
                            showPhotoPicker = true
                        }) {
                            HStack(spacing: 12) {
                                Image(systemName: "photo.on.rectangle")
                                    .font(.title2)
                                Text(NSLocalizedString("camera.selectWideAngle", comment: "Select Wide Photo"))
                                    .font(.headline)
                            }
                            .foregroundColor(.orange)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(Color.orange.opacity(0.15))
                            .cornerRadius(12)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 20)
                } else {
                    // Regular camera button for portrait/landscape
                    Button(action: {
                        logDebug("📷 [Camera] Opening camera with orientation: \(selectedOrientation.rawValue)")
                        showCamera = true
                    }) {
                        HStack(spacing: 12) {
                            Image(systemName: "camera.fill")
                                .font(.title2)
                            Text(NSLocalizedString("camera.takePhoto", comment: "Take Photo"))
                                .font(.headline)
                        }
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(Color.blue)
                        .cornerRadius(12)
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 20)
                }
            }
            .navigationTitle(NSLocalizedString("camera.title", comment: "Camera"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(L10n.Common.cancel) {
                        dismiss()
                    }
                }
            }
            .fullScreenCover(isPresented: $showCamera) {
                CameraViewRepresentable(
                    capturedImage: $capturedImage,
                    orientation: selectedOrientation
                )
                .ignoresSafeArea()
            }
            .fullScreenCover(isPresented: $showWideAngleCamera) {
                WideAngleCameraView(capturedImage: $capturedImage)
                    .ignoresSafeArea()
            }
            .sheet(isPresented: $showPhotoPicker) {
                PhotoLibraryPicker(selectedImage: $capturedImage)
            }
            .onChange(of: capturedImage) { _, newImage in
                if let image = newImage {
                    logDebug("📷 [Camera] Photo captured: \(image.size)")
                    selectedImage = image
                    dismiss()
                }
            }
        }
    }
}

// MARK: - Photo Library Picker
struct PhotoLibraryPicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.dismiss) var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .photoLibrary
        picker.delegate = context.coordinator
        picker.allowsEditing = false
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: PhotoLibraryPicker

        init(_ parent: PhotoLibraryPicker) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                logDebug("📷 [PhotoPicker] Selected image: \(image.size)")
                parent.selectedImage = image.fixedOrientation()
            }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

// MARK: - Wide Angle Camera View (AVFoundation-based with Ultra-Wide Lens)
import AVFoundation

struct WideAngleCameraView: UIViewControllerRepresentable {
    @Binding var capturedImage: UIImage?
    @Environment(\.dismiss) var dismiss

    func makeUIViewController(context: Context) -> WideAngleCameraViewController {
        let controller = WideAngleCameraViewController()
        controller.delegate = context.coordinator
        return controller
    }

    func updateUIViewController(_ uiViewController: WideAngleCameraViewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, WideAngleCameraDelegate {
        let parent: WideAngleCameraView

        init(_ parent: WideAngleCameraView) {
            self.parent = parent
        }

        func wideAngleCameraDidCapture(_ image: UIImage) {
            logDebug("📷 [WideAngle] Captured image: \(image.size)")
            parent.capturedImage = image.fixedOrientation()
            parent.dismiss()
        }

        func wideAngleCameraDidCancel() {
            logDebug("📷 [WideAngle] User cancelled")
            parent.dismiss()
        }
    }
}

protocol WideAngleCameraDelegate: AnyObject {
    func wideAngleCameraDidCapture(_ image: UIImage)
    func wideAngleCameraDidCancel()
}

class WideAngleCameraViewController: UIViewController {
    weak var delegate: WideAngleCameraDelegate?

    private var captureSession: AVCaptureSession?
    private var photoOutput: AVCapturePhotoOutput?
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var currentDevice: AVCaptureDevice?

    // UI Elements
    private let captureButton = UIButton(type: .system)
    private let cancelButton = UIButton(type: .system)
    private let guideLabel = UILabel()
    private let zoomLabel = UILabel()
    private let gridOverlay = UIView()

    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        return .landscape
    }

    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation {
        return .landscapeRight
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        setupCamera()
        setupUI()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
        updateGridOverlay()
    }

    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .photo

        // Try to get ultra-wide camera first for wider field of view
        var device: AVCaptureDevice?

        // Check for ultra-wide camera (0.5x zoom equivalent)
        if let ultraWide = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back) {
            device = ultraWide
            logDebug("📷 [WideAngle] Using ultra-wide camera for wide-angle capture")
        } else if let wideAngle = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) {
            device = wideAngle
            logDebug("📷 [WideAngle] Using wide-angle camera (ultra-wide not available)")
        }

        guard let captureDevice = device else {
            logDebug("❌ [WideAngle] No camera available")
            return
        }

        currentDevice = captureDevice

        do {
            let input = try AVCaptureDeviceInput(device: captureDevice)
            if captureSession?.canAddInput(input) == true {
                captureSession?.addInput(input)
            }

            photoOutput = AVCapturePhotoOutput()
            if let photoOutput = photoOutput, captureSession?.canAddOutput(photoOutput) == true {
                captureSession?.addOutput(photoOutput)

                // Configure for high resolution using maxPhotoDimensions (iOS 16+)
                photoOutput.maxPhotoDimensions = captureDevice.activeFormat.supportedMaxPhotoDimensions.first ?? CMVideoDimensions(width: 4032, height: 3024)
            }

            previewLayer = AVCaptureVideoPreviewLayer(session: captureSession!)
            previewLayer?.videoGravity = .resizeAspectFill
            previewLayer?.frame = view.bounds

            if let previewLayer = previewLayer {
                view.layer.addSublayer(previewLayer)
            }

            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.captureSession?.startRunning()
            }

        } catch {
            logDebug("❌ [WideAngle] Camera setup error: \(error)")
        }
    }

    private func setupUI() {
        // Guide label at top
        guideLabel.text = NSLocalizedString("camera.wideAngle.holdSteady", comment: "Hold steady and capture wide view")
        guideLabel.textColor = .white
        guideLabel.font = .systemFont(ofSize: 16, weight: .medium)
        guideLabel.textAlignment = .center
        guideLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        guideLabel.layer.cornerRadius = 8
        guideLabel.clipsToBounds = true
        guideLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(guideLabel)

        // Zoom indicator
        let isUltraWide = currentDevice?.deviceType == .builtInUltraWideCamera
        zoomLabel.text = isUltraWide ? "0.5x Ultra Wide" : "1x Wide"
        zoomLabel.textColor = .yellow
        zoomLabel.font = .systemFont(ofSize: 14, weight: .bold)
        zoomLabel.textAlignment = .center
        zoomLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(zoomLabel)

        // Grid overlay for composition
        gridOverlay.translatesAutoresizingMaskIntoConstraints = false
        gridOverlay.isUserInteractionEnabled = false
        view.addSubview(gridOverlay)

        // Capture button
        captureButton.setImage(UIImage(systemName: "circle.inset.filled", withConfiguration: UIImage.SymbolConfiguration(pointSize: 70)), for: .normal)
        captureButton.tintColor = .white
        captureButton.translatesAutoresizingMaskIntoConstraints = false
        captureButton.addTarget(self, action: #selector(capturePhoto), for: .touchUpInside)
        view.addSubview(captureButton)

        // Cancel button
        cancelButton.setTitle(NSLocalizedString("common.cancel", comment: "Cancel"), for: .normal)
        cancelButton.setTitleColor(.white, for: .normal)
        cancelButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .medium)
        cancelButton.translatesAutoresizingMaskIntoConstraints = false
        cancelButton.addTarget(self, action: #selector(cancelCapture), for: .touchUpInside)
        view.addSubview(cancelButton)

        NSLayoutConstraint.activate([
            guideLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            guideLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            guideLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 200),
            guideLabel.heightAnchor.constraint(equalToConstant: 36),

            zoomLabel.topAnchor.constraint(equalTo: guideLabel.bottomAnchor, constant: 8),
            zoomLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),

            gridOverlay.topAnchor.constraint(equalTo: view.topAnchor),
            gridOverlay.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            gridOverlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            gridOverlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),

            captureButton.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            captureButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -30),

            cancelButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            cancelButton.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20)
        ])
    }

    private func updateGridOverlay() {
        // Remove existing grid lines
        gridOverlay.layer.sublayers?.forEach { $0.removeFromSuperlayer() }

        let bounds = gridOverlay.bounds
        let lineColor = UIColor.white.withAlphaComponent(0.3).cgColor

        // Horizontal lines (rule of thirds)
        for i in 1...2 {
            let y = bounds.height * CGFloat(i) / 3
            let line = CAShapeLayer()
            let path = UIBezierPath()
            path.move(to: CGPoint(x: 0, y: y))
            path.addLine(to: CGPoint(x: bounds.width, y: y))
            line.path = path.cgPath
            line.strokeColor = lineColor
            line.lineWidth = 1
            gridOverlay.layer.addSublayer(line)
        }

        // Vertical lines (rule of thirds)
        for i in 1...2 {
            let x = bounds.width * CGFloat(i) / 3
            let line = CAShapeLayer()
            let path = UIBezierPath()
            path.move(to: CGPoint(x: x, y: 0))
            path.addLine(to: CGPoint(x: x, y: bounds.height))
            line.path = path.cgPath
            line.strokeColor = lineColor
            line.lineWidth = 1
            gridOverlay.layer.addSublayer(line)
        }

        // Center horizontal guide line (yellow)
        let centerLine = CAShapeLayer()
        let centerPath = UIBezierPath()
        let centerY = bounds.height / 2
        centerPath.move(to: CGPoint(x: bounds.width * 0.3, y: centerY))
        centerPath.addLine(to: CGPoint(x: bounds.width * 0.7, y: centerY))
        centerLine.path = centerPath.cgPath
        centerLine.strokeColor = UIColor.yellow.withAlphaComponent(0.6).cgColor
        centerLine.lineWidth = 2
        centerLine.lineDashPattern = [10, 5]
        gridOverlay.layer.addSublayer(centerLine)
    }

    @objc private func capturePhoto() {
        guard let photoOutput = photoOutput else { return }

        let settings = AVCapturePhotoSettings()
        // Use maxPhotoDimensions from photoOutput (iOS 16+)
        settings.maxPhotoDimensions = photoOutput.maxPhotoDimensions

        // Flash off for wide-angle captures (usually room interiors)
        settings.flashMode = .off

        photoOutput.capturePhoto(with: settings, delegate: self)

        // Visual feedback
        UIView.animate(withDuration: 0.1) {
            self.view.alpha = 0.5
        } completion: { _ in
            UIView.animate(withDuration: 0.1) {
                self.view.alpha = 1.0
            }
        }
    }

    @objc private func cancelCapture() {
        captureSession?.stopRunning()
        delegate?.wideAngleCameraDidCancel()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        captureSession?.stopRunning()
    }
}

extension WideAngleCameraViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            logDebug("❌ [WideAngle] Photo capture error: \(error)")
            return
        }

        guard let imageData = photo.fileDataRepresentation(),
              let image = UIImage(data: imageData) else {
            logDebug("❌ [WideAngle] Failed to create image from photo data")
            return
        }

        logDebug("✅ [WideAngle] Photo captured: \(image.size)")
        captureSession?.stopRunning()
        delegate?.wideAngleCameraDidCapture(image)
    }
}

// MARK: - Orientation Option Button
struct OrientationOptionButton: View {
    let orientation: CaptureOrientation
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 16) {
                // Orientation icon
                Image(systemName: orientation.icon)
                    .font(.system(size: 28))
                    .foregroundColor(isSelected ? .blue : .secondary)
                    .frame(width: 50)

                VStack(alignment: .leading, spacing: 4) {
                    Text(NSLocalizedString(orientation.localizationKey, comment: ""))
                        .font(.headline)
                        .foregroundColor(.primary)
                    Text(orientation.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                // Selection indicator
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.title2)
                    .foregroundColor(isSelected ? .blue : .secondary)
            }
            .padding()
            .background(isSelected ? Color.blue.opacity(0.1) : Color(.systemGray6))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 2)
            )
        }
    }
}

// MARK: - Oriented Camera View (AVFoundation-based with forced orientation)
struct CameraViewRepresentable: UIViewControllerRepresentable {
    @Binding var capturedImage: UIImage?
    let orientation: CaptureOrientation
    @Environment(\.dismiss) var dismiss

    func makeUIViewController(context: Context) -> OrientedCameraViewController {
        logDebug("📷 [Camera] Creating oriented camera for \(orientation.rawValue) mode")
        let controller = OrientedCameraViewController(captureOrientation: orientation)
        controller.delegate = context.coordinator
        return controller
    }

    func updateUIViewController(_ uiViewController: OrientedCameraViewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, OrientedCameraDelegate {
        let parent: CameraViewRepresentable

        init(_ parent: CameraViewRepresentable) {
            self.parent = parent
            logDebug("📷 [Camera] Coordinator initialized for \(parent.orientation.rawValue) mode")
        }

        func orientedCameraDidCapture(_ image: UIImage) {
            logDebug("📷 [Camera] Photo captured: \(image.size)")
            parent.capturedImage = image.fixedOrientation()
            parent.dismiss()
        }

        func orientedCameraDidCancel() {
            logDebug("📷 [Camera] User cancelled")
            parent.dismiss()
        }
    }
}

protocol OrientedCameraDelegate: AnyObject {
    func orientedCameraDidCapture(_ image: UIImage)
    func orientedCameraDidCancel()
}

class OrientedCameraViewController: UIViewController {
    weak var delegate: OrientedCameraDelegate?
    let captureOrientation: CaptureOrientation

    private var captureSession: AVCaptureSession?
    private var photoOutput: AVCapturePhotoOutput?
    private var previewLayer: AVCaptureVideoPreviewLayer?

    // UI Elements
    private let captureButton = UIButton(type: .system)
    private let cancelButton = UIButton(type: .system)
    private let orientationLabel = UILabel()
    private let gridOverlay = UIView()

    init(captureOrientation: CaptureOrientation) {
        self.captureOrientation = captureOrientation
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        switch captureOrientation {
        case .portrait:
            return .portrait
        case .landscape:
            return .landscape
        case .wideAngle:
            return .landscape
        }
    }

    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation {
        switch captureOrientation {
        case .portrait:
            return .portrait
        case .landscape, .wideAngle:
            return .landscapeRight
        }
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        setupCamera()
        setupUI()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
        updateGridOverlay()
    }

    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .photo

        // Use standard wide-angle camera for portrait/landscape
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            logDebug("❌ [OrientedCamera] No camera available")
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession?.canAddInput(input) == true {
                captureSession?.addInput(input)
            }

            photoOutput = AVCapturePhotoOutput()
            if let photoOutput = photoOutput, captureSession?.canAddOutput(photoOutput) == true {
                captureSession?.addOutput(photoOutput)
                // Configure for high resolution using maxPhotoDimensions (iOS 16+)
                photoOutput.maxPhotoDimensions = device.activeFormat.supportedMaxPhotoDimensions.first ?? CMVideoDimensions(width: 4032, height: 3024)
            }

            previewLayer = AVCaptureVideoPreviewLayer(session: captureSession!)
            previewLayer?.videoGravity = .resizeAspectFill
            previewLayer?.frame = view.bounds

            if let previewLayer = previewLayer {
                view.layer.addSublayer(previewLayer)
            }

            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.captureSession?.startRunning()
            }

            logDebug("📷 [OrientedCamera] Camera setup complete for \(captureOrientation.rawValue)")

        } catch {
            logDebug("❌ [OrientedCamera] Camera setup error: \(error)")
        }
    }

    private func setupUI() {
        // Orientation label at top
        let modeText = captureOrientation == .portrait ?
            NSLocalizedString("camera.portrait.mode", comment: "Portrait Mode") :
            NSLocalizedString("camera.landscape.mode", comment: "Landscape Mode")
        orientationLabel.text = modeText
        orientationLabel.textColor = .white
        orientationLabel.font = .systemFont(ofSize: 16, weight: .medium)
        orientationLabel.textAlignment = .center
        orientationLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        orientationLabel.layer.cornerRadius = 8
        orientationLabel.clipsToBounds = true
        orientationLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(orientationLabel)

        // Grid overlay for composition
        gridOverlay.translatesAutoresizingMaskIntoConstraints = false
        gridOverlay.isUserInteractionEnabled = false
        view.addSubview(gridOverlay)

        // Capture button
        captureButton.setImage(UIImage(systemName: "circle.inset.filled", withConfiguration: UIImage.SymbolConfiguration(pointSize: 70)), for: .normal)
        captureButton.tintColor = .white
        captureButton.translatesAutoresizingMaskIntoConstraints = false
        captureButton.addTarget(self, action: #selector(capturePhoto), for: .touchUpInside)
        view.addSubview(captureButton)

        // Cancel button
        cancelButton.setTitle(NSLocalizedString("common.cancel", comment: "Cancel"), for: .normal)
        cancelButton.setTitleColor(.white, for: .normal)
        cancelButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .medium)
        cancelButton.translatesAutoresizingMaskIntoConstraints = false
        cancelButton.addTarget(self, action: #selector(cancelCapture), for: .touchUpInside)
        view.addSubview(cancelButton)

        // Layout depends on orientation
        if captureOrientation == .portrait {
            NSLayoutConstraint.activate([
                orientationLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
                orientationLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
                orientationLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 150),
                orientationLabel.heightAnchor.constraint(equalToConstant: 36),

                gridOverlay.topAnchor.constraint(equalTo: view.topAnchor),
                gridOverlay.bottomAnchor.constraint(equalTo: view.bottomAnchor),
                gridOverlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
                gridOverlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),

                captureButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -30),
                captureButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),

                cancelButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -40),
                cancelButton.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 30)
            ])
        } else {
            // Landscape layout
            NSLayoutConstraint.activate([
                orientationLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
                orientationLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
                orientationLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 150),
                orientationLabel.heightAnchor.constraint(equalToConstant: 36),

                gridOverlay.topAnchor.constraint(equalTo: view.topAnchor),
                gridOverlay.bottomAnchor.constraint(equalTo: view.bottomAnchor),
                gridOverlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
                gridOverlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),

                captureButton.centerYAnchor.constraint(equalTo: view.centerYAnchor),
                captureButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -30),

                cancelButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
                cancelButton.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20)
            ])
        }
    }

    private func updateGridOverlay() {
        gridOverlay.layer.sublayers?.forEach { $0.removeFromSuperlayer() }

        let bounds = gridOverlay.bounds
        let lineColor = UIColor.white.withAlphaComponent(0.3).cgColor

        // Rule of thirds grid
        for i in 1...2 {
            // Horizontal lines
            let y = bounds.height * CGFloat(i) / 3
            let hLine = CAShapeLayer()
            let hPath = UIBezierPath()
            hPath.move(to: CGPoint(x: 0, y: y))
            hPath.addLine(to: CGPoint(x: bounds.width, y: y))
            hLine.path = hPath.cgPath
            hLine.strokeColor = lineColor
            hLine.lineWidth = 1
            gridOverlay.layer.addSublayer(hLine)

            // Vertical lines
            let x = bounds.width * CGFloat(i) / 3
            let vLine = CAShapeLayer()
            let vPath = UIBezierPath()
            vPath.move(to: CGPoint(x: x, y: 0))
            vPath.addLine(to: CGPoint(x: x, y: bounds.height))
            vLine.path = vPath.cgPath
            vLine.strokeColor = lineColor
            vLine.lineWidth = 1
            gridOverlay.layer.addSublayer(vLine)
        }
    }

    @objc private func capturePhoto() {
        guard let photoOutput = photoOutput else { return }

        let settings = AVCapturePhotoSettings()
        // Use maxPhotoDimensions from photoOutput (iOS 16+)
        settings.maxPhotoDimensions = photoOutput.maxPhotoDimensions
        settings.flashMode = .off

        photoOutput.capturePhoto(with: settings, delegate: self)

        // Visual feedback
        UIView.animate(withDuration: 0.1) {
            self.view.alpha = 0.5
        } completion: { _ in
            UIView.animate(withDuration: 0.1) {
                self.view.alpha = 1.0
            }
        }
    }

    @objc private func cancelCapture() {
        captureSession?.stopRunning()
        delegate?.orientedCameraDidCancel()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        captureSession?.stopRunning()
    }
}

extension OrientedCameraViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            logDebug("❌ [OrientedCamera] Photo capture error: \(error)")
            return
        }

        guard let imageData = photo.fileDataRepresentation(),
              let image = UIImage(data: imageData) else {
            logDebug("❌ [OrientedCamera] Failed to create image from photo data")
            return
        }

        logDebug("✅ [OrientedCamera] Photo captured: \(image.size)")
        captureSession?.stopRunning()
        delegate?.orientedCameraDidCapture(image)
    }
}

// MARK: - SceneKit Viewer
struct SceneKitViewer: View {
    let scene: SCNScene
    @Environment(\.dismiss) private var dismiss
    @State private var showControls = true
    @State private var cameraNode: SCNNode?

    // Loading state for initial setup
    @State private var isSettingUp = true

    // Save room state - lazy initialization to avoid loading on appear
    @State private var modelManager: USDZModelManager?
    @State private var isSavingRoom = false
    @State private var saveProgress: Double = 0.0
    @State private var savingTimer: Timer?
    @State private var showSaveAlert = false
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false
    @State private var showRoomNameInput = false
    @State private var roomName = ""

    // SmartyPants and screenshot state
    @State private var showingSmartyPants = false
    @State private var mlModel: MLModel? = nil
    @State private var showScreenshotFlash = false

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
                Task {
                    setupCamera()
                    // Small delay to ensure camera is ready
                    try? await Task.sleep(nanoseconds: 300_000_000)
                    await MainActor.run {
                        isSettingUp = false
                    }
                    // Load ML model in background (not blocking)
                    loadMLModel()
                }
            }
            .onDisappear {
                GlobalCameraController.shared.clearCamera()
            }

            // Loading overlay while setting up
            if isSettingUp {
                ZStack {
                    Color.black.opacity(0.8)
                        .ignoresSafeArea()

                    VStack(spacing: 16) {
                        ProgressView()
                            .scaleEffect(1.5)
                            .tint(.orange)

                        Text("Loading 3D Room...")
                            .font(.headline)
                            .foregroundColor(.white)
                    }
                    .padding(32)
                    .background(Color(.systemBackground).opacity(0.95))
                    .cornerRadius(16)
                }
                .transition(.opacity)
            }

            // Save progress overlay
            if isSavingRoom {
                saveRoomProgressOverlay
            }

            // Screenshot flash effect
            if showScreenshotFlash {
                Color.white
                    .ignoresSafeArea()
                    .transition(.opacity)
            }

            // ✅ GLOBAL JOYSTICK - uses GlobalCameraController
            SimpleJoystickOverlay(photoOrientation: .portrait)
                .allowsHitTesting(true)
                .zIndex(99996)

            // SmartyPants overlay when active
            if showingSmartyPants {
                SmartyPantsUIView(
                    capturedImage: .constant(nil),
                    roomImage: nil,
                    mlModel: mlModel,
                    processInterval: 0.07,
                    active: true
                )
                .ignoresSafeArea()
                .zIndex(99997)
            }

            // Bottom row buttons (Brain + Screenshot) - HIGHEST zIndex to stay on top of SmartyPants
            VStack {
                Spacer()
                    .allowsHitTesting(false)
                HStack {
                    // Brain button (bottom-left)
                    Button(action: {
                        showingSmartyPants.toggle()
                    }) {
                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 28))
                            .foregroundColor(.white)
                            .frame(width: 60, height: 60)
                            .background(Circle().fill(showingSmartyPants ? Color.green : Color.blue).shadow(radius: 5))
                    }
                    .padding(.leading, 20)

                    Spacer()
                        .allowsHitTesting(false)

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
                    .padding(.trailing, 20)
                }
                .padding(.bottom, 30)
            }
            .zIndex(99999)

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

        // Lazy-initialize modelManager only when saving
        if modelManager == nil {
            modelManager = USDZModelManager()
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
                self.modelManager?.saveRoom(scene: scene, name: savedName) { success, error in
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

    // MARK: - Screenshot
    private func takeScreenshot() {
        logDebug("📸 Taking screenshot...")

        // Capture the window
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let windows = scenes.flatMap { $0.windows }
        guard let window = windows.first(where: { $0.isKeyWindow }) ?? windows.first else {
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

        // Save to photos
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        logDebug("✅ Screenshot saved to Photos")

        // Show flash effect
        withAnimation(.easeIn(duration: 0.1)) {
            showScreenshotFlash = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            withAnimation(.easeOut(duration: 0.2)) {
                showScreenshotFlash = false
            }
        }
    }

    // MARK: - ML Model Loading
    private func loadMLModel() {
        Task {
            do {
                if let modelURL = Bundle.main.url(forResource: "yoloe-11l-seg-pf", withExtension: "mlmodelc") {
                    let config = MLModelConfiguration()
                    config.computeUnits = .cpuAndNeuralEngine
                    let model = try MLModel(contentsOf: modelURL, configuration: config)
                    await MainActor.run {
                        self.mlModel = model
                    }
                    logDebug("✅ [SceneKitViewer] ML model loaded")
                }
            } catch {
                logDebug("❌ [SceneKitViewer] Failed to load ML model: \(error)")
            }
        }
    }
}

