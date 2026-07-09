import SwiftUI
import SceneKit
import Accelerate
import CoreML
import Photos
import PhotosUI
import UniformTypeIdentifiers

// MARK: - Room Boundary Detection View with DRAGGABLE boundaries
struct RoomBoundaryDetectionView: View {
    let originalImage: UIImage
    @Binding var savedBoundaries: RoomStructure?
    // Optional: pass reconstructor for in-view processing
    @ObservedObject var reconstructor: SinglePhotoRoomReconstructor
    var roomDimensions: SinglePhotoRoomReconstructor.RoomDimensions?
    var onProcessingComplete: (() -> Void)?
    var photoOrientation: PhotoOrientation = .portrait

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

    private var isLandscape: Bool {
        photoOrientation == .landscape
    }
    
    var body: some View {
        NavigationView {
            GeometryReader { outerGeometry in
                let isLandscapeScreen = outerGeometry.size.width > outerGeometry.size.height

                if isLandscapeScreen {
                    // Landscape layout: full-screen image with horizontal bottom overlay
                    ZStack {
                        // Black background to fill any gaps
                        Color.black.ignoresSafeArea()

                        // Image area - uses full screen
                        GeometryReader { geometry in
                            ZStack {
                                Image(uiImage: originalImage)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: geometry.size.width, height: geometry.size.height)

                                BoundaryLinesCanvas(
                                    imageSize: originalImage.size,
                                    floorY: floorY,
                                    ceilingY: ceilingY,
                                    leftX: leftX,
                                    rightX: rightX,
                                    vanishingX: vanishingX,
                                    vanishingY: vanishingY
                                )

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
                        .ignoresSafeArea()

                        // Horizontal bottom overlay bar
                        VStack {
                            Spacer()
                            landscapeBottomBar
                        }
                        .ignoresSafeArea(edges: .horizontal)
                    }
                    .ignoresSafeArea()
                } else {
                    // Portrait layout: image on top, controls below
                    VStack(spacing: 0) {
                        GeometryReader { geometry in
                            ZStack {
                                Image(uiImage: originalImage)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: geometry.size.width)

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
                    
                        // Adjustment instructions for portrait
                        portraitControls
                    }
                }
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
                        dismiss()
                    }
                    .disabled(isProcessingInView)
                }
            }
            .overlay {
                if isProcessingInView {
                    progressOverlay
                }
            }
        }
        .interactiveDismissDisabled(isProcessingInView)
    }

    // MARK: - Portrait Controls
    private var portraitControls: some View {
        VStack(spacing: 12) {
            // Orientation label
            HStack(spacing: 6) {
                Image(systemName: isLandscape ? "iphone.landscape" : "iphone")
                    .font(.caption)
                Text(isLandscape
                     ? NSLocalizedString("orientation.heldHorizontally", comment: "")
                     : NSLocalizedString("orientation.heldVertically", comment: ""))
                    .font(.caption2)
                Text("-")
                    .font(.caption2)
                Text(isLandscape
                     ? NSLocalizedString("orientation.landscape", comment: "")
                     : NSLocalizedString("orientation.portrait", comment: ""))
                    .font(.caption2)
                    .fontWeight(.medium)
            }
            .foregroundColor(.white.opacity(0.9))
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.blue.opacity(0.8))
            .cornerRadius(8)
            .padding(.top, 8)

            Text(L10n.Boundary.instructions)
                .font(.headline)

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

            controlButtons
                .padding(.bottom, 8)
        }
        .background(.ultraThinMaterial)
    }

    // MARK: - Landscape Bottom Bar (horizontal overlay)
    private var landscapeBottomBar: some View {
        HStack(spacing: 20) {
            // Orientation badge
            HStack(spacing: 4) {
                Image(systemName: "iphone.landscape")
                    .font(.caption)
                Text(isLandscape ? "Landscape" : "Portrait")
                    .font(.caption)
                    .fontWeight(.medium)
            }
            .foregroundColor(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.blue.opacity(0.9))
            .cornerRadius(8)

            // Color legend - horizontal
            HStack(spacing: 12) {
                HStack(spacing: 4) {
                    Circle().fill(Color.green).frame(width: 10, height: 10)
                    Text("Floor").font(.caption)
                }
                HStack(spacing: 4) {
                    Circle().fill(Color.cyan).frame(width: 10, height: 10)
                    Text("Ceiling").font(.caption)
                }
                HStack(spacing: 4) {
                    Circle().fill(Color.red).frame(width: 10, height: 10)
                    Text("Walls").font(.caption)
                }
                HStack(spacing: 4) {
                    Circle().fill(magentaColor).frame(width: 10, height: 10)
                    Text("Vanish").font(.caption)
                }
            }
            .foregroundColor(.white)

            Spacer()

            // Buttons - horizontal
            HStack(spacing: 12) {
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
                .tint(.white)

                Button(L10n.Common.done) {
                    processBoundaries()
                }
                .buttonStyle(.borderedProminent)
                .disabled(isProcessingInView)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Color.black.opacity(0.7))
    }

    // MARK: - Control Buttons
    private var controlButtons: some View {
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
                processBoundaries()
            }
            .buttonStyle(.borderedProminent)
            .disabled(isProcessingInView)
        }
    }

    // MARK: - Progress Overlay
    private var progressOverlay: some View {
        ZStack(alignment: .bottom) {
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

                Text(L10n.PhotoRoom.buildingRoom)
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            .padding(32)
            .background(Color(.systemBackground).opacity(0.95))
            .cornerRadius(16)
            .shadow(radius: 10)
        }
    }

    // MARK: - Process Boundaries
    private func processBoundaries() {
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

        isProcessingInView = true
        Task {
            let startTime = Date()
            let minimumDisplayTime: TimeInterval = 2.0

            if let dims = roomDimensions {
                await MainActor.run {
                    reconstructor.estimatedDimensions = dims
                }
            }
            await reconstructor.processPhotoWithBoundaries(originalImage, boundaries: boundaries)

            let elapsed = Date().timeIntervalSince(startTime)
            if elapsed < minimumDisplayTime {
                try? await Task.sleep(nanoseconds: UInt64((minimumDisplayTime - elapsed) * 1_000_000_000))
            }

            await MainActor.run {
                savedBoundaries = boundaries
                isProcessingInView = false
            }
            await Task.yield()
            await MainActor.run {
                onProcessingComplete?()
                dismiss()
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

/// Pushes `SplatRoomView` with PLY in **one** state update (avoids `isPresented` building stale destinations).
private struct SplatViewerDestination: Identifiable, Hashable {
    let id: UUID
    let plyURL: URL
    /// Scene-unit AABB from Splat at write time (for `[PLY_BOUNDS] SPLAT_ROOM_COMPARE`).
    let splatPlyW: Float?
    let splatPlyH: Float?
    let splatPlyD: Float?
    let roomWidth: Float?
    let roomHeight: Float?
    let roomDepth: Float?
    let sourcePhotoPxW: Int?
    let sourcePhotoPxH: Int?
    let roomCoordinateFrame: RoomCoordinateFrame

    init(
        plyURL: URL,
        splatPlyAabb: (Float, Float, Float)? = nil,
        roomMeters: (Float, Float, Float)? = nil,
        sourcePhotoPixels: (Int, Int)? = nil,
        roomCoordinateFrame: RoomCoordinateFrame = .classicSplatPly
    ) {
        self.id = UUID()
        self.plyURL = plyURL
        self.roomCoordinateFrame = roomCoordinateFrame
        if let a = splatPlyAabb {
            self.splatPlyW = a.0
            self.splatPlyH = a.1
            self.splatPlyD = a.2
        } else {
            self.splatPlyW = nil
            self.splatPlyH = nil
            self.splatPlyD = nil
        }
        if let roomMeters {
            self.roomWidth = roomMeters.0
            self.roomHeight = roomMeters.1
            self.roomDepth = roomMeters.2
        } else {
            self.roomWidth = nil
            self.roomHeight = nil
            self.roomDepth = nil
        }
        if let p = sourcePhotoPixels {
            self.sourcePhotoPxW = p.0
            self.sourcePhotoPxH = p.1
        } else {
            self.sourcePhotoPxW = nil
            self.sourcePhotoPxH = nil
        }
    }
}

private struct USDZViewerDestination: Identifiable, Hashable {
    let id = UUID()
    let measurementImageURL: URL
    let photoOrientation: PhotoOrientation
    let roomCoordinateFrame: RoomCoordinateFrame
    let summary: String
    /// Depth Anything metric dims at generation — same pattern as ``SplatViewerDestination`` room metres for Splat.
    let roomWidthMeters: Float
    let roomHeightMeters: Float
    let roomDepthMeters: Float
    /// Debug-mode overlay: measurement calibration inputs (camera height, scale, focal source).
    let measurementDebugLine: String?
}

private struct DepthAnythingPreparedPreview: Sendable {
    let measurementImageURL: URL
    let imageWidth: Int
    let imageHeight: Int
    let roomWidthMeters: Float
    let roomHeightMeters: Float
    let roomDepthMeters: Float

    var summary: String {
        String(
            format: "preview image=%dx%d dims=%.2fx%.2fx%.2fm source=%@",
            imageWidth,
            imageHeight,
            roomWidthMeters,
            roomHeightMeters,
            roomDepthMeters,
            measurementImageURL.lastPathComponent
        )
    }
}

private enum DepthAnythingPreviewPrepareError: LocalizedError {
    case invalidImage

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Could not prepare source image for room preview."
        }
    }
}

private func makeDepthAnythingPreviewDestination(
    image: UIImage,
    cameraMetadata: [String: Double],
    photoOrientation _: PhotoOrientation
) throws -> DepthAnythingPreparedPreview {
    let fixed = image.fixedOrientation()
    let previewImage = downsampleDepthAnythingPreviewImage(fixed, maxDimension: 1600)
    guard let data = previewImage.jpegData(compressionQuality: 0.92),
          let cgImage = previewImage.cgImage else {
        throw DepthAnythingPreviewPrepareError.invalidImage
    }

    let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let directory = documentsURL.appendingPathComponent("DepthAnythingRooms", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let url = directory.appendingPathComponent("DepthAnythingPreview_\(UUID().uuidString).jpg")
    try data.write(to: url, options: [.atomic])
    if !cameraMetadata.isEmpty {
        CameraExifSidecar.mergeDerivedValues(roomURL: url, additions: cameraMetadata)
    }

    let width = max(1, cgImage.width)
    let height = max(1, cgImage.height)
    let roomWidth: Float = 2.0
    let roomHeight = max(1.8, roomWidth * Float(height) / Float(width))
    let roomDepth: Float = 3.0
    logDebug(
        "[DepthAnythingRoom][PreviewFast] skipping depth_anything/geocalib/rtmdet/room_height/usdz_export during creation " +
        "image=\(width)x\(height) W=\(String(format: "%.3f", roomWidth)) " +
        "H=\(String(format: "%.3f", roomHeight)) D=\(String(format: "%.3f", roomDepth))"
    )
    return DepthAnythingPreparedPreview(
        measurementImageURL: url,
        imageWidth: width,
        imageHeight: height,
        roomWidthMeters: roomWidth,
        roomHeightMeters: roomHeight,
        roomDepthMeters: roomDepth
    )
}

private func downsampleDepthAnythingPreviewImage(_ image: UIImage, maxDimension: CGFloat) -> UIImage {
    let maxSide = max(image.size.width, image.size.height)
    guard maxSide > maxDimension, maxSide > 0 else { return image }
    let scale = maxDimension / maxSide
    let targetSize = CGSize(
        width: max(1, floor(image.size.width * scale)),
        height: max(1, floor(image.size.height * scale))
    )
    let renderer = UIGraphicsImageRenderer(size: targetSize)
    return renderer.image { _ in
        image.draw(in: CGRect(origin: .zero, size: targetSize))
    }
}

private final class DepthAnythingPreviewSCNView: SCNView {
    var onViewportSizeChanged: ((CGSize) -> Void)?
    private var lastViewportSize: CGSize = .zero

    override func layoutSubviews() {
        super.layoutSubviews()
        let size = bounds.size
        guard size.width > 1, size.height > 1 else { return }
        guard abs(size.width - lastViewportSize.width) > 0.5 ||
                abs(size.height - lastViewportSize.height) > 0.5 else {
            return
        }
        lastViewportSize = size
        onViewportSizeChanged?(size)
    }
}

private struct DepthAnythingPreviewSceneView: UIViewRepresentable {
    let imageURL: URL
    let roomWidthMeters: Float
    let roomHeightMeters: Float
    @Binding var cameraOffset: CGSize
    @Binding var cameraZoom: CGFloat
    @Binding var shouldResetCamera: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> SCNView {
        let view = DepthAnythingPreviewSCNView(frame: .zero)
        view.backgroundColor = UIColor(white: 0.12, alpha: 1.0)
        view.allowsCameraControl = false
        view.autoenablesDefaultLighting = false
        context.coordinator.configure(cameraOffset: $cameraOffset, cameraZoom: $cameraZoom)
        context.coordinator.installGestureRecognizersIfNeeded(on: view)
        view.scene = makeScene(context: context)
        view.pointOfView = context.coordinator.cameraNode
        view.onViewportSizeChanged = { [weak cameraNode = context.coordinator.cameraNode] size in
            applyCameraPose(cameraNode, viewportSize: size)
        }
        applyCameraPose(context.coordinator.cameraNode, viewportSize: view.bounds.size)
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        view.pointOfView = context.coordinator.cameraNode
        context.coordinator.configure(cameraOffset: $cameraOffset, cameraZoom: $cameraZoom)
        if let previewView = view as? DepthAnythingPreviewSCNView {
            previewView.onViewportSizeChanged = { [weak cameraNode = context.coordinator.cameraNode] size in
                applyCameraPose(cameraNode, viewportSize: size)
            }
        }
        if shouldResetCamera {
            DispatchQueue.main.async {
                shouldResetCamera = false
            }
        }
        applyCameraPose(context.coordinator.cameraNode, viewportSize: view.bounds.size)
    }

    private func makeScene(context: Context) -> SCNScene {
        let scene = SCNScene()
        let image = UIImage(contentsOfFile: imageURL.path)
        let width = CGFloat(max(roomWidthMeters, 0.05))
        let height = CGFloat(max(roomHeightMeters, 0.05))
        let plane = SCNPlane(width: width, height: height)
        let material = SCNMaterial()
        material.diffuse.contents = image
        material.emission.contents = image
        material.lightingModel = .constant
        material.isDoubleSided = true
        material.diffuse.wrapS = .clamp
        material.diffuse.wrapT = .clamp
        plane.materials = [material]

        let planeNode = SCNNode(geometry: plane)
        planeNode.name = "DepthAnythingPreviewImagePlane"
        scene.rootNode.addChildNode(planeNode)

        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.fieldOfView = 60
        cameraNode.camera?.zNear = 0.001
        cameraNode.camera?.zFar = Double(max(roomWidthMeters, roomHeightMeters, 1.0) * 12)
        scene.rootNode.addChildNode(cameraNode)
        context.coordinator.cameraNode = cameraNode
        return scene
    }

    private func applyCameraPose(_ cameraNode: SCNNode?, viewportSize: CGSize) {
        guard let cameraNode else { return }
        let planeWidth = CGFloat(max(roomWidthMeters, 0.05))
        let planeHeight = CGFloat(max(roomHeightMeters, 0.05))
        let viewportWidth = max(viewportSize.width, 1)
        let viewportHeight = max(viewportSize.height, 1)
        let aspect = max(viewportWidth / viewportHeight, 0.1)
        let verticalFOV = CGFloat((cameraNode.camera?.fieldOfView ?? 60) * .pi / 180)
        let fitHeightDistance = (planeHeight * 0.5) / tan(verticalFOV * 0.5)
        let fitWidthDistance = (planeWidth * 0.5) / (tan(verticalFOV * 0.5) * aspect)
        let clampedZoom = min(max(cameraZoom, 0.55), 4.0)
        let standoff = Float((max(fitHeightDistance, fitWidthDistance) * 1.08) / clampedZoom)
        let panUnit = max(planeWidth, planeHeight) * 0.09
        let centerX = Float(cameraOffset.width * panUnit)
        let centerY = Float(cameraOffset.height * panUnit)
        let lookAt = SCNVector3(centerX, centerY, 0)

        cameraNode.position = SCNVector3(centerX, centerY, standoff)
        cameraNode.look(at: lookAt)
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var cameraNode: SCNNode?
        private var cameraOffset: Binding<CGSize>?
        private var cameraZoom: Binding<CGFloat>?
        private var panStartOffset: CGSize = .zero
        private var pinchStartZoom: CGFloat = 1
        private var didInstallGestureRecognizers = false

        func configure(cameraOffset: Binding<CGSize>, cameraZoom: Binding<CGFloat>) {
            self.cameraOffset = cameraOffset
            self.cameraZoom = cameraZoom
        }

        func installGestureRecognizersIfNeeded(on view: SCNView) {
            guard !didInstallGestureRecognizers else { return }
            didInstallGestureRecognizers = true

            let panRecognizer = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            panRecognizer.minimumNumberOfTouches = 1
            panRecognizer.maximumNumberOfTouches = 1
            panRecognizer.delegate = self
            view.addGestureRecognizer(panRecognizer)

            let pinchRecognizer = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            pinchRecognizer.delegate = self
            view.addGestureRecognizer(pinchRecognizer)
        }

        @objc private func handlePan(_ recognizer: UIPanGestureRecognizer) {
            guard let cameraOffset else { return }
            guard let view = recognizer.view else { return }
            switch recognizer.state {
            case .began:
                panStartOffset = cameraOffset.wrappedValue
            case .changed:
                let translation = recognizer.translation(in: view)
                let horizontalUnits = (translation.x / max(view.bounds.width, 1)) * 6
                let verticalUnits = (translation.y / max(view.bounds.height, 1)) * 6
                cameraOffset.wrappedValue = CGSize(
                    width: panStartOffset.width - horizontalUnits,
                    height: panStartOffset.height + verticalUnits
                )
            default:
                break
            }
        }

        @objc private func handlePinch(_ recognizer: UIPinchGestureRecognizer) {
            guard let cameraZoom else { return }
            switch recognizer.state {
            case .began:
                pinchStartZoom = cameraZoom.wrappedValue
            case .changed:
                cameraZoom.wrappedValue = min(max(pinchStartZoom * recognizer.scale, 0.55), 4.0)
            default:
                break
            }
        }

        func gestureRecognizer(
            _ gestureRecognizer: UIGestureRecognizer,
            shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer
        ) -> Bool {
            true
        }
    }
}

/// Pre-save Depth Anything preview — matches Splat ML navigation chrome (nav-bar save, name prompt, discard alert).
private struct DepthAnythingPreviewRoomView: View {
    let destination: USDZViewerDestination

    @Environment(\.dismiss) private var dismiss
    @StateObject private var modelManager = USDZModelManager()
    @State private var shouldResetCamera = false
    @State private var previewCameraOffset: CGSize = .zero
    @State private var previewCameraZoom: CGFloat = 1
    @State private var isSavingRoom = false
    @State private var saveProgress: Double = 0
    @State private var showRoomNameInput = false
    @State private var roomName = ""
    @State private var showSaveAlert = false
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false
    @State private var showDiscardUnsavedAlert = false

    private var depthAnythingRoomDimensions: (width: Float, height: Float, depth: Float)? {
        let width = destination.roomWidthMeters
        let height = destination.roomHeightMeters
        let depth = destination.roomDepthMeters
        guard width.isFinite, height.isFinite, depth.isFinite,
              width > 0.05, height > 0.05, depth > 0.05 else {
            return nil
        }
        return (width, height, depth)
    }

    var body: some View {
        ZStack {
            DepthAnythingPreviewSceneView(
                imageURL: destination.measurementImageURL,
                roomWidthMeters: destination.roomWidthMeters,
                roomHeightMeters: destination.roomHeightMeters,
                cameraOffset: $previewCameraOffset,
                cameraZoom: $previewCameraZoom,
                shouldResetCamera: $shouldResetCamera
            )
            .ignoresSafeArea()

            previewCameraButtonsOverlay

            if let debugLine = destination.measurementDebugLine,
               AppStateManager.shared.qualitySettings.debugMode {
                VStack {
                    Spacer()
                    Text(debugLine)
                        .font(.caption2.monospaced())
                        .foregroundColor(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Color.black.opacity(0.6), in: Capsule())
                        .padding(.bottom, 96)
                }
                .allowsHitTesting(false)
            }

            if isSavingRoom {
                saveRoomProgressOverlay
            }
        }
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)
        .toolbar { previewToolbarContent }
        .alert(L10n.RoomViewer.saveRoom, isPresented: $showRoomNameInput) {
            TextField(L10n.RoomViewer.roomName, text: $roomName)
                .autocorrectionDisabled(true)
            Button(L10n.Common.cancel, role: .cancel) {
                roomName = ""
            }
            Button(L10n.Common.save) {
                startSavingRoom()
            }
            .disabled(roomName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        } message: {
            Text(L10n.RoomViewer.enterName)
        }
        .alert(L10n.RoomViewer.roomSaveTitle, isPresented: $showSaveAlert) {
            Button(L10n.Common.ok, role: .cancel) {
                if saveWasSuccessful {
                    NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil)
                }
            }
        } message: {
            Text(saveAlertMessage)
        }
        .alert(L10n.RoomPreview.unsavedTitle, isPresented: $showDiscardUnsavedAlert) {
            Button(L10n.RoomPreview.stay, role: .cancel) {}
            Button(L10n.RoomPreview.leave, role: .destructive) {
                dismiss()
            }
        } message: {
            Text(L10n.RoomPreview.unsavedMessage)
        }
        .disableBackSwipeIf(true)
        .onAppear {
            if let dimensions = depthAnythingRoomDimensions {
                logDebug(
                    "[DepthAnythingRoom][Preview] nav dims W=\(String(format: "%.3f", dimensions.width)) " +
                    "H=\(String(format: "%.3f", dimensions.height)) " +
                    "D=\(String(format: "%.3f", dimensions.depth))"
                )
            } else {
                logDebug("[DepthAnythingRoom][Preview] nav dims unavailable")
            }
            if destination.photoOrientation == .landscape {
                OrientationLockManager.shared.lockToLandscape()
            } else {
                OrientationLockManager.shared.lockToPortrait()
            }
        }
        .onDisappear {
            OrientationLockManager.shared.unlock()
        }
    }

    @ToolbarContentBuilder
    private var previewToolbarContent: some ToolbarContent {
        ToolbarItem(placement: .navigationBarLeading) {
            Button(action: handleBackTap) {
                Image(systemName: "chevron.left")
            }
            .accessibilityLabel(L10n.Common.back)
        }
        ToolbarItem(placement: .navigationBarTrailing) {
            HStack(spacing: 14) {
                Button {
                    previewCameraOffset = .zero
                    previewCameraZoom = 1
                    shouldResetCamera = true
                } label: {
                    Image(systemName: "viewfinder")
                        .font(.title3)
                }
                .accessibilityLabel(L10n.RoomViewer.recenterView)

                Button {
                    roomName = ""
                    showRoomNameInput = true
                } label: {
                    Image(systemName: "square.and.arrow.down")
                        .font(.title3)
                }
                .disabled(isSavingRoom)
                .accessibilityLabel(L10n.RoomViewer.saveRoom)
            }
        }
    }

    private var previewCameraButtonsOverlay: some View {
        ZStack(alignment: .topLeading) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            VStack(alignment: .leading, spacing: 10) {
                previewCameraDPadCluster
                    .padding(.leading, 12)
                    .padding(.top, 56)
                if destination.photoOrientation == .landscape {
                    Spacer(minLength: 0)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .opacity(isSavingRoom ? 0 : 1)
        .zIndex(102)
    }

    private var previewCameraDPadCluster: some View {
        HStack(spacing: 8) {
            previewCameraMoveButton(systemName: "arrow.left") {
                nudgePreviewCamera(dx: -1, dy: 0)
            }
            VStack(spacing: 8) {
                previewCameraMoveButton(systemName: "arrow.up") {
                    nudgePreviewCamera(dx: 0, dy: 1)
                }
                previewCameraMoveButton(systemName: "arrow.down") {
                    nudgePreviewCamera(dx: 0, dy: -1)
                }
            }
            previewCameraMoveButton(systemName: "arrow.right") {
                nudgePreviewCamera(dx: 1, dy: 0)
            }
        }
    }

    private func previewCameraMoveButton(systemName: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 20, weight: .semibold))
                .foregroundColor(.white)
                .frame(width: 44, height: 44)
                .background(Circle().fill(Color.black.opacity(0.5)))
        }
        .buttonStyle(.plain)
        .disabled(isSavingRoom)
    }

    private func nudgePreviewCamera(dx: CGFloat, dy: CGFloat) {
        previewCameraOffset.width += dx
        previewCameraOffset.height += dy
    }

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

                Text(L10n.RoomViewer.savingRoomEllipsis)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)

                ProgressView(value: saveProgress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .green))
                    .frame(width: 200)

                Text("\(Int(saveProgress * 100))%")
                    .font(.headline)
                    .foregroundColor(.gray)
            }
        }
    }

    private func handleBackTap() {
        if saveWasSuccessful {
            dismiss()
        } else {
            showDiscardUnsavedAlert = true
        }
    }

    private func startSavingRoom() {
        let trimmedRoomName = roomName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedRoomName.isEmpty else { return }
        guard !modelManager.hasSavedRoomNameConflict(trimmedRoomName) else {
            saveAlertMessage = L10n.RoomViewer.duplicateRoomName
            saveWasSuccessful = false
            showSaveAlert = true
            return
        }
        let measurementImageURL = destination.measurementImageURL
        guard FileManager.default.fileExists(atPath: measurementImageURL.path) else {
            saveAlertMessage = "Could not find source image for room measurement."
            saveWasSuccessful = false
            showSaveAlert = true
            return
        }
        let photoOrientationRawValue = destination.photoOrientation.rawValue
        let roomCoordinateFrameRawValue = destination.roomCoordinateFrame.rawValue

        showRoomNameInput = false
        withAnimation(.easeIn(duration: 0.2)) {
            isSavingRoom = true
            saveProgress = 0
        }

        Task {
            await MainActor.run { saveProgress = 0.15 }
            do {
                let savedURL = try await Task.detached(priority: .userInitiated) {
                    let cameraMetadata = CameraExifSidecar.load(roomURL: measurementImageURL)
                    guard let measurementImage = UIImage(contentsOfFile: measurementImageURL.path)?.fixedOrientation() else {
                        throw DepthAnythingPreviewSaveError.sourceImageUnavailable
                    }
                    let reconstructor = try DepthAnythingRoomReconstructor()
                    let measuredResult = try await reconstructor.reconstructWithResult(
                        image: measurementImage,
                        cameraMetadata: cameraMetadata.isEmpty ? nil : cameraMetadata
                    )
                    var measuredCameraMetadata = cameraMetadata
                    for (key, value) in measuredResult.calibrationMetadata {
                        measuredCameraMetadata[key] = value
                    }
                    if !measuredCameraMetadata.isEmpty {
                        CameraExifSidecar.mergeDerivedValues(roomURL: measuredResult.usdzURL, additions: measuredCameraMetadata)
                    }
                    let metadata = depthAnythingSavedRoomMetadata(
                        photoOrientationRawValue: photoOrientationRawValue,
                        roomCoordinateFrameRawValue: roomCoordinateFrameRawValue,
                        displayName: trimmedRoomName,
                        roomWidth: measuredResult.roomWidthMeters,
                        roomHeight: measuredResult.roomHeightMeters,
                        roomDepth: measuredResult.roomDepthMeters
                    )
                    return try copyDepthAnythingRoomToSavedRooms(sourceURL: measuredResult.usdzURL, metadata: metadata)
                }.value
                await MainActor.run {
                    saveProgress = 1.0
                    withAnimation(.easeOut(duration: 0.3)) {
                        isSavingRoom = false
                    }
                    saveAlertMessage = L10n.RoomViewer.saveSuccess(trimmedRoomName)
                    saveWasSuccessful = true
                    showSaveAlert = true
                    roomName = ""
                    logDebug("✅ [DepthAnythingRoom] Saved room to \(savedURL.lastPathComponent)")
                }
            } catch {
                await MainActor.run {
                    isSavingRoom = false
                    saveAlertMessage = error.localizedDescription
                    saveWasSuccessful = false
                    showSaveAlert = true
                    logDebug("❌ [DepthAnythingRoom] Save failed: \(error.localizedDescription)")
                }
            }
        }
    }
}

private enum DepthAnythingPreviewSaveError: LocalizedError {
    case sourceImageUnavailable

    var errorDescription: String? {
        switch self {
        case .sourceImageUnavailable:
            return "Could not load source image for room measurement."
        }
    }
}

private func depthAnythingSavedRoomMetadata(
    photoOrientationRawValue: String,
    roomCoordinateFrameRawValue: String,
    displayName: String,
    roomWidth: Float?,
    roomHeight: Float?,
    roomDepth: Float?
) -> [String: String] {
    var metadata: [String: String] = [
        "photoOrientation": photoOrientationRawValue,
        "roomCoordinateFrame": roomCoordinateFrameRawValue,
        "displayName": displayName
    ]

    if let roomWidth, roomWidth.isFinite, roomWidth > 0 {
        metadata["roomWidth"] = String(format: "%.2f", roomWidth)
    }
    if let roomHeight, roomHeight.isFinite, roomHeight > 0 {
        metadata["roomHeight"] = String(format: "%.2f", roomHeight)
    }
    if let roomDepth, roomDepth.isFinite, roomDepth > 0 {
        metadata["roomDepth"] = String(format: "%.2f", roomDepth)
    }

    return metadata
}

private func copyDepthAnythingRoomToSavedRooms(sourceURL: URL, metadata: [String: String]) throws -> URL {
    let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let savedRoomsURL = documentsURL.appendingPathComponent("SavedRooms", isDirectory: true)
    try FileManager.default.createDirectory(at: savedRoomsURL, withIntermediateDirectories: true)

    let destinationURL = uniqueDepthAnythingSavedRoomURL(in: savedRoomsURL, sourceURL: sourceURL)
    try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
    CameraExifSidecar.copySidecarIfPresent(fromRoomURL: sourceURL, toSavedRoomURL: destinationURL)

    let metadataURL = savedRoomsURL.appendingPathComponent("\(destinationURL.deletingPathExtension().lastPathComponent).usdz.meta")
    let metadataData = try JSONEncoder().encode(metadata)
    try metadataData.write(to: metadataURL, options: [.atomic])
    return destinationURL
}

private func uniqueDepthAnythingSavedRoomURL(in directory: URL, sourceURL: URL) -> URL {
    let baseName = sourceURL.deletingPathExtension().lastPathComponent
    let fileExtension = sourceURL.pathExtension.isEmpty ? "usdz" : sourceURL.pathExtension
    var candidate = directory.appendingPathComponent(baseName).appendingPathExtension(fileExtension)
    var suffix = 1

    while FileManager.default.fileExists(atPath: candidate.path) {
        candidate = directory
            .appendingPathComponent("\(baseName)_\(suffix)")
            .appendingPathExtension(fileExtension)
        suffix += 1
    }

    return candidate
}

struct SinglePhotoRoomView: View {
    @StateObject private var reconstructor = SinglePhotoRoomReconstructor()
    @State private var selectedImage: UIImage?
    @State private var showImagePicker = false
    @State private var showCameraCapture = false  // Show camera capture view
    @State private var captureOrientation: CaptureOrientation = .standard  // Camera mode selection
    @State private var selectedOrientation: PhotoOrientation = .portrait  // User-selected orientation
    @State private var adjustedBoundaries: RoomStructure?
    @State private var navigateToViewer = false
    @State private var fixedImageItem: IdentifiedImage?
    @State private var singlePhotoGenerationStatus: String?
    @State private var generationErrorMessage: String?
    @State private var usdzViewerDestination: USDZViewerDestination?
    @Environment(\.dismiss) private var dismiss
    @AppStorage("singlePhotoRoom.width") private var roomWidth: Double = 4.0
    @AppStorage("singlePhotoRoom.depth") private var roomDepth: Double = 4.5
    @AppStorage("singlePhotoRoom.height") private var roomHeight: Double = 2.8
    @State private var sourceImageURL: URL?
    @State private var captureMediaMetadata: [AnyHashable: Any]?
    @State private var photoLibraryAssetLocalId: String?
    @State private var supplementalCameraDoubles: [String: Double]?
    @State private var hasRequestedDepthAnythingPrewarm = false

    struct IdentifiedImage: Identifiable {
        let id = UUID()
        let image: UIImage
    }

    var body: some View {
        ZStack {
            VStack {
                if let image = selectedImage {
                    // Show image preview and the only supported room generator.
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxHeight: 250)
                        .cornerRadius(12)
                        .padding()
                        .onAppear { logDebug("🖼️ [View] Displaying selected image for Depth Anything generation") }

                    VStack(spacing: 4) {
                        Text(L10n.PhotoRoom.howToCreate)
                            .font(.headline)
                        Text(L10n.PhotoRoom.tapOption)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .padding(.top, 8)
                    .padding(.bottom, 8)

                    Button(action: {
                        logDebug("🤖 [View] Depth Anything method selected")
                        logDebug("📸 User selected pic type: \(selectedOrientation == .portrait ? "Portrait" : "Landscape")")
                        startDepthAnythingGeneration(image: image)
                    }) {
                        HStack(spacing: 16) {
                            Image(systemName: "camera.metering.matrix")
                                .font(.system(size: 30))
                                .foregroundColor(.blue)
                                .frame(width: 50)

                            VStack(alignment: .leading, spacing: 4) {
                                Text(L10n.PhotoRoom.title)
                                    .font(.headline)
                                    .foregroundColor(.primary)
                                Text(L10n.PhotoRoom.aiPowered)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                            Image(systemName: "chevron.right")
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(12)
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.blue, lineWidth: 2)
                        )
                    }
                    .padding(.horizontal)

                    Button(action: {
                        logDebug("🏠 [View] Manual setup selected")
                        logDebug("📸 User selected pic type: \(selectedOrientation == .portrait ? "Portrait" : "Landscape")")
                        fixedImageItem = IdentifiedImage(image: image)
                    }) {
                        HStack(spacing: 16) {
                            Image(systemName: "square.resize")
                                .font(.system(size: 30))
                                .foregroundColor(.orange)
                                .frame(width: 50)

                            VStack(alignment: .leading, spacing: 4) {
                                Text(L10n.PhotoRoom.manualSetup)
                                    .font(.headline)
                                    .foregroundColor(.primary)
                                Text(L10n.PhotoRoom.manualSetupDesc)
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

                    Button("Choose Different Photo") {
                        selectedImage = nil
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
                                    Text(L10n.Camera.takePhoto)
                                        .font(.headline)
                                    Text(L10n.Camera.chooseOrientationShort)
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
                            Text(L10n.Common.or)
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
                                    Text(L10n.PhotoRoom.selectPhoto)
                                        .font(.headline)
                                    Text(L10n.PhotoRoom.fromLibrary)
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
                            Text(L10n.PhotoRoom.screenshotWarning)
                                .font(.subheadline)
                        }
                        .foregroundColor(.red)
                        .padding(.top, 12)

                        Spacer()
                    }
                }
            }

            if let singlePhotoGenerationStatus {
                VStack(spacing: 16) {
                    ProgressView()
                        .scaleEffect(1.35)
                    VStack(spacing: 8) {
                        Text(L10n.GenerationProgress.generating3DModel)
                            .font(.headline)
                            .foregroundColor(.primary)
                        Text(singlePhotoGenerationStatus)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                }
                .padding(28)
                .background(Color(.systemBackground).opacity(0.95))
                .cornerRadius(16)
                .shadow(radius: 10)
                .padding(24)
            }

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

                    Text(L10n.PhotoRoom.buildingRoom)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(32)
                .background(Color(.systemBackground).opacity(0.95))
                .cornerRadius(16)
                .shadow(radius: 10)
            }

        }
        .navigationTitle(L10n.PhotoRoom.title)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button(L10n.Common.back) {
                    handlePhotoRoomBackTap()
                }
            }
        }
        .sheet(isPresented: $showImagePicker) {
            PhotoPickerView(
                selectedImage: $selectedImage,
                sourceImageURL: $sourceImageURL,
                captureMediaMetadata: $captureMediaMetadata,
                photoLibraryAssetLocalId: $photoLibraryAssetLocalId,
                supplementalCameraDoubles: $supplementalCameraDoubles,
            )
                .onDisappear {
                    logDebug("📱 [View] Image picker dismissed")
                    if selectedImage != nil {
                        logDebug("✅ [View] Image selected, ready for Depth Anything")
                    } else {
                        logDebug("⚠️ [View] No image selected")
                    }
                }
        }
        .sheet(isPresented: $showCameraCapture) {
            CameraCaptureView(
                selectedImage: $selectedImage,
                selectedOrientation: $captureOrientation,
                sourceImageURL: $sourceImageURL,
                captureMediaMetadata: $captureMediaMetadata,
                photoLibraryAssetLocalId: $photoLibraryAssetLocalId,
                supplementalCameraDoubles: $supplementalCameraDoubles,
            )
                .onDisappear {
                    logDebug("📷 [View] Camera capture dismissed")
                    if selectedImage != nil {
                        logDebug("✅ [View] Photo captured with orientation: \(captureOrientation.rawValue), ready for Depth Anything")
                    } else {
                        logDebug("⚠️ [View] No photo captured")
                    }
                }
        }
        .onChange(of: selectedImage) { _, newValue in
            if newValue == nil {
                sourceImageURL = nil
                captureMediaMetadata = nil
                photoLibraryAssetLocalId = nil
                supplementalCameraDoubles = nil
            }
            guard let image = newValue else { return }
            logDebug("✅ [View] Image selected")
            prewarmDepthAnythingModelIfNeeded()
            logDebug("🤖 [View] Depth Anything generation ready")
            // Auto-detect orientation and pre-select it (user can override)
            let detectedOrientation = PhotoOrientation.detect(from: image)
            selectedOrientation = detectedOrientation
            logDebug("📐 [View] Auto-detected orientation: \(detectedOrientation.rawValue)")
        }
        .fullScreenCover(item: $fixedImageItem) { item in
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
                    logDebug("✅ [onProcessingComplete] Processing complete, navigating to viewer")
                    navigateToViewer = true
                },
                photoOrientation: selectedOrientation
            )
            .onAppear {
                logDebug("✅ [Sheet] Opening RoomBoundaryDetectionView with image: \(item.image.size)")
                if selectedOrientation == .landscape {
                    OrientationLockManager.shared.lockToLandscape()
                } else {
                    OrientationLockManager.shared.lockToPortrait()
                }
            }
            .onDisappear {
                OrientationLockManager.shared.unlock()
            }
        }
        .onAppear {
            logDebug("👁️ [View] SinglePhotoRoomView appeared")
        }
        .onChange(of: adjustedBoundaries) { _, newValue in
            guard let bounds = newValue else { return }
            logDebug("📋 [View] adjustedBoundaries updated")
            logDebug("   Boundaries: L=\(bounds.leftX), R=\(bounds.rightX), T=\(bounds.ceilingY), B=\(bounds.floorY)")
        }
        .navigationDestination(isPresented: $navigateToViewer) {
            if let image = selectedImage, let boundaries = adjustedBoundaries {
                MeshRoomView(
                    roomWidth: Float(roomWidth),
                    roomHeight: Float(roomHeight),
                    roomDepth: Float(roomDepth),
                    frontWallImage: image,
                    photoOrientation: selectedOrientation,
                    leftX: boundaries.leftX,
                    rightX: boundaries.rightX,
                    ceilingY: boundaries.ceilingY,
                    floorY: boundaries.floorY
                )
            } else if let image = selectedImage {
                MeshRoomView(
                    roomWidth: Float(roomWidth),
                    roomHeight: Float(roomHeight),
                    roomDepth: Float(roomDepth),
                    frontWallImage: image,
                    photoOrientation: selectedOrientation
                )
            }
        }
        .navigationDestination(item: $usdzViewerDestination) { destination in
            DepthAnythingPreviewRoomView(destination: destination)
                .onAppear {
                    logDebug("🚀 [Navigation] DepthAnythingPreviewRoomView (fast image-plane preview)")
                    logDebug("   \(destination.summary)")
                }
                .onDisappear {
                    usdzViewerDestination = nil
                }
        }
        .alert(L10n.PhotoRoom.generationFailedTitle, isPresented: Binding(
            get: { generationErrorMessage != nil },
            set: { if !$0 { generationErrorMessage = nil } }
        )) {
            Button(L10n.Common.ok, role: .cancel) {
                selectedImage = nil
            }
        } message: {
            Text(generationErrorMessage ?? L10n.PhotoRoom.errorMessage)
        }
    }

    private func handlePhotoRoomBackTap() {
        dismiss()
    }

    private func prewarmDepthAnythingModelIfNeeded() {
        guard !hasRequestedDepthAnythingPrewarm else { return }
        hasRequestedDepthAnythingPrewarm = true
        Task.detached(priority: .utility) {
            do {
                _ = try DepthAnythingRoomReconstructor()
                logDebug("[DepthAnythingRoom][Prewarm] shared model ready")
            } catch {
                logDebug("[DepthAnythingRoom][Prewarm] skipped: \(error.localizedDescription)")
            }
        }
    }

    private func startDepthAnythingGeneration(image: UIImage) {
        let generationImage = image.fixedOrientation()
        let orientation = selectedOrientation
        let generationSourceImageURL = sourceImageURL
        let generationMediaMetadata = captureMediaMetadata
        let generationPhotoLibraryAssetLocalId = photoLibraryAssetLocalId
        let generationSupplementalCameraDoubles = supplementalCameraDoubles
        selectedImage = nil
        fixedImageItem = nil
        usdzViewerDestination = nil
        generationErrorMessage = nil
        singlePhotoGenerationStatus = L10n.RoomGeneration.creatingRoom

        Task {
            do {
                let cameraMetadata = await CameraExifSidecar.collectMerged(
                    imageURL: generationSourceImageURL,
                    mediaMetadata: generationMediaMetadata,
                    photoLibraryAssetLocalId: generationPhotoLibraryAssetLocalId,
                    supplementalDoubles: generationSupplementalCameraDoubles
                )
                if cameraMetadata.isEmpty {
                    logDebug("[DepthAnythingRoom][CameraMetadata] unavailable; focal will use EXIF on UIImage or fallback")
                } else {
                    logDebug("[DepthAnythingRoom][CameraMetadata] keys=\(cameraMetadata.keys.sorted())")
                }
                let preview = try await Task.detached(priority: .userInitiated) {
                    try makeDepthAnythingPreviewDestination(
                        image: generationImage,
                        cameraMetadata: cameraMetadata,
                        photoOrientation: orientation
                    )
                }.value
                logDebug("✅ [DepthAnythingRoom] Fast preview ready: \(preview.summary)")
                logDebug(
                    "[DepthAnythingRoom][PreviewFast][UIResult] " +
                    "image=\(preview.imageWidth)x\(preview.imageHeight) " +
                    "W=\(String(format: "%.4f", preview.roomWidthMeters)) " +
                    "H=\(String(format: "%.4f", preview.roomHeightMeters)) " +
                    "D=\(String(format: "%.4f", preview.roomDepthMeters)) " +
                    "source=\(preview.measurementImageURL.lastPathComponent)"
                )
                await MainActor.run {
                    singlePhotoGenerationStatus = nil
                    usdzViewerDestination = USDZViewerDestination(
                        measurementImageURL: preview.measurementImageURL,
                        photoOrientation: orientation,
                        roomCoordinateFrame: .depthAnythingImageDepthMeters,
                        summary: preview.summary,
                        roomWidthMeters: preview.roomWidthMeters,
                        roomHeightMeters: preview.roomHeightMeters,
                        roomDepthMeters: preview.roomDepthMeters,
                        measurementDebugLine: nil
                    )
                }
            } catch {
                logDebug("❌ [DepthAnythingRoom] generation failed: \(error.localizedDescription)")
                await MainActor.run {
                    singlePhotoGenerationStatus = nil
                    generationErrorMessage = error.localizedDescription
                }
            }
        }
    }

    /// Crop image to the selected front wall boundaries
    private func cropImageToFrontWall(image: UIImage, leftX: CGFloat, rightX: CGFloat, ceilingY: CGFloat, floorY: CGFloat) -> UIImage {
        logDebug("🔲 [cropImageToFrontWall] Starting crop with boundaries: L=\(leftX), R=\(rightX), T=\(ceilingY), B=\(floorY)")
        logDebug("   Input image: \(image.size), orientation: \(image.imageOrientation.rawValue)")

        // First, normalize orientation so CGImage matches what user saw
        let normalizedImage = image.imageOrientation == .up ? image : image.fixedOrientation()

        guard let cgImage = normalizedImage.cgImage else {
            logDebug("⚠️ [cropImageToFrontWall] Failed to get CGImage, returning original")
            return image
        }

        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)
        logDebug("   CGImage size: \(Int(imageWidth))x\(Int(imageHeight))")

        // Convert normalized coordinates (0-1) to pixel coordinates
        let cropX = leftX * imageWidth
        let cropY = ceilingY * imageHeight
        let cropWidth = (rightX - leftX) * imageWidth
        let cropHeight = (floorY - ceilingY) * imageHeight

        // Ensure valid crop rect
        let cropRect = CGRect(
            x: max(0, cropX),
            y: max(0, cropY),
            width: min(cropWidth, imageWidth - cropX),
            height: min(cropHeight, imageHeight - cropY)
        )

        logDebug("   Crop rect: x=\(Int(cropRect.minX)), y=\(Int(cropRect.minY)), w=\(Int(cropRect.width)), h=\(Int(cropRect.height))")

        // Perform the crop
        guard cropRect.width > 0, cropRect.height > 0,
              let croppedCGImage = cgImage.cropping(to: cropRect) else {
            logDebug("⚠️ [cropImageToFrontWall] Invalid crop rect, returning original image")
            return image
        }

        // Return cropped image with .up orientation (already normalized)
        let croppedImage = UIImage(cgImage: croppedCGImage, scale: normalizedImage.scale, orientation: .up)
        logDebug("✅ [cropImageToFrontWall] Cropped image from \(Int(imageWidth))x\(Int(imageHeight)) to \(Int(cropRect.width))x\(Int(cropRect.height))")

        return croppedImage
    }

}

// MARK: - Photo Picker View
struct PhotoPickerView: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Binding var sourceImageURL: URL?
    @Binding var captureMediaMetadata: [AnyHashable: Any]?
    @Binding var photoLibraryAssetLocalId: String?
    @Binding var supplementalCameraDoubles: [String: Double]?
    @Environment(\.dismiss) var dismiss

    func makeUIViewController(context: Context) -> PHPickerViewController {
        logDebug("📱 [PhotoPicker] Creating PHPickerViewController")
        var config = PHPickerConfiguration(photoLibrary: .shared())
        config.filter = .images
        config.selectionLimit = 1
        config.preferredAssetRepresentationMode = .current
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}
    
    func makeCoordinator() -> Coordinator { Coordinator(self) }
    
    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: PhotoPickerView
        
        init(_ parent: PhotoPickerView) {
            self.parent = parent
            logDebug("📱 [PhotoPicker] Coordinator initialized")
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            logDebug("📱 [PhotoPicker] PHPicker finished results=\(results.count)")
            parent.captureMediaMetadata = nil
            parent.supplementalCameraDoubles = nil
            guard let result = results.first else {
                logDebug("❌ [PhotoPicker] No result selected")
                parent.dismiss()
                return
            }

            parent.photoLibraryAssetLocalId = result.assetIdentifier
            logDebug("📱 [PhotoPicker] assetIdentifier=\(result.assetIdentifier ?? "nil")")

            copyOriginalImageFile(from: result.itemProvider)

            guard result.itemProvider.canLoadObject(ofClass: UIImage.self) else {
                logDebug("❌ [PhotoPicker] Item provider cannot load UIImage")
                parent.dismiss()
                return
            }

            result.itemProvider.loadObject(ofClass: UIImage.self) { object, error in
                DispatchQueue.main.async {
                    if let error {
                        logDebug("❌ [PhotoPicker] UIImage load failed: \(error.localizedDescription)")
                    }
                    if let image = object as? UIImage {
                        logDebug("✅ [PhotoPicker] Got UIImage: \(image.size), orientation: \(image.imageOrientation.rawValue)")
                        self.parent.selectedImage = image
                    } else {
                        logDebug("❌ [PhotoPicker] Failed to get UIImage")
                    }
                    self.parent.dismiss()
                }
            }
        }

        private func copyOriginalImageFile(from provider: NSItemProvider) {
            provider.loadFileRepresentation(forTypeIdentifier: UTType.image.identifier) { url, error in
                if let url {
                    self.copyPickedFile(at: url, source: "file_representation")
                    return
                }
                if let error {
                    logDebug("❌ [PhotoPicker] fileRepresentation failed: \(error.localizedDescription)")
                }
                provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, error in
                    if let error {
                        logDebug("❌ [PhotoPicker] dataRepresentation failed: \(error.localizedDescription)")
                    }
                    guard let data else { return }
                    let ext = provider.suggestedName?.split(separator: ".").last.map(String.init) ?? "img"
                    let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("picker_original_\(UUID().uuidString).\(ext)")
                    do {
                        try data.write(to: tempURL, options: [.atomic])
                        DispatchQueue.main.async {
                            self.parent.sourceImageURL = tempURL
                            logDebug("📱 [PhotoPicker] Copied original data to: \(tempURL.lastPathComponent) bytes=\(data.count)")
                        }
                    } catch {
                        logDebug("❌ [PhotoPicker] temp write failed: \(error.localizedDescription)")
                    }
                }
            }
        }

        private func copyPickedFile(at url: URL, source: String) {
            let ext = url.pathExtension.isEmpty ? "img" : url.pathExtension
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("picker_original_\(UUID().uuidString).\(ext)")
            do {
                if FileManager.default.fileExists(atPath: tempURL.path) {
                    try FileManager.default.removeItem(at: tempURL)
                }
                try FileManager.default.copyItem(at: url, to: tempURL)
                DispatchQueue.main.async {
                    self.parent.sourceImageURL = tempURL
                    logDebug("📱 [PhotoPicker] Copied original \(source) to: \(tempURL.lastPathComponent)")
                }
            } catch {
                logDebug("❌ [PhotoPicker] copy \(source) failed: \(error.localizedDescription)")
            }
        }
    }
}

// MARK: - Camera Mode Enum
enum CaptureOrientation: String, CaseIterable {
    case standard = "Standard"
    case wideAngle = "Wide Angle"

    var icon: String {
        switch self {
        case .standard: return "camera"
        case .wideAngle: return "camera.filters"
        }
    }

    var description: String {
        switch self {
        case .standard: return NSLocalizedString("camera.standard.desc", comment: "Standard 1x camera")
        case .wideAngle: return NSLocalizedString("camera.wideAngle.desc", comment: "Ultra-wide 0.5x lens")
        }
    }

    var localizationKey: String {
        switch self {
        case .standard: return "camera.standard"
        case .wideAngle: return "camera.wideAngle"
        }
    }
}

// MARK: - Camera Capture View with Orientation Selection
struct CameraCaptureView: View {
    @Binding var selectedImage: UIImage?
    @Binding var selectedOrientation: CaptureOrientation
    @Binding var sourceImageURL: URL?
    @Binding var captureMediaMetadata: [AnyHashable: Any]?
    @Binding var photoLibraryAssetLocalId: String?
    @Binding var supplementalCameraDoubles: [String: Double]?
    @Environment(\.dismiss) var dismiss

    @State private var showCamera = false
    @State private var capturedImage: UIImage?
    @State private var showWideAngleGuide = false
    @State private var showPhotoPicker = false
    @State private var showWideAngleCamera = false

    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // Camera Mode Selection Header
                VStack(spacing: 8) {
                    Image(systemName: "camera.viewfinder")
                        .font(.system(size: 50))
                        .foregroundColor(.blue)

                    Text(NSLocalizedString("camera.chooseMode", comment: "Choose Camera Mode"))
                        .font(.title2)
                        .fontWeight(.bold)

                    Text(NSLocalizedString("camera.chooseModeHint", comment: "Select camera lens for your room"))
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
                    // Standard camera button - works in any orientation
                    Button(action: {
                        logDebug("📷 [Camera] Opening standard camera")
                        showCamera = true
                    }) {
                        HStack(spacing: 12) {
                            Image(systemName: "camera.fill")
                                .font(.title2)
                            Text(L10n.Camera.takePhoto)
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
                Group {
                    if ARRoomPhotoCapturePolicy.useARKitForStandardRoomPhoto {
                        ARRoomPhotoCaptureRepresentable(
                            capturedImage: $capturedImage,
                            sourceImageURL: $sourceImageURL,
                            captureMediaMetadata: $captureMediaMetadata,
                            supplementalCameraDoubles: $supplementalCameraDoubles,
                        )
                    } else {
                        CameraViewRepresentable(
                            capturedImage: $capturedImage,
                            sourceImageURL: $sourceImageURL,
                            captureMediaMetadata: $captureMediaMetadata,
                            photoLibraryAssetLocalId: $photoLibraryAssetLocalId,
                            supplementalCameraDoubles: $supplementalCameraDoubles,
                            orientation: selectedOrientation,
                        )
                    }
                }
                .ignoresSafeArea()
            }
            .fullScreenCover(isPresented: $showWideAngleCamera) {
                WideAngleCameraView(
                    capturedImage: $capturedImage,
                    photoLibraryAssetLocalId: $photoLibraryAssetLocalId,
                    supplementalCameraDoubles: $supplementalCameraDoubles,
                )
                .ignoresSafeArea()
            }
            .sheet(isPresented: $showPhotoPicker) {
                PhotoLibraryPicker(
                    selectedImage: $capturedImage,
                    sourceImageURL: $sourceImageURL,
                    captureMediaMetadata: $captureMediaMetadata,
                    photoLibraryAssetLocalId: $photoLibraryAssetLocalId,
                    supplementalCameraDoubles: $supplementalCameraDoubles,
                )
            }
            .onChange(of: capturedImage) { _, newImage in
                if let image = newImage {
                    logDebug("📷 [Camera] Photo captured: \(image.size)")
                    selectedImage = image
                    dismiss()
                }
            }
            .onChange(of: showCamera) { _, isShowing in
                if isShowing {
                    supplementalCameraDoubles = nil
                }
            }
        }
    }
}

// MARK: - Photo Library Picker
struct PhotoLibraryPicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Binding var sourceImageURL: URL?
    @Binding var captureMediaMetadata: [AnyHashable: Any]?
    @Binding var photoLibraryAssetLocalId: String?
    @Binding var supplementalCameraDoubles: [String: Double]?
    @Environment(\.dismiss) var dismiss

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration(photoLibrary: .shared())
        config.filter = .images
        config.selectionLimit = 1
        config.preferredAssetRepresentationMode = .current
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: PhotoLibraryPicker

        init(_ parent: PhotoLibraryPicker) {
            self.parent = parent
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            parent.captureMediaMetadata = nil
            parent.supplementalCameraDoubles = nil
            guard let result = results.first else {
                parent.dismiss()
                return
            }

            parent.photoLibraryAssetLocalId = result.assetIdentifier
            logDebug("📷 [PhotoPicker] PHPicker assetIdentifier=\(result.assetIdentifier ?? "nil")")
            copyOriginalImageFile(from: result.itemProvider)

            guard result.itemProvider.canLoadObject(ofClass: UIImage.self) else {
                parent.dismiss()
                return
            }

            result.itemProvider.loadObject(ofClass: UIImage.self) { object, error in
                DispatchQueue.main.async {
                    if let error {
                        logDebug("❌ [PhotoPicker] UIImage load failed: \(error.localizedDescription)")
                    }
                    if let image = object as? UIImage {
                        logDebug("📷 [PhotoPicker] Selected image: \(image.size)")
                        self.parent.selectedImage = image
                    }
                    self.parent.dismiss()
                }
            }
        }

        private func copyOriginalImageFile(from provider: NSItemProvider) {
            provider.loadFileRepresentation(forTypeIdentifier: UTType.image.identifier) { url, error in
                if let url {
                    self.copyPickedFile(at: url, source: "file_representation")
                    return
                }
                if let error {
                    logDebug("❌ [PhotoPicker] fileRepresentation failed: \(error.localizedDescription)")
                }
                provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, error in
                    if let error {
                        logDebug("❌ [PhotoPicker] dataRepresentation failed: \(error.localizedDescription)")
                    }
                    guard let data else { return }
                    let ext = provider.suggestedName?.split(separator: ".").last.map(String.init) ?? "img"
                    let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("picker_original_\(UUID().uuidString).\(ext)")
                    do {
                        try data.write(to: tempURL, options: [.atomic])
                        DispatchQueue.main.async {
                            self.parent.sourceImageURL = tempURL
                            logDebug("📷 [PhotoPicker] Copied original data to: \(tempURL.lastPathComponent) bytes=\(data.count)")
                        }
                    } catch {
                        logDebug("❌ [PhotoPicker] temp write failed: \(error.localizedDescription)")
                    }
                }
            }
        }

        private func copyPickedFile(at url: URL, source: String) {
            let ext = url.pathExtension.isEmpty ? "img" : url.pathExtension
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("picker_original_\(UUID().uuidString).\(ext)")
            do {
                if FileManager.default.fileExists(atPath: tempURL.path) {
                    try FileManager.default.removeItem(at: tempURL)
                }
                try FileManager.default.copyItem(at: url, to: tempURL)
                DispatchQueue.main.async {
                    self.parent.sourceImageURL = tempURL
                    logDebug("📷 [PhotoPicker] Copied original \(source) to: \(tempURL.lastPathComponent)")
                }
            } catch {
                logDebug("❌ [PhotoPicker] copy \(source) failed: \(error.localizedDescription)")
            }
        }
    }
}

// MARK: - Wide Angle Camera View (AVFoundation-based with Ultra-Wide Lens)
import AVFoundation

struct WideAngleCameraView: UIViewControllerRepresentable {
    @Binding var capturedImage: UIImage?
    @Binding var photoLibraryAssetLocalId: String?
    @Binding var supplementalCameraDoubles: [String: Double]?
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
            CameraOwnershipDiagnostics.log(owner: "WideAngleCameraView", event: "capturedImage")
            parent.supplementalCameraDoubles = nil
            parent.photoLibraryAssetLocalId = nil
            parent.capturedImage = image.fixedOrientation()
            parent.dismiss()
        }

        func wideAngleCameraDidCancel() {
            logDebug("📷 [WideAngle] User cancelled")
            CameraOwnershipDiagnostics.log(owner: "WideAngleCameraView", event: "cancel")
            parent.supplementalCameraDoubles = nil
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
    private var captureSessionObserverTokens: [NSObjectProtocol] = []
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
        if let captureSession {
            captureSessionObserverTokens = CameraOwnershipDiagnostics.makeCaptureSessionObservers(
                session: captureSession,
                owner: "WideAngleCameraViewController.AVCapture"
            )
        }

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
                CameraOwnershipDiagnostics.log(owner: "WideAngleCameraViewController.AVCapture", event: "capture_startRequested")
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
        CameraOwnershipDiagnostics.log(owner: "WideAngleCameraViewController.AVCapture", event: "capture_stopRequested", details: "reason=cancel")
        captureSession?.stopRunning()
        delegate?.wideAngleCameraDidCancel()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        CameraOwnershipDiagnostics.log(owner: "WideAngleCameraViewController.AVCapture", event: "capture_stopRequested", details: "reason=viewWillDisappear")
        captureSession?.stopRunning()
    }

    deinit {
        CameraOwnershipDiagnostics.removeObservers(captureSessionObserverTokens)
        CameraOwnershipDiagnostics.log(owner: "WideAngleCameraViewController", event: "deinit")
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
        CameraOwnershipDiagnostics.log(owner: "WideAngleCameraViewController", event: "photoOutput_didFinishProcessing")
        CameraOwnershipDiagnostics.log(owner: "WideAngleCameraViewController.AVCapture", event: "capture_stopRequested", details: "reason=photoCaptured")
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

// MARK: - Standard Camera View (UIImagePickerController - works in any orientation)
struct CameraViewRepresentable: UIViewControllerRepresentable {
    @Binding var capturedImage: UIImage?
    @Binding var sourceImageURL: URL?
    @Binding var captureMediaMetadata: [AnyHashable: Any]?
    @Binding var photoLibraryAssetLocalId: String?
    @Binding var supplementalCameraDoubles: [String: Double]?
    let orientation: CaptureOrientation
    @Environment(\.dismiss) var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        logDebug("📷 [Camera] Opening standard camera")
        CameraOwnershipDiagnostics.log(owner: "CameraViewRepresentable.UIImagePickerController", event: "present")
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        picker.allowsEditing = false
        picker.cameraCaptureMode = .photo
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraViewRepresentable

        init(_ parent: CameraViewRepresentable) {
            self.parent = parent
            logDebug("📷 [Camera] Coordinator initialized")
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            logDebug("📷 [Camera] Photo captured")
            CameraOwnershipDiagnostics.log(owner: "CameraViewRepresentable.UIImagePickerController", event: "didFinishPicking")
            parent.supplementalCameraDoubles = nil
            parent.sourceImageURL = nil
            parent.photoLibraryAssetLocalId = nil
            if let md = info[.mediaMetadata] {
                parent.captureMediaMetadata = md as? [AnyHashable: Any]
            } else {
                parent.captureMediaMetadata = nil
            }
            if let image = info[.originalImage] as? UIImage {
                logDebug("✅ [Camera] Got UIImage: \(image.size)")
                parent.capturedImage = image.fixedOrientation()
            } else {
                logDebug("❌ [Camera] Failed to get UIImage")
            }
            CameraOwnershipDiagnostics.log(owner: "CameraViewRepresentable.UIImagePickerController", event: "dismiss_requested", details: "reason=didFinishPicking")
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            logDebug("📷 [Camera] User cancelled")
            CameraOwnershipDiagnostics.log(owner: "CameraViewRepresentable.UIImagePickerController", event: "dismiss_requested", details: "reason=cancel")
            parent.supplementalCameraDoubles = nil
            parent.dismiss()
        }
    }
}

// MARK: - SceneKit Viewer
struct SceneKitViewer: View {
    let scene: SCNScene
    let photoOrientation: PhotoOrientation
    let roomWidth: Float
    let roomHeight: Float
    var allowSave: Bool = true
    @Environment(\.dismiss) private var dismiss
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

    var body: some View {
        ZStack {
            SceneView(
                scene: scene,
                pointOfView: cameraNode,
                options: [.autoenablesDefaultLighting]  // Removed .allowsCameraControl - SceneKitGestureOverlay handles gestures
            )
            .allowsHitTesting(false)  // Let TouchDragOverlay receive all touches
            .onAppear {
                logDebug("🎬 [Viewer] SceneKit viewer appeared")
                logDebug("   - Scene nodes: \(scene.rootNode.childNodes.count)")
                logDebug("🎬 [SceneKitViewer] orientation=\(photoOrientation.rawValue) allowSave=\(allowSave) room=\(roomWidth)x\(roomHeight)")
                Task {
                    setupCamera()
                    // Small delay to ensure camera is ready
                    try? await Task.sleep(nanoseconds: 300_000_000)
                    await MainActor.run {
                        isSettingUp = false
                    }
                }
            }
            .onDisappear {
                // Camera cleanup handled by SceneKitGestureOverlay
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

                        Text(L10n.PhotoRoom.loading3DRoom)
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

            // ✅ UNIFIED GESTURE HANDLER - same gestures as RealityKit rooms
            SceneKitGestureOverlay(cameraNode: cameraNode)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(true)
                .zIndex(99996)
                .onAppear {
                    logDebug("🪟 [SceneKitViewer] Gesture overlay appeared")
                }

            // Custom back button (top-left) - matches SplatRoomView style
            VStack {
                HStack {
                    Button(action: {
                        dismiss()
                    }) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(Color.black.opacity(0.6))
                            .clipShape(Circle())
                    }
                    .padding(.leading, 16)
                    .padding(.top, 8)
                    Spacer()
                }
                Spacer()
            }
            .zIndex(99999)

            // Orientation label overlay
            VStack {
                Spacer()
                    .allowsHitTesting(false)

                HStack(spacing: 6) {
                    Image(systemName: photoOrientation == .landscape ? "iphone.landscape" : "iphone")
                        .font(.caption)
                    Text(photoOrientation == .landscape
                         ? NSLocalizedString("orientation.heldHorizontally", comment: "")
                         : NSLocalizedString("orientation.heldVertically", comment: ""))
                        .font(.caption2)
                    Text("-")
                        .font(.caption2)
                    Text(photoOrientation == .landscape
                         ? NSLocalizedString("orientation.landscape", comment: "")
                         : NSLocalizedString("orientation.portrait", comment: ""))
                        .font(.caption2)
                        .fontWeight(.medium)
                }
                .foregroundColor(.white.opacity(0.9))
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.5))
                .cornerRadius(8)
                .padding(.bottom, 30)
            }
            .zIndex(99995)
        }
        .navigationTitle(L10n.RoomViewer.approximateRoomHeight(roomHeight))
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if allowSave {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        if roomName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            roomName = RoomDisplayName.myRoomWithTimestamp()
                        }
                        showRoomNameInput = true
                    } label: {
                        Image(systemName: "square.and.arrow.down")
                    }
                }
            }
        }
        // Room name input alert
        .alert(L10n.RoomViewer.saveRoom, isPresented: $showRoomNameInput) {
            TextField(L10n.RoomViewer.roomName, text: $roomName)
                .autocorrectionDisabled(true)
            Button(L10n.Common.cancel, role: .cancel) {
                roomName = ""
            }
            Button(L10n.Common.save) {
                startSavingRoom()
            }
            .disabled(roomName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
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
        .onAppear {
            // Lock orientation based on photo orientation
            if photoOrientation == .landscape {
                OrientationLockManager.shared.lockToLandscape()
            } else {
                OrientationLockManager.shared.lockToPortrait()
            }
            logDebug("📐 [SceneKitViewer] Locking to \(photoOrientation == .landscape ? "landscape" : "portrait")")
        }
        .onDisappear {
            OrientationLockManager.shared.unlock()
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
        let trimmedRoomName = roomName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedRoomName.isEmpty else {
            return
        }

        // Lazy-initialize modelManager only when saving
        if modelManager == nil {
            modelManager = USDZModelManager()
        }
        if modelManager?.hasSavedRoomNameConflict(trimmedRoomName) == true {
            saveAlertMessage = L10n.RoomViewer.duplicateRoomName
            saveWasSuccessful = false
            showSaveAlert = true
            return
        }

        let savedName = trimmedRoomName  // ✅ Capture the name BEFORE clearing
        roomName = trimmedRoomName
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
        logDebug("   🎛️ Camera euler after lookAt: \(camNode.eulerAngles)")

        scene.rootNode.addChildNode(camNode)
        cameraNode = camNode

        // Camera control handled by SceneKitGestureOverlay (unified gesture handler)
        logDebug("   ✅ Camera setup complete - gestures handled by SceneKitGestureOverlay")
    }
}
