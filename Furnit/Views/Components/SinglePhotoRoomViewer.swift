import SwiftUI
import SceneKit
import UIKit
import Accelerate
import CoreML
import Photos
import PhotosUI
import UniformTypeIdentifiers
import simd

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
        PaafektBuildingRoomOverlay(
            progress: Double(reconstructor.progress),
            statusMessage: reconstructor.statusMessage
        )
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

private enum DepthAnythingPreviewPlacementIntelligenceRoomStub {
    static func axisAlignedBoxMeters(width: Float, height: Float, depth: Float) -> RoomModel {
        let w = max(width, 0.2)
        let h = max(height, 0.2)
        let d = max(depth, 0.2)
        let wHalf = w * 0.5
        let dHalf = d * 0.5
        let aabb = AABB3(
            min: SIMD3<Float>(-wHalf, 0, -dHalf),
            max: SIMD3<Float>(wHalf, h, dHalf)
        )
        let floor = DetectedPlane(type: .floor, normal: SIMD3<Float>(0, 1, 0), pointOnPlane: .zero)
        let ceiling = DetectedPlane(type: .ceiling, normal: SIMD3<Float>(0, -1, 0), pointOnPlane: SIMD3<Float>(0, h, 0))
        let walls: [DetectedPlane] = [
            DetectedPlane(type: .wall, normal: SIMD3<Float>(1, 0, 0), pointOnPlane: SIMD3<Float>(-wHalf, 0, 0)),
            DetectedPlane(type: .wall, normal: SIMD3<Float>(-1, 0, 0), pointOnPlane: SIMD3<Float>(wHalf, 0, 0)),
            DetectedPlane(type: .wall, normal: SIMD3<Float>(0, 0, 1), pointOnPlane: SIMD3<Float>(0, 0, -dHalf)),
            DetectedPlane(type: .wall, normal: SIMD3<Float>(0, 0, -1), pointOnPlane: SIMD3<Float>(0, 0, dHalf))
        ]
        let uvMin = SIMD2<Float>(-wHalf, -dHalf)
        let uvMax = SIMD2<Float>(wHalf, dHalf)
        let freeFloor = FreeFloorRegion(
            polygon: [
                uvMin,
                SIMD2<Float>(wHalf, -dHalf),
                uvMax,
                SIMD2<Float>(-wHalf, dHalf)
            ],
            areaSqM: w * d,
            uvBounds: FloorUVBounds(min: uvMin, max: uvMax)
        )
        return RoomModel(
            aabb: aabb,
            floor: floor,
            ceiling: ceiling,
            walls: walls,
            corners: [],
            freeFloorRegions: [freeFloor],
            surfacePalette: .empty,
            cameraInfo: nil,
            sceneToMeters: 1.0
        )
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
            return L10n.RoomPreview.prepareSourceImageFailed
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

private func downsampleUIImageForDisplay(_ image: UIImage, maxDimension: CGFloat) -> UIImage {
    let maxSide = max(image.size.width * image.scale, image.size.height * image.scale)
    guard maxSide > maxDimension, maxSide > 0 else { return image }
    let scale = maxDimension / maxSide
    let targetSize = CGSize(
        width: max(1, floor(image.size.width * scale)),
        height: max(1, floor(image.size.height * scale))
    )
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
    return renderer.image { _ in
        image.draw(in: CGRect(origin: .zero, size: targetSize))
    }
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
    let photoOrientation: PhotoOrientation
    @Binding var cameraOffset: CGSize
    @Binding var cameraZoom: CGFloat
    @Binding var shouldResetCamera: Bool
    var allowsSceneInteraction: Bool = true

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
        context.coordinator.setSceneInteractionEnabled(allowsSceneInteraction, on: view)
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
        let pose = DepthAnythingFlatPhotoCameraFraming.sceneKitPreviewCameraPose(
            planeWidthMeters: max(roomWidthMeters, 0.05),
            planeHeightMeters: max(roomHeightMeters, 0.05),
            photoOrientation: photoOrientation,
            viewportSize: viewportSize,
            cameraOffset: cameraOffset,
            cameraZoom: cameraZoom
        )
        cameraNode.camera?.fieldOfView = pose.verticalFieldOfViewDegrees
        cameraNode.position = pose.position
        cameraNode.look(at: pose.lookAt)
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var cameraNode: SCNNode?
        private var cameraOffset: Binding<CGSize>?
        private var cameraZoom: Binding<CGFloat>?
        private var panStartOffset: CGSize = .zero
        private var pinchStartZoom: CGFloat = 1
        private var panRecognizer: UIPanGestureRecognizer?
        private var pinchRecognizer: UIPinchGestureRecognizer?
        private var didInstallGestureRecognizers = false

        func configure(cameraOffset: Binding<CGSize>, cameraZoom: Binding<CGFloat>) {
            self.cameraOffset = cameraOffset
            self.cameraZoom = cameraZoom
        }

        func setSceneInteractionEnabled(_ enabled: Bool, on view: SCNView) {
            view.isUserInteractionEnabled = enabled
            panRecognizer?.isEnabled = enabled
            pinchRecognizer?.isEnabled = enabled
        }

        func installGestureRecognizersIfNeeded(on view: SCNView) {
            guard !didInstallGestureRecognizers else { return }
            didInstallGestureRecognizers = true

            let panRecognizer = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            panRecognizer.minimumNumberOfTouches = 1
            panRecognizer.maximumNumberOfTouches = 1
            panRecognizer.delegate = self
            view.addGestureRecognizer(panRecognizer)
            self.panRecognizer = panRecognizer

            let pinchRecognizer = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            pinchRecognizer.delegate = self
            view.addGestureRecognizer(pinchRecognizer)
            self.pinchRecognizer = pinchRecognizer
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
    @ObservedObject private var rtmdetService = RTMDetModelService.shared
    @StateObject private var modelManager = USDZModelManager()
    @State private var shouldResetCamera = false
    @State private var previewCameraOffset: CGSize = .zero
    @State private var previewCameraZoom: CGFloat = 1
    @State private var showingFurnitureFit = false
    @State private var furnitureFitSegmentationMode: FurnitureFitSegmentationMode = .identifyOnly
    @State private var furnitureFitShowIdentifyLivePreview = true
    @State private var selectedFurnitureFitLabels: [String] = []
    @State private var furnitureFitInitialSegmentationDone = false
    @State private var capturedImage: UIImage?
    @State private var isCapturingSnapshot = false
    @State private var showFullVideoWithIdentifications = false
    @State private var fullVideoFurnitureTapHintVisible = false
    @State private var fullVideoSelectionHelperVisible = false
    @State private var fullVideoSelectionHelperHideTask: Task<Void, Never>?
    @State private var detectedFurnitureWidth: Float?
    @State private var furnitureFitEstimatedHeightM: Float?
    @State private var detectedFurnitureHeightAR: Float?
    @State private var latestFitCheckResult: FitCheckResult?
    @State private var latestAestheticScore: AestheticScore?
    @State private var segmentedFurnitureMeanSRGB: SIMD3<Float>?
    @State private var isPlacementIntelligenceExpanded = false
    @State private var isSavingRoom = false
    @State private var saveProgress: Double = 0
    @State private var saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
    @State private var showRoomNameInput = false
    @State private var roomName = ""
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false
    @State private var showSaveSuccessSnackbar = false
    @State private var saveSuccessSnackbarMessage = ""
    @State private var showSaveErrorNotice = false
    @State private var showDiscardUnsavedAlert = false
    @StateObject private var immersiveChrome = PaafektViewerChromeController()

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

    private var authoritativeRoomModelForMetrics: RoomModel? {
        guard let dims = depthAnythingRoomDimensions else { return nil }
        return DepthAnythingPreviewPlacementIntelligenceRoomStub.axisAlignedBoxMeters(
            width: dims.width,
            height: dims.height,
            depth: dims.depth
        )
    }

    private var canSegmentSelectedFurniture: Bool {
        showingFurnitureFit && !selectedFurnitureFitLabels.isEmpty
    }

    private var previewSceneAndFurnitureUnderlay: some View {
        ZStack {
            DepthAnythingPreviewSceneView(
                imageURL: destination.measurementImageURL,
                roomWidthMeters: destination.roomWidthMeters,
                roomHeightMeters: destination.roomHeightMeters,
                photoOrientation: destination.photoOrientation,
                cameraOffset: $previewCameraOffset,
                cameraZoom: $previewCameraZoom,
                shouldResetCamera: $shouldResetCamera,
                allowsSceneInteraction: !showingFurnitureFit
            )
            .allowsHitTesting(!showingFurnitureFit)

            if showingFurnitureFit {
                FurnitureFitUIView(
                    capturedImage: $capturedImage,
                    roomImage: nil,
                    mlModel: rtmdetService.model,
                    processInterval: 0.07,
                    active: true,
                    lockedOrientation: destination.photoOrientation,
                    roomWidthMeters: destination.roomWidthMeters,
                    roomHeightMeters: destination.roomHeightMeters,
                    roomDepthMeters: destination.roomDepthMeters,
                    onFurnitureSizeEstimated: { estimate in
                        detectedFurnitureWidth = estimate.widthMeters
                        furnitureFitEstimatedHeightM = estimate.heightMeters
                        detectedFurnitureHeightAR = estimate.arHeightMeters
                    },
                    suppressStartupProgress: furnitureFitInitialSegmentationDone,
                    onFirstSegmentationComplete: { furnitureFitInitialSegmentationDone = true },
                    onSegmentationMaskMeanColorSRGB: { meanSRGB in
                        segmentedFurnitureMeanSRGB = meanSRGB
                    },
                    // Match ModelViewer brain default: classic AVCapture preview. With AR sizing on,
                    // LiDAR devices take the AR-as-camera path (AVCapture stopped, hidden ARSCNView) → black full-video feed.
                    arAssistedSizingEnabled: false,
                    segmentationMode: furnitureFitSegmentationMode,
                    onSelectedClassLabelsChanged: { labels in
                        selectedFurnitureFitLabels = labels
                    },
                    onSegmentationModeChangeRequested: { mode in
                        logDebug("BRAIN FLOW: [DepthAnythingPreview] FurnitureFit requested segmentationMode=\(mode)")
                        furnitureFitSegmentationMode = mode
                    },
                    showIdentifyLivePreview: furnitureFitShowIdentifyLivePreview,
                    showFullVideoWithIdentificationsOverride: showFullVideoWithIdentifications
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .ignoresSafeArea()
                .zIndex(9000)
            }
        }
        .paafektImmersiveRoomSummonTap(
            chrome: immersiveChrome,
            enabled: !(showingFurnitureFit && showFullVideoWithIdentifications),
            hideForCapture: isSavingRoom || isCapturingSnapshot,
            onRestingTap: {
                if showingFurnitureFit,
                   showFullVideoWithIdentifications,
                   furnitureFitSegmentationMode == .segmentSelected {
                    furnitureFitSegmentationMode = .identifyOnly
                    furnitureFitShowIdentifyLivePreview = true
                } else {
                    immersiveChrome.summon()
                }
            }
        )
    }

    var body: some View {
        GeometryReader { _ in
            ZStack {
                previewSceneAndFurnitureUnderlay
                    .ignoresSafeArea()

                if immersiveChrome.isSummoned {
                    previewFullVideoToolbarHelperOverlay
                    previewFullVideoFurnitureTapBubbleOverlay
                }

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

                previewImmersiveChromeOverlay
                PaafektViewerOnboardingLayer(isReady: !isSavingRoom)
                    .zIndex(100_000)
            }
        }
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)
        .toolbar(.hidden, for: .navigationBar)
        .sheet(isPresented: $showRoomNameInput) {
            PaafektNameRoomSheet(
                isPresented: $showRoomNameInput,
                roomName: $roomName,
                onSave: { startSavingRoom() }
            )
        }
        .overlay {
            if showSaveErrorNotice {
                PaafektErrorNotice(isPresented: $showSaveErrorNotice, message: saveAlertMessage)
            }
        }
        .overlay(alignment: .bottom) {
            if showSaveSuccessSnackbar {
                PaafektRoomSavedSnackbar(
                    message: saveSuccessSnackbarMessage,
                    isShowing: $showSaveSuccessSnackbar
                )
            }
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
        .onChange(of: showingFurnitureFit) { _, isOn in
            handlePreviewShowingFurnitureFitChanged(isOn: isOn)
        }
        .onChange(of: furnitureFitSegmentationMode) { _, mode in
            PaafektFullVideoSegmentationExitDiagnostics.logModeChange(
                viewer: "SinglePhotoRoomViewer",
                mode: mode,
                showingFurnitureFit: showingFurnitureFit,
                showFullVideoWithIdentifications: showFullVideoWithIdentifications
            )
        }
        .onChange(of: selectedFurnitureFitLabels) { oldLabels, newLabels in
            restorePreviewFullVideoIdentifyAfterSegmentPinsLost(oldLabels: oldLabels, newLabels: newLabels)
        }
        .onChange(of: segmentedFurnitureMeanSRGB) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: detectedFurnitureWidth) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: detectedFurnitureHeightAR) { _, _ in updateRoomPlacementIntelligence() }
        .onChange(of: furnitureFitEstimatedHeightM) { _, _ in updateRoomPlacementIntelligence() }
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
            // List-page viewers preload RTMDet on appear; preview used to wait until brain tap (multi-second lag).
            rtmdetService.ensureModelLoaded()
            DepthAnythingRoomReconstructor.prewarmSharedModelIfNeeded()
            logDebug("[DepthAnythingRoom][Preview] RTMDet + Depth Anything prewarm requested on appear")
        }
        .onDisappear {
            dismissPreviewFullVideoFurnitureTapHint()
            cancelPreviewFullVideoSelectionHelper()
            OrientationLockManager.shared.unlock()
        }
    }

    private var previewRestingMeasurementPillText: String? {
        guard let dims = depthAnythingRoomDimensions else { return nil }
        if dims.width > 0.05, dims.depth > 0.05, dims.width.isFinite, dims.depth.isFinite {
            return String(format: "%.1f m × %.1f m", dims.width, dims.depth)
        }
        if dims.height > 0.05, dims.height.isFinite {
            return L10n.RoomViewer.approximateRoomHeight(dims.height)
        }
        return nil
    }

    private var previewImmersiveChromeOverlay: some View {
        PaafektImmersiveViewerChromeStack(
            chrome: immersiveChrome,
            onBack: handleBackTap,
            onFit: {
                immersiveChrome.noteChromeInteraction()
                togglePreviewFurnitureFit()
            },
            fitActive: showingFurnitureFit,
            fitDisabled: isSavingRoom,
            measurementText: previewRestingMeasurementPillText,
            hideForCapture: isSavingRoom || isCapturingSnapshot
        ) {
            PaafektImmersiveSummonedToolbar(chrome: immersiveChrome) {
                HStack(spacing: Theme.Space.sm) {
                    PaafektViewerToolbarIconButton(
                        systemName: "viewfinder",
                        accessibilityLabel: L10n.RoomViewer.recenterView
                    ) {
                        immersiveChrome.noteChromeInteraction()
                        previewCameraOffset = .zero
                        previewCameraZoom = 1
                        shouldResetCamera = true
                    }
                    if !selectedFurnitureFitLabels.isEmpty {
                        Button {
                            immersiveChrome.noteChromeInteraction()
                            NotificationCenter.default.post(
                                name: NSNotification.Name("FurnitureFitClearSelectedObjects"),
                                object: nil
                            )
                        } label: {
                            Text(selectedFurnitureChipTitle)
                                .font(.caption2)
                                .fontWeight(.semibold)
                                .foregroundColor(.white)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Capsule().fill(Theme.Palette.viewerCapsuleFill))
                        }
                        .buttonStyle(.plain)
                    }
                    if showingFurnitureFit {
                        PaafektViewerToolbarIconButton(
                            systemName: "text.viewfinder",
                            isActive: showFullVideoWithIdentifications,
                            accessibilityLabel: L10n.Settings.fullVideoWithIdentifications
                        ) {
                            immersiveChrome.noteChromeInteraction()
                            togglePreviewFullVideoIdentifications()
                        }
                    }
                    PaafektViewerToolbarIconButton(
                        systemName: "square.and.arrow.down",
                        accessibilityLabel: L10n.RoomViewer.saveRoom
                    ) {
                        immersiveChrome.noteChromeInteraction()
                        roomName = ""
                        showRoomNameInput = true
                    }
                    .disabled(isSavingRoom)
                }
            } heroContent: {
                PaafektImmersiveCompactHeroAction(
                    assetName: "PaafektIconSnapshot",
                    title: L10n.RoomViewer.immersiveCaptureShort,
                    isDisabled: isSavingRoom || isCapturingSnapshot
                ) {
                    immersiveChrome.noteChromeInteraction()
                    savePreviewSnapshot()
                }
            }
        } summonedExtras: {
            VStack(spacing: 10) {
                previewSegmentModeToggleChrome
                previewRoomIntelligencePlacementCardResetOnExit
            }
            .padding(.horizontal, Theme.Space.lg)
        } restingAccessory: {
            EmptyView()
        } persistentOverlay: {
            PaafektFurnitureFitDonePersistentOverlay(
                showingFurnitureFit: showingFurnitureFit,
                showFullVideoWithIdentifications: showFullVideoWithIdentifications,
                segmentationMode: furnitureFitSegmentationMode,
                viewerLabel: "SinglePhotoRoomViewer",
                onExitFullVideoSegmentation: {
                    furnitureFitSegmentationMode = .identifyOnly
                    furnitureFitShowIdentifyLivePreview = true
                },
                onExitFurnitureFit: togglePreviewFurnitureFit
            )
        }
        .zIndex(99_998)
    }

    private var selectedFurnitureChipTitle: String {
        let labels = selectedFurnitureFitLabels
        if labels.count == 1 { return labels[0] }
        if labels.count == 2 { return "\(labels[0]), \(labels[1])" }
        return "\(labels.count) selected"
    }

    @ViewBuilder
    private var previewSegmentModeToggleChrome: some View {
        Group {
            if showingFurnitureFit && showFullVideoWithIdentifications,
               furnitureFitSegmentationMode != .segmentSelected {
                Button(action: {
                    guard canSegmentSelectedFurniture else { return }
                    furnitureFitSegmentationMode = .segmentSelected
                }) {
                    Text(L10n.RoomViewer.segmentFurnitureAction)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .frame(height: 44)
                    .background(
                        Capsule().fill(
                            canSegmentSelectedFurniture
                                ? Color.orange
                                : Color.black.opacity(0.45)
                        )
                    )
                    .shadow(radius: 4)
                }
                .disabled(!canSegmentSelectedFurniture)
                .accessibilityLabel(L10n.RoomViewer.segmentFurnitureAccessibility)
            }
        }
    }

    private var previewFullVideoToolbarHelperOverlay: some View {
        ZStack(alignment: .topTrailing) {
            Color.clear
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(false)
            if showingFurnitureFit &&
                !showFullVideoWithIdentifications &&
                furnitureFitSegmentationMode == .segmentPrimary &&
                fullVideoSelectionHelperVisible {
                PaafektHintChip(
                    systemImage: "text.viewfinder",
                    text: L10n.RoomViewer.fullVideoSelectionHelper,
                    maxWidth: 220
                )
                .padding(.top, 6)
                .padding(.trailing, 54)
                .offset(y: 108)
            }
        }
        .allowsHitTesting(false)
        .zIndex(106)
    }

    private var previewFullVideoFurnitureTapBubbleOverlay: some View {
        Group {
            if fullVideoFurnitureTapHintVisible {
                VStack {
                    PaafektHintChip(
                        systemImage: "hand.tap.fill",
                        text: L10n.RoomViewer.fullVideoFurnitureTapHint,
                        maxWidth: 280
                    )
                    .padding(.top, 12)
                    Spacer()
                }
                .allowsHitTesting(false)
                .zIndex(105)
            }
        }
    }

    private func handlePreviewShowingFurnitureFitChanged(isOn: Bool) {
        if isOn {
            rtmdetService.ensureModelLoaded()
            updateRoomPlacementIntelligence()
            presentPreviewFullVideoSelectionHelperIfNeeded()
        } else {
            dismissPreviewFullVideoFurnitureTapHint()
            cancelPreviewFullVideoSelectionHelper()
            showFullVideoWithIdentifications = false
            furnitureFitSegmentationMode = .identifyOnly
            furnitureFitShowIdentifyLivePreview = true
            selectedFurnitureFitLabels = []
            capturedImage = nil
            detectedFurnitureWidth = nil
            furnitureFitEstimatedHeightM = nil
            detectedFurnitureHeightAR = nil
            latestFitCheckResult = nil
            latestAestheticScore = nil
            segmentedFurnitureMeanSRGB = nil
            isPlacementIntelligenceExpanded = false
        }
    }

    private func placementIntelligenceRingColor(fit: FitCheckResult?) -> Color {
        guard let fit else { return .cyan }
        return fit.fitsInRoom ? .green : .red
    }

    @ViewBuilder
    private func placementIntelligenceExpandedContent(
        dimensions: RoomFurnitureDimensions?,
        fit: FitCheckResult?,
        aesthetic: AestheticScore
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(L10n.RoomViewer.placementIntelligenceTitle)
                    .font(.caption.bold())
                    .foregroundColor(.white)
                Spacer(minLength: 4)
                if let fit {
                    Text(
                        fit.fitsInRoom
                            ? L10n.RoomViewer.placementFitCount(max(fit.fitLocations.count, 1))
                            : L10n.RoomViewer.placementNoFit
                    )
                    .font(.caption2.bold())
                    .foregroundColor(fit.fitsInRoom ? .green : .red)
                } else {
                    Text(L10n.RoomViewer.placementBadgeStyleOnly)
                        .font(.caption2.bold())
                        .foregroundColor(.cyan.opacity(0.95))
                }
            }
            if dimensions == nil {
                Text(L10n.RoomViewer.placementMetricUnavailableNote)
                    .font(.caption2)
                    .foregroundColor(.gray)
            }
            if let dimensions {
                Text(
                    L10n.RoomViewer.placementDetectedSizeMeters(
                        width: Double(dimensions.widthM),
                        height: Double(dimensions.heightM),
                        depth: Double(dimensions.depthM)
                    )
                )
                .font(.caption2)
                .foregroundColor(.white.opacity(0.92))
            }
            if let fit {
                Text(fit.fitsInRoom ? L10n.RoomViewer.placementFitsRoom : L10n.RoomViewer.placementExceedsRoom)
                    .font(.caption2)
                    .foregroundColor(fit.fitsInRoom ? .green : .red)
            }
            Text(
                L10n.RoomViewer.placementHarmonySummary(
                    harmonyScore: aesthetic.harmonyScore,
                    harmonyTypeName: aesthetic.harmonyType.localizedDisplayName,
                    contrastScore: aesthetic.contrastScore,
                    styleFit: aesthetic.styleCompatibilityScore
                )
            )
            .font(.caption2)
            .foregroundColor(.white.opacity(0.88))
            ForEach(Array(aesthetic.recommendations.prefix(4).enumerated()), id: \.offset) { _, line in
                Text("• \(line)")
                    .font(.caption2)
                    .foregroundColor(.white.opacity(0.86))
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: 300, alignment: .leading)
        .background(Color.black.opacity(0.88))
        .cornerRadius(14)
        .shadow(color: .black.opacity(0.45), radius: 8, x: 0, y: 4)
    }

    private func derivedDetectedFurnitureDimensionsForRoomIntelligence() -> RoomFurnitureDimensions? {
        guard let width = detectedFurnitureWidth, width.isFinite, width > 0.05 else { return nil }
        let height = furnitureFitEstimatedHeightM ?? detectedFurnitureHeightAR
        guard let height, height.isFinite, height > 0.05 else { return nil }
        let estimatedDepth = max(0.25, min(width * 0.72, 1.4))
        return RoomFurnitureDimensions(widthM: width, heightM: height, depthM: estimatedDepth)
    }

    private func inferredRoomStyleTags(from palette: SurfacePalette) -> [String] {
        var tags = Set<String>()
        let layers = [palette.floor, palette.walls, palette.ceiling]
        for layer in layers {
            guard let layer else { continue }
            switch layer.hint {
            case .wood: tags.formUnion(["rustic", "traditional"])
            case .tile: tags.insert("modern")
            case .concrete: tags.formUnion(["industrial", "modern"])
            case .carpet: tags.formUnion(["traditional", "eclectic"])
            case .plaster: tags.formUnion(["modern", "scandinavian"])
            case .brick: tags.formUnion(["traditional", "industrial"])
            case .marble: tags.formUnion(["modern", "luxury"])
            case .unknown: break
            }
        }
        if tags.isEmpty { return ["modern", "minimalist"] }
        return Array(tags).sorted().prefix(6).map { $0 }
    }

    private func heuristicFurnitureProfileForAesthetic(
        roomModel: RoomModel,
        segmentedMeanSRGB: SIMD3<Float>?
    ) -> FurnitureProfile {
        let palette = roomModel.surfacePalette
        let primary: SIMD3<Float>
        if let cutoutMean = segmentedMeanSRGB {
            primary = cutoutMean
        } else if let wall = palette.walls?.dominantColors.first {
            primary = SIMD3(
                min(wall.x * 0.82 + 0.06, 1),
                min(wall.y * 0.78 + 0.05, 1),
                min(wall.z * 0.74 + 0.04, 1)
            )
        } else if let floor = palette.floor?.dominantColors.first {
            primary = SIMD3(repeating: 0.38) * 0.55 + floor * 0.45
        } else if let ceiling = palette.ceiling?.dominantColors.first {
            primary = ceiling * SIMD3(0.55, 0.52, 0.48)
        } else {
            primary = SIMD3(0.44, 0.40, 0.36)
        }
        return FurnitureProfile(
            primaryColor: primary,
            accentColor: nil,
            styleTags: ["modern", "minimalist", "contemporary"]
        )
    }

    private func updateRoomPlacementIntelligence() {
        guard showingFurnitureFit, let roomModel = authoritativeRoomModelForMetrics else {
            latestFitCheckResult = nil
            latestAestheticScore = nil
            return
        }
        if let furniture = derivedDetectedFurnitureDimensionsForRoomIntelligence() {
            let fitEngine = FitCheckEngine(roomModel: roomModel)
            latestFitCheckResult = fitEngine.checkFit(furniture: furniture)
        } else {
            latestFitCheckResult = nil
        }
        let palette = roomModel.surfacePalette
        let roomStyleTags = inferredRoomStyleTags(from: palette)
        let furnitureProfile = heuristicFurnitureProfileForAesthetic(
            roomModel: roomModel,
            segmentedMeanSRGB: segmentedFurnitureMeanSRGB
        )
        let aestheticAdvisor = AestheticAdvisor(palette: palette, roomStyleTags: roomStyleTags)
        latestAestheticScore = aestheticAdvisor.evaluate(furniture: furnitureProfile)
    }

    @ViewBuilder
    private var previewRoomIntelligencePlacementCardResetOnExit: some View {
        if showingFurnitureFit, authoritativeRoomModelForMetrics != nil {
            let dimensions = derivedDetectedFurnitureDimensionsForRoomIntelligence()
            let fit = latestFitCheckResult
            VStack(spacing: 10) {
                if isPlacementIntelligenceExpanded, let aesthetic = latestAestheticScore {
                    placementIntelligenceExpandedContent(dimensions: dimensions, fit: fit, aesthetic: aesthetic)
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
                Button {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        isPlacementIntelligenceExpanded.toggle()
                    }
                } label: {
                    ZStack {
                        Circle()
                            .fill(
                                LinearGradient(
                                    colors: [Color(white: 0.22), Color(white: 0.12)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 46, height: 46)
                            .overlay(
                                Circle()
                                    .stroke(placementIntelligenceRingColor(fit: fit), lineWidth: 2.5)
                            )
                        Image(systemName: "square.split.2x2.fill")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundStyle(.white)
                            .symbolRenderingMode(.hierarchical)
                    }
                }
                .buttonStyle(.plain)
                .accessibilityLabel(L10n.RoomViewer.placementIntelligenceTitle)
                .accessibilityAddTraits(.isButton)
            }
            .onChange(of: showingFurnitureFit) { _, isShowing in
                if !isShowing { isPlacementIntelligenceExpanded = false }
            }
            .onChange(of: latestFitCheckResult?.fitsInRoom) { _, _ in
                if latestFitCheckResult == nil, latestAestheticScore == nil {
                    isPlacementIntelligenceExpanded = false
                }
            }
        }
    }

    private func dismissPreviewFullVideoFurnitureTapHint() {
        fullVideoFurnitureTapHintVisible = false
    }

    private func presentPreviewFullVideoFurnitureTapHintIfNeeded() {
        guard showFullVideoWithIdentifications else { return }
        fullVideoFurnitureTapHintVisible = true
    }

    private func cancelPreviewFullVideoSelectionHelper() {
        fullVideoSelectionHelperHideTask?.cancel()
        fullVideoSelectionHelperHideTask = nil
        fullVideoSelectionHelperVisible = false
    }

    private func presentPreviewFullVideoSelectionHelperIfNeeded() {
        guard showingFurnitureFit,
              !showFullVideoWithIdentifications,
              furnitureFitSegmentationMode == .segmentPrimary else {
            cancelPreviewFullVideoSelectionHelper()
            return
        }
        fullVideoSelectionHelperHideTask?.cancel()
        fullVideoSelectionHelperVisible = true
        fullVideoSelectionHelperHideTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(3))
            guard !Task.isCancelled else { return }
            fullVideoSelectionHelperVisible = false
        }
    }

    private func restorePreviewFullVideoIdentifyAfterSegmentPinsLost(oldLabels: [String], newLabels: [String]) {
        guard showingFurnitureFit else { return }
        guard furnitureFitSegmentationMode == .segmentSelected else { return }
        guard newLabels.isEmpty, !oldLabels.isEmpty else { return }
        dismissPreviewFullVideoFurnitureTapHint()
        showFullVideoWithIdentifications = true
        furnitureFitSegmentationMode = .identifyOnly
        furnitureFitShowIdentifyLivePreview = true
        presentPreviewFullVideoFurnitureTapHintIfNeeded()
    }

    private func togglePreviewFurnitureFit() {
        if showingFurnitureFit {
            dismissPreviewFullVideoFurnitureTapHint()
            cancelPreviewFullVideoSelectionHelper()
            showFullVideoWithIdentifications = false
            showingFurnitureFit = false
        } else {
            rtmdetService.ensureModelLoaded()
            showFullVideoWithIdentifications = false
            furnitureFitSegmentationMode = .segmentPrimary
            furnitureFitShowIdentifyLivePreview = true
            selectedFurnitureFitLabels = []
            furnitureFitInitialSegmentationDone = false
            showingFurnitureFit = true
        }
    }

    private func togglePreviewFullVideoIdentifications() {
        showFullVideoWithIdentifications.toggle()
        dismissPreviewFullVideoFurnitureTapHint()
        if showFullVideoWithIdentifications {
            cancelPreviewFullVideoSelectionHelper()
            furnitureFitSegmentationMode = .identifyOnly
            furnitureFitShowIdentifyLivePreview = true
            presentPreviewFullVideoFurnitureTapHintIfNeeded()
        } else {
            furnitureFitSegmentationMode = .segmentPrimary
            furnitureFitShowIdentifyLivePreview = true
            presentPreviewFullVideoSelectionHelperIfNeeded()
        }
    }

    private func savePreviewSnapshot() {
        isCapturingSnapshot = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.06) {
            guard let uiImage = capturePreviewAppWindowImage() else {
                isCapturingSnapshot = false
                logDebug("❌ [DepthAnythingPreview] Failed to capture snapshot")
                return
            }
            savePreviewUIImageToPhotos(uiImage)
            DispatchQueue.main.async {
                isCapturingSnapshot = false
            }
        }
    }

    private func capturePreviewAppWindowImage() -> UIImage? {
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let windows = scenes.flatMap(\.windows)
        guard let window = windows.first(where: \.isKeyWindow) ?? windows.first else {
            return nil
        }
        let format = UIGraphicsImageRendererFormat()
        format.scale = window.traitCollection.displayScale
        let renderer = UIGraphicsImageRenderer(bounds: window.bounds, format: format)
        return renderer.image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }
    }

    private func savePreviewUIImageToPhotos(_ image: UIImage) {
        let saveBlock = {
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: image)
            }, completionHandler: { success, error in
                DispatchQueue.main.async {
                    if success {
                        logDebug("✅ [DepthAnythingPreview] Snapshot saved to Photos")
                    } else {
                        logDebug("❌ [DepthAnythingPreview] Snapshot save failed: \(error?.localizedDescription ?? "unknown")")
                    }
                }
            })
        }

        if #available(iOS 14, *) {
            PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
                DispatchQueue.main.async {
                    switch status {
                    case .authorized, .limited:
                        saveBlock()
                    default:
                        logDebug("❌ [DepthAnythingPreview] Photos access not granted")
                    }
                }
            }
        } else {
            switch PHPhotoLibrary.authorizationStatus() {
            case .authorized, .limited:
                saveBlock()
            case .notDetermined:
                PHPhotoLibrary.requestAuthorization { status in
                    DispatchQueue.main.async {
                        if status == .authorized || status == .limited {
                            saveBlock()
                        }
                    }
                }
            default:
                logDebug("❌ [DepthAnythingPreview] Photos access denied")
            }
        }
    }

    private var saveRoomProgressOverlay: some View {
        PaafektSavingRoomOverlay(
            progress: saveProgress,
            title: L10n.RoomViewer.savingRoom,
            subtitle: saveProgressStatusText
        )
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
            showSaveErrorNotice = true
            return
        }
        let measurementImageURL = destination.measurementImageURL
        guard FileManager.default.fileExists(atPath: measurementImageURL.path) else {
            saveAlertMessage = L10n.RoomPreview.sourceImageUnavailable
            saveWasSuccessful = false
            showSaveErrorNotice = true
            return
        }
        let photoOrientationRawValue = destination.photoOrientation.rawValue
        let roomCoordinateFrameRawValue = destination.roomCoordinateFrame.rawValue

        showRoomNameInput = false
        withAnimation(.easeIn(duration: 0.2)) {
            isSavingRoom = true
            saveProgress = 0.02
            saveProgressStatusText = L10n.RoomViewer.measuringRoom
        }

        Task {
            // Commit overlay before heavy first-save CoreML work (matches SplatRoomView save).
            try? await Task.sleep(nanoseconds: 220_000_000)

            do {
                let savedURL = try await Task.detached(priority: .userInitiated) {
                    let cameraMetadata = CameraExifSidecar.load(roomURL: measurementImageURL)
                    guard let measurementImage = UIImage(contentsOfFile: measurementImageURL.path)?.fixedOrientation() else {
                        throw DepthAnythingPreviewSaveError.sourceImageUnavailable
                    }

                    let reconstructor = try DepthAnythingRoomReconstructor()
                    let measuredResult = try await reconstructor.reconstructWithResult(
                        image: measurementImage,
                        cameraMetadata: cameraMetadata.isEmpty ? nil : cameraMetadata,
                        progressHandler: { fraction in
                            Task { @MainActor in
                                saveProgress = 0.05 + Double(fraction) * 0.85
                                if fraction < 0.58 {
                                    saveProgressStatusText = L10n.RoomViewer.measuringRoom
                                } else if fraction < 0.85 {
                                    saveProgressStatusText = L10n.GenerationProgress.generating3DModel
                                } else {
                                    saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
                                }
                            }
                        }
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
                    saveSuccessSnackbarMessage = L10n.RoomViewer.saveSuccess(trimmedRoomName)
                    saveWasSuccessful = true
                    withAnimation { showSaveSuccessSnackbar = true }
                    roomName = ""
                    logDebug("✅ [DepthAnythingRoom] Saved room to \(savedURL.lastPathComponent)")
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                        NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil)
                    }
                }
            } catch {
                await MainActor.run {
                    isSavingRoom = false
                    saveProgress = 0
                    saveProgressStatusText = L10n.RoomViewer.savingRoomEllipsis
                    saveAlertMessage = error.localizedDescription
                    saveWasSuccessful = false
                    showSaveErrorNotice = true
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
            return L10n.RoomPreview.sourceImageUnavailable
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
    @State private var selectedImagePreview: UIImage?
    @State private var selectedImagePreviewToken = UUID()
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
    @State private var hasRequestedRTMDetPrewarm = false

    struct IdentifiedImage: Identifiable {
        let id = UUID()
        let image: UIImage
    }

    var body: some View {
        ZStack {
            VStack {
                if selectedImage != nil {
                    // Downsampled preview avoids decoding a 12MP+ UIImage on the main thread.
                    Group {
                        if let preview = selectedImagePreview {
                            Image(uiImage: preview)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                        } else {
                            ProgressView()
                                .frame(maxWidth: .infinity, minHeight: 180)
                        }
                    }
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
                        guard let image = selectedImage else { return }
                        logDebug("🤖 [View] Depth Anything method selected")
                        logDebug("📸 User selected pic type: \(selectedOrientation == .portrait ? "Portrait" : "Landscape")")
                        startDepthAnythingGeneration(image: image)
                    }) {
                        HStack(spacing: 16) {
                            Image(systemName: "camera.metering.matrix")
                                .font(.system(size: 30))
                                .foregroundStyle(Theme.Palette.accent)
                                .frame(width: 50)

                            VStack(alignment: .leading, spacing: 4) {
                                Text(L10n.PhotoRoom.title)
                                    .font(Theme.Typo.headline())
                                    .foregroundStyle(Theme.Palette.textPrimary)
                                Text(L10n.PhotoRoom.aiPowered)
                                    .font(Theme.Typo.caption())
                                    .foregroundStyle(Theme.Palette.textSecondary)
                            }
                            Spacer()
                            Image(systemName: "chevron.right")
                                .foregroundStyle(Theme.Palette.textSecondary)
                        }
                    }
                    .buttonStyle(PaafektCreationCardStyle(variant: .primary))
                    .padding(.horizontal)

                    Button(action: {
                        guard let image = selectedImage else { return }
                        logDebug("🏠 [View] Manual setup selected")
                        logDebug("📸 User selected pic type: \(selectedOrientation == .portrait ? "Portrait" : "Landscape")")
                        fixedImageItem = IdentifiedImage(image: image)
                    }) {
                        HStack(spacing: 16) {
                            Image(systemName: "square.resize")
                                .font(.system(size: 30))
                                .foregroundStyle(Theme.Palette.textPrimary)
                                .frame(width: 50)

                            VStack(alignment: .leading, spacing: 4) {
                                Text(L10n.PhotoRoom.manualSetup)
                                    .font(Theme.Typo.headline())
                                    .foregroundStyle(Theme.Palette.textPrimary)
                                Text(L10n.PhotoRoom.manualSetupDesc)
                                    .font(Theme.Typo.caption())
                                    .foregroundStyle(Theme.Palette.textSecondary)
                            }
                            Spacer()
                            Image(systemName: "chevron.right")
                                .foregroundStyle(Theme.Palette.textSecondary)
                        }
                    }
                    .buttonStyle(PaafektCreationCardStyle(variant: .secondary))
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
                                    .foregroundStyle(Theme.Palette.accent)

                                VStack(spacing: 4) {
                                    Text(L10n.Camera.takePhoto)
                                        .font(Theme.Typo.headline())
                                    Text(L10n.Camera.chooseOrientationShort)
                                        .font(Theme.Typo.caption())
                                        .foregroundStyle(Theme.Palette.textSecondary)
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .padding(Theme.Space.xl)
                        }
                        .buttonStyle(PaafektCreationCardStyle(variant: .primary))
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
                                    .foregroundStyle(Theme.Palette.textPrimary)

                                VStack(spacing: 4) {
                                    Text(L10n.PhotoRoom.selectPhoto)
                                        .font(Theme.Typo.headline())
                                    Text(L10n.PhotoRoom.fromLibrary)
                                        .font(Theme.Typo.caption())
                                        .foregroundStyle(Theme.Palette.textSecondary)
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .padding(Theme.Space.xl)
                        }
                        .buttonStyle(PaafektCreationCardStyle(variant: .secondary))
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
                PaafektBuildingRoomOverlay(
                    progress: Double(reconstructor.progress),
                    statusMessage: reconstructor.statusMessage
                )
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
                selectedImagePreview = nil
                sourceImageURL = nil
                captureMediaMetadata = nil
                photoLibraryAssetLocalId = nil
                supplementalCameraDoubles = nil
            }
            guard let image = newValue else { return }
            logDebug("✅ [View] Image selected")
            let previewToken = UUID()
            selectedImagePreviewToken = previewToken
            selectedImagePreview = nil

            Task { @MainActor in
                // Let SwiftUI paint the AI / Manual buttons before any follow-up work.
                await Task.yield()
                selectedOrientation = PhotoOrientation.detect(from: image)
                logDebug("📐 [View] Auto-detected orientation: \(selectedOrientation.rawValue)")
                prewarmDepthAnythingModelIfNeeded()
                logDebug("🤖 [View] Depth Anything prewarm requested (RTMDet deferred until Create)")
            }

            Task.detached(priority: .userInitiated) {
                let preview = downsampleUIImageForDisplay(image, maxDimension: 960)
                await MainActor.run {
                    guard selectedImagePreviewToken == previewToken else { return }
                    selectedImagePreview = preview
                }
            }
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
        DepthAnythingRoomReconstructor.prewarmSharedModelIfNeeded()
    }

    private func prewarmRTMDetModelIfNeeded() {
        guard !hasRequestedRTMDetPrewarm else { return }
        hasRequestedRTMDetPrewarm = true
        RTMDetModelService.shared.ensureModelLoaded()
        logDebug("[RTMDet][Prewarm] requested from photo room flow")
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
        prewarmRTMDetModelIfNeeded()

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
    @State private var saveAlertMessage = ""
    @State private var saveWasSuccessful = false
    @State private var showSaveSuccessSnackbar = false
    @State private var saveSuccessSnackbarMessage = ""
    @State private var showSaveErrorNotice = false
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
        .sheet(isPresented: $showRoomNameInput) {
            PaafektNameRoomSheet(
                isPresented: $showRoomNameInput,
                roomName: $roomName,
                onSave: { startSavingRoom() }
            )
        }
        .overlay {
            if showSaveErrorNotice {
                PaafektErrorNotice(isPresented: $showSaveErrorNotice, message: saveAlertMessage)
            }
        }
        .overlay(alignment: .bottom) {
            if showSaveSuccessSnackbar {
                PaafektRoomSavedSnackbar(
                    message: saveSuccessSnackbarMessage,
                    isShowing: $showSaveSuccessSnackbar
                )
            }
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
        PaafektSavingRoomOverlay(
            progress: saveProgress,
            title: L10n.RoomViewer.savingRoom,
            subtitle: saveProgressMessage
        )
        .overlay(alignment: .bottom) {
            Button(L10n.Common.cancel) {
                cancelSavingRoom()
            }
            .foregroundStyle(Theme.Palette.danger)
            .padding(.bottom, Theme.Space.xxl)
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
            showSaveErrorNotice = true
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
                        self.saveSuccessSnackbarMessage = L10n.RoomViewer.saveSuccess(savedName)
                        self.saveWasSuccessful = true
                        withAnimation { self.showSaveSuccessSnackbar = true }
                        self.roomName = ""
                        logDebug("✅ [Viewer] Save complete!")
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                            NotificationCenter.default.post(name: NSNotification.Name("DismissPhotoRoomSheet"), object: nil)
                        }
                    } else {
                        self.saveAlertMessage = L10n.RoomViewer.saveFailed(saveError ?? "Unknown error")
                        self.saveWasSuccessful = false
                        self.showSaveErrorNotice = true
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
