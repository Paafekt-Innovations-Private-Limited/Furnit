import SwiftUI
import SceneKit
import Accelerate

// MARK: - Room Boundary Detection View with DRAGGABLE boundaries
struct RoomBoundaryDetectionView: View {
    let originalImage: UIImage
    @Binding var savedBoundaries: RoomStructure?
    @State private var detectedBoundariesImage: UIImage?

    // ✅ Fix image orientation ONCE to prevent 90° tilt
    @State private var fixedImage: UIImage?
    private var displayImage: UIImage { fixedImage ?? originalImage }

    // GPU-accelerated CIContext for image processing
    private static let ciContext: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() {
            print("🚀 [BoundaryView] Using Metal GPU for image processing")
            return CIContext(mtlDevice: device, options: [.useSoftwareRenderer: false])
        }
        print("⚠️ [BoundaryView] Metal not available, using CPU")
        return CIContext(options: [.useSoftwareRenderer: true])
    }()
    @State private var scale: CGFloat = 1.0
    @State private var lastScale: CGFloat = 1.0
    @State private var offset: CGSize = .zero
    @State private var lastOffset: CGSize = .zero
    @Environment(\.dismiss) var dismiss
    
    // Boundary positions (as percentages of image dimensions)
    @State private var floorY: CGFloat = 0.85
    @State private var ceilingY: CGFloat = 0.15
    @State private var leftX: CGFloat = 0.12
    @State private var rightX: CGFloat = 0.88
    @State private var vanishingX: CGFloat = 0.5
    @State private var vanishingY: CGFloat = 0.45
    
    @State private var showAdjustmentMode = false
    
    // Custom magenta color
    private let magentaColor = Color(red: 1.0, green: 0.0, blue: 1.0)
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                if showAdjustmentMode {
                    // Interactive adjustment view
                    GeometryReader { geometry in
                        ZStack {
                            // Background image (orientation-fixed)
                            Image(uiImage: displayImage)
                                .resizable()
                                .scaledToFit()
                                .frame(width: geometry.size.width)

                            // Overlay with draggable boundaries
                            BoundaryLinesCanvas(
                                imageSize: displayImage.size,
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
                                imageSize: displayImage.size,
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
                        Text("Drag the circles to adjust boundaries")
                            .font(.headline)
                            .padding(.top, 8)
                        
                        HStack(spacing: 16) {
                            Label("Floor", systemImage: "arrow.down")
                                .foregroundColor(.green)
                                .font(.caption)
                            Label("Ceiling", systemImage: "arrow.up")
                                .foregroundColor(.cyan)
                                .font(.caption)
                            Label("Walls", systemImage: "arrow.left.and.right")
                                .foregroundColor(.red)
                                .font(.caption)
                            Label("Vanish", systemImage: "scope")
                                .foregroundColor(magentaColor)
                                .font(.caption)
                        }
                        .padding(.horizontal)
                        
                        HStack(spacing: 20) {
                            Button("Reset") {
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
                            
                            Button("Done Adjusting") {
                                // ✅ SAVE BOUNDARIES HERE
                                var boundaries = RoomStructure()
                                boundaries.floorY = floorY
                                boundaries.ceilingY = ceilingY
                                boundaries.leftX = leftX
                                boundaries.rightX = rightX
                                boundaries.vanishingX = vanishingX
                                boundaries.vanishingY = vanishingY
                                
                                savedBoundaries = boundaries
                                print("✅ Saved adjusted boundaries:")
                                print("   Floor: \(floorY), Ceiling: \(ceilingY)")
                                print("   Left: \(leftX), Right: \(rightX)")
                                print("   VP: (\(vanishingX), \(vanishingY))")
                                
                                showAdjustmentMode = false
                                generateFinalImage()
                            }
                            .buttonStyle(.borderedProminent)
                        }
                        .padding(.bottom, 8)
                    }
                    .background(.ultraThinMaterial)
                    
                } else if let boundariesImage = detectedBoundariesImage {
                    // View mode with zoom controls
                    GeometryReader { geometry in
                        ScrollView([.horizontal, .vertical], showsIndicators: true) {
                            Image(uiImage: boundariesImage)
                                .resizable()
                                .scaledToFit()
                                .frame(width: geometry.size.width)
                                .scaleEffect(scale)
                                .offset(offset)
                                .gesture(
                                    MagnificationGesture()
                                        .onChanged { value in
                                            let delta = value / lastScale
                                            lastScale = value
                                            scale *= delta
                                        }
                                        .onEnded { _ in
                                            lastScale = 1.0
                                            scale = min(max(scale, 0.5), 5.0)
                                        }
                                )
                                .gesture(
                                    DragGesture()
                                        .onChanged { value in
                                            offset = CGSize(
                                                width: lastOffset.width + value.translation.width,
                                                height: lastOffset.height + value.translation.height
                                            )
                                        }
                                        .onEnded { _ in
                                            lastOffset = offset
                                        }
                                )
                        }
                    }
                    
                    // Zoom controls
                    HStack(spacing: 20) {
                        Button(action: {
                            withAnimation { scale = max(0.5, scale - 0.5) }
                        }) {
                            Image(systemName: "minus.magnifyingglass")
                                .font(.title2)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .clipShape(Circle())
                        }
                        
                        Button("Adjust Boundaries") {
                            showAdjustmentMode = true
                        }
                        .font(.headline)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(Color.orange)
                        .foregroundColor(.white)
                        .cornerRadius(20)
                        
                        Button(action: {
                            withAnimation {
                                scale = 1.0
                                offset = .zero
                                lastOffset = .zero
                            }
                        }) {
                            Text("Reset")
                                .font(.headline)
                                .padding(.horizontal, 20)
                                .padding(.vertical, 10)
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(20)
                        }
                        
                        Button(action: {
                            withAnimation { scale = min(5.0, scale + 0.5) }
                        }) {
                            Image(systemName: "plus.magnifyingglass")
                                .font(.title2)
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .clipShape(Circle())
                        }
                    }
                    .padding()
                    
                } else {
                    ProgressView("Detecting room boundaries...")
                        .padding()
                }
            }
            .navigationTitle("Room Boundaries")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .onAppear {
            // ✅ Fix image orientation ONCE on appear to prevent 90° tilt
            if fixedImage == nil {
                fixedImage = originalImage.fixedOrientation()
                print("🔧 [BoundaryView] Fixed image orientation on appear")
            }
            generateFinalImage()
        }
    }
    
    func generateFinalImage() {
        Task {
            let result = await drawBoundariesOnImage()
            await MainActor.run { detectedBoundariesImage = result }
        }
    }
    
    func drawBoundariesOnImage() async -> UIImage {
        // ✅ Use displayImage (orientation-fixed) instead of originalImage
        let sourceImage = displayImage

        // ✅ OPTIMIZATION: Downscale large images to prevent memory crashes
        // Using vImage from Accelerate framework for GPU/NEON acceleration
        let maxDimension: CGFloat = 1600  // Max 1600px - balances quality & memory
        let originalWidth = sourceImage.size.width
        let originalHeight = sourceImage.size.height
        let scaleFactor = min(maxDimension / max(originalWidth, originalHeight), 1.0)

        let workingImage: UIImage
        if scaleFactor < 1.0 {
            print("🚀 [BoundaryView] Downscaling \(Int(originalWidth))x\(Int(originalHeight)) → \(Int(originalWidth * scaleFactor))x\(Int(originalHeight * scaleFactor))")
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
            print("❌ vImage source buffer init failed: \(error)")
            return nil
        }
        defer { free(sourceBuffer.data) }

        var destBuffer = vImage_Buffer()
        error = vImageBuffer_Init(&destBuffer, vImagePixelCount(newHeight), vImagePixelCount(newWidth), 32, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else {
            print("❌ vImage dest buffer init failed: \(error)")
            return nil
        }
        defer { free(destBuffer.data) }

        // High-quality Lanczos scaling
        error = vImageScale_ARGB8888(&sourceBuffer, &destBuffer, nil, vImage_Flags(kvImageHighQualityResampling))
        guard error == kvImageNoError else {
            print("❌ vImage scale failed: \(error)")
            return nil
        }

        guard let scaledCGImage = vImageCreateCGImageFromBuffer(&destBuffer, &format, nil, nil, vImage_Flags(kvImageNoFlags), &error)?.takeRetainedValue() else {
            print("❌ vImage CGImage creation failed: \(error)")
            return nil
        }

        print("✅ [vImage] Downscaled to \(newWidth)x\(newHeight)")
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

struct SinglePhotoRoomView: View {
    @StateObject private var reconstructor = SinglePhotoRoomReconstructor()
    @State private var selectedImage: UIImage?
    @State private var showImagePicker = false
    @State private var showRoomBoundaries = false
    @State private var adjustedBoundaries: RoomStructure?
    @State private var adjustedWidth: Float = 4.0
    @State private var adjustedDepth: Float = 4.0
    @State private var adjustedHeight: Float = 2.8
    
    // NEW: State for Option 2 (3D Room Scan)
    @State private var show3DScanOption = false
    @State private var showUnsupportedAlert = false
    
    var body: some View {
        ZStack {
            VStack {
            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 300)
                    .cornerRadius(12)
                    .padding()
                    .onAppear { print("🖼️ [View] Displaying selected image") }
                
                Button("Show Room Boundaries") {
                    print("🏠 [View] Room boundaries button tapped")
                    showRoomBoundaries = true
                }
                .buttonStyle(.borderedProminent)
                .padding()
                
            } else {
                // NEW: Show two options instead of just photo picker
                VStack(spacing: 20) {
                    Text("Choose Your Method")
                        .font(.title2)
                        .fontWeight(.bold)
                        .padding(.top, 40)
                    
                    Text("Create a 3D room model")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    // Option 1: Photo Selection (Existing)
                    Button(action: {
                        print("🖼️ [View] Select photo button tapped")
                        showImagePicker = true
                    }) {
                        VStack(spacing: 16) {
                            Image(systemName: "photo.on.rectangle")
                                .font(.system(size: 50))
                                .foregroundColor(.orange)
                            
                            VStack(spacing: 4) {
                                Text("Quick Photo")
                                    .font(.headline)
                                Text("Single photo capture")
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
                    
                    // Option 2: 3D Room Scan (NEW)
                    Button(action: {
                        if #available(iOS 16.0, *) {
                            if RoomCaptureSession.isSupported {
                                print("📷 [View] 3D Room Scan button tapped")
                                show3DScanOption = true
                            } else {
                                showUnsupportedAlert = true
                            }
                        } else {
                            showUnsupportedAlert = true
                        }
                    }) {
                        VStack(spacing: 16) {
                            Image(systemName: "camera.metering.matrix")
                                .font(.system(size: 50))
                                .foregroundColor(.blue)
                            
                            VStack(spacing: 4) {
                                Text("3D Room Scan")
                                    .font(.headline)
                                Text("Camera scanning with RoomPlan")
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
                    
                    Spacer()
                }
            }
            
            if reconstructor.isProcessing {
                VStack {
                    ProgressView(value: reconstructor.progress)
                        .progressViewStyle(LinearProgressViewStyle())
                        .padding(.horizontal)
                    
                    Text(reconstructor.statusMessage)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .onAppear { print("⏳ [View] Processing view appeared") }
            }
            
            if let dimensions = reconstructor.estimatedDimensions {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Estimated Dimensions")
                        .font(.headline)
                    
                    HStack {
                        Image(systemName: "arrow.left.and.right")
                        Text("Width: \(String(format: "%.1f", adjustedWidth))m")
                        Slider(value: $adjustedWidth, in: 2...8, step: 0.1)
                            .onChange(of: adjustedWidth) { oldValue, newValue in
                                print("📏 [View] Width adjusted: \(oldValue) -> \(newValue)")
                            }
                    }
                    
                    HStack {
                        Image(systemName: "arrow.up.and.down")
                        Text("Depth: \(String(format: "%.1f", adjustedDepth))m")
                        Slider(value: $adjustedDepth, in: 2...8, step: 0.1)
                            .onChange(of: adjustedDepth) { oldValue, newValue in
                                print("📏 [View] Depth adjusted: \(oldValue) -> \(newValue)")
                            }
                    }
                    
                    HStack {
                        Image(systemName: "arrow.up.to.line")
                        Text("Height: \(String(format: "%.1f", adjustedHeight))m")
                        Slider(value: $adjustedHeight, in: 2.2...4, step: 0.1)
                            .onChange(of: adjustedHeight) { oldValue, newValue in
                                print("📏 [View] Height adjusted: \(oldValue) -> \(newValue)")
                            }
                    }
                    
                    Text("Confidence: \(Int(dimensions.confidence * 100))%")
                        .font(.caption)
                        .foregroundColor(confidenceColor(dimensions.confidence))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(confidenceColor(dimensions.confidence).opacity(0.1))
                        .cornerRadius(6)
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(12)
                .padding()
                .onAppear { print("📊 [View] Dimensions view appeared") }
                
                Button("Rebuild with Adjusted Dimensions") {
                    print("🔄 [View] Rebuild button tapped")
                    rebuildRoom()
                }
                .buttonStyle(.borderedProminent)
            }
            
            // ✅ CHANGED: Using scene instead of URL
            if let roomScene = reconstructor.generatedRoomScene {
                NavigationLink(destination: SceneKitViewer(scene: roomScene)) {
                    Label("View 3D Room", systemImage: "eye.fill")
                }
                .buttonStyle(.borderedProminent)
                .padding()
                .onAppear {
                    print("🎯 [View] View 3D Room button appeared")
                }
            }
            
            Spacer()
        }
    }
        .navigationTitle("Photo to 3D Room")
        .sheet(isPresented: $showImagePicker) {
            PhotoPickerView(selectedImage: $selectedImage)
                .onDisappear {
                    print("📱 [View] Image picker dismissed")
                    if let image = selectedImage {
                        print("✅ [View] Image selected, starting processing...")
                        Task {
                            await reconstructor.processPhoto(image)
                            if let dims = reconstructor.estimatedDimensions {
                                adjustedWidth = dims.width
                                adjustedDepth = dims.depth
                                adjustedHeight = dims.height
                                print("📏 [View] Sliders updated with estimated dimensions")
                            }
                        }
                    } else {
                        print("⚠️ [View] No image selected")
                    }
                }
        }
        // Room boundaries sheet: always returns a View via wrapper
        .sheet(isPresented: $showRoomBoundaries) {
            RoomBoundarySheetView(
                image: selectedImage,
                savedBoundaries: $adjustedBoundaries
            )
        }
        // NEW: Full screen cover for 3D Room Scan
        .fullScreenCover(isPresented: $show3DScanOption) {
            if #available(iOS 16.0, *) {
                NavigationView {
                    RoomCaptureView(
                        onSaveComplete: {
                            print("🎯 [SinglePhotoRoomViewer] onSaveComplete callback triggered!")
                            print("   - Current show3DScanOption: \(show3DScanOption)")
                            
                            // Set to false to dismiss the fullScreenCover
                            show3DScanOption = false
                            
                            print("   - Updated show3DScanOption: \(show3DScanOption)")
                            print("   - View should dismiss now")
                        }
                    )
                    .navigationBarItems(
                        leading: Button("Cancel") {
                            print("❌ [SinglePhotoRoomViewer] Cancel button pressed")
                            show3DScanOption = false
                        }
                    )
                }
                .onAppear {
                    print("🔷 [SinglePhotoRoomViewer] RoomCaptureView appeared")
                }
                .onDisappear {
                    print("🔶 [SinglePhotoRoomViewer] RoomCaptureView disappeared")
                }
            }
        }
        // NEW: Alert for unsupported devices
        .alert("Device Not Supported", isPresented: $showUnsupportedAlert) {
            Button("OK", role: .cancel) {}
        } message: {
            Text("3D Room Scanning requires an iPhone or iPad with LiDAR sensor (iPhone 12 Pro or later, iPad Pro 2020 or later).")
        }
        .onAppear {
            print("👁️ [View] SinglePhotoRoomView appeared")
            adjustedWidth = reconstructor.estimatedDimensions?.width ?? 4.0
            adjustedDepth = reconstructor.estimatedDimensions?.depth ?? 4.0
            adjustedHeight = reconstructor.estimatedDimensions?.height ?? 2.8
        }
        // ✅ NEW: Watch for boundary changes and rebuild automatically
        .onChange(of: adjustedBoundaries) { oldValue, newValue in
            if let boundaries = newValue, let image = selectedImage {
                print("🔄 [View] Boundaries adjusted, rebuilding room...")
                Task {
                    await reconstructor.processPhotoWithBoundaries(image, boundaries: boundaries)
                }
            }
        }
    }
    
    private func rebuildRoom() {
        print("🔄 [View] Rebuilding room with adjusted dimensions")
        var updatedDimensions = reconstructor.estimatedDimensions ?? SinglePhotoRoomReconstructor.RoomDimensions()
        updatedDimensions.width = adjustedWidth
        updatedDimensions.depth = adjustedDepth
        updatedDimensions.height = adjustedHeight
        
        print("   - New dimensions: W:\(adjustedWidth) D:\(adjustedDepth) H:\(adjustedHeight)")
        
        Task {
            await MainActor.run {
                reconstructor.estimatedDimensions = updatedDimensions
            }
            
            if let image = selectedImage {
                if let boundaries = adjustedBoundaries {
                    await reconstructor.processPhotoWithBoundaries(image, boundaries: boundaries)
                } else {
                    await reconstructor.processPhoto(image)
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

// Wrapper view to guarantee the sheet always returns a View
private struct RoomBoundarySheetView: View {
    let image: UIImage?
    @Binding var savedBoundaries: RoomStructure?
    
    var body: some View {
        Group {
            if let image {
                RoomBoundaryDetectionView(
                    originalImage: image,
                    savedBoundaries: $savedBoundaries
                )
            } else {
                EmptyView()
            }
        }
    }
}

// MARK: - Photo Picker View
struct PhotoPickerView: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.dismiss) var dismiss
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        print("📱 [PhotoPicker] Creating UIImagePickerController")
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
            print("📱 [PhotoPicker] Coordinator initialized")
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            print("📱 [PhotoPicker] Image picked from library")
            if let image = info[.originalImage] as? UIImage {
                print("✅ [PhotoPicker] Got UIImage: \(image.size)")
                parent.selectedImage = image
            } else {
                print("❌ [PhotoPicker] Failed to get UIImage")
            }
            parent.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            print("❌ [PhotoPicker] User cancelled")
            parent.dismiss()
        }
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
                print("🎬 [Viewer] SceneKit viewer appeared")
                print("   - Scene nodes: \(scene.rootNode.childNodes.count)")
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
                        Text("Use joystick to move • Pinch to zoom")
                            .font(.caption)
                    }
                    .padding(8)
                    .background(.ultraThinMaterial)
                    .cornerRadius(8)
                    .padding()
                    .padding(.bottom, 100) // Move above joystick
                }
                .onAppear {
                    print("ℹ️ [Viewer] Controls hint displayed")
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                        withAnimation { showControls = false }
                    }
                }
            }
        }
        .navigationTitle("3D Room View")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            // Help button
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    showControls.toggle()
                    print("ℹ️ [Viewer] Controls hint toggled: \(showControls)")
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
        .alert("Save Room", isPresented: $showRoomNameInput) {
            TextField("Room Name", text: $roomName)
            Button("Cancel", role: .cancel) {
                roomName = ""
            }
            Button("Save") {
                startSavingRoom()
            }
            .disabled(roomName.isEmpty)
        } message: {
            Text("Enter a name for your room")
        }
        // Save result alert
        .alert("Room Save", isPresented: $showSaveAlert) {
            Button("OK", role: .cancel) {
                if saveAlertMessage.contains("successfully") {
                    dismiss() // Go back to list after successful save
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
                    Text("Saving Room")
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
                    Text("Cancel")
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
            return "Preparing room model..."
        } else if saveProgress < 0.6 {
            return "Exporting to USDZ format..."
        } else if saveProgress < 0.9 {
            return "Saving to library..."
        } else {
            return "Almost done..."
        }
    }
    
    // MARK: - Save Room Functions
    private func startSavingRoom() {
        guard !roomName.isEmpty else {
            return
        }
        
        let savedName = roomName  // ✅ Capture the name BEFORE clearing
        print("💾 [Viewer] Starting room save process: \(savedName)")
        
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
                print("📦 [Viewer] Preparing model...")
            } else if self.saveProgress >= 0.6 && !saveStarted {
                print("📄 [Viewer] Exporting USDZ...")
                saveStarted = true
                
                // ✅ Actually save the room with completion handler
                self.modelManager.saveRoom(scene: scene, name: savedName) { success, error in
                    DispatchQueue.main.async {
                        saveCompleted = true
                        saveSuccess = success
                        saveError = error
                        print(success ? "✅ [Viewer] Room saved successfully" : "❌ [Viewer] Failed to save: \(error ?? "unknown")")
                    }
                }
            } else if self.saveProgress >= 0.9 && self.saveProgress < 0.92 {
                print("💾 [Viewer] Finalizing...")
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
                        self.saveAlertMessage = "Room '\(savedName)' saved successfully!"
                        self.showSaveAlert = true
                        self.roomName = ""
                        print("✅ [Viewer] Save complete!")
                    } else {
                        self.saveAlertMessage = "Failed to save room: \(saveError ?? "Unknown error")"
                        self.showSaveAlert = true
                        print("❌ [Viewer] Save failed!")
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
        print("❌ [Viewer] Room save cancelled")
    }

    // ✅ Setup camera position like vintage room (outside, looking at front wall)
    private func setupCamera() {
        print("📷 [SceneKitViewer] Setting up camera position...")

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

        print("   📦 Room bounds: min(\(minBounds)), max(\(maxBounds))")
        print("   📏 Room size: \(roomSize.x) x \(roomSize.y) x \(roomSize.z)")
        print("   🎯 Room center: \(roomCenter)")

        // ✅ Camera positioning: OUTSIDE the room (beyond max Z), looking at FRONT wall (min Z)
        // Same strategy as RealityKitBoundaryManager.getOptimalCameraPosition()
        let camX = roomCenter.x  // Center X
        let camY = roomCenter.y  // Center height
        let camZ = maxBounds.z + (roomSize.z * 0.3)  // OUTSIDE room, beyond back

        let lookAtX = roomCenter.x  // Center X
        let lookAtY = roomCenter.y  // Center height
        let lookAtZ = minBounds.z   // FRONT wall (where photo is)

        print("   📷 Camera position: (\(camX), \(camY), \(camZ))")
        print("   👁️ Looking at: (\(lookAtX), \(lookAtY), \(lookAtZ))")

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

        print("   ✅ Camera setup complete and registered with GlobalCameraController")
    }
}
