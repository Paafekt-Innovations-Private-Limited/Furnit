// SmartyPantsRendering.swift
// Image rendering, label drawing, debug visualization

import UIKit
import CoreGraphics
import Photos
import Accelerate

extension SmartyPantsContainerView {
    
    // MARK: - Draw Labels and Boxes
    func drawLabelsAndBoxes(
        ctx: CGContext,
        stage1: [DetectionSmarty],
        stage2: [DetectionSmarty],
        imageWidth: Int,
        imageHeight: Int,
        drawBoxes: Bool
    ) {
        let all = stage1 + stage2
        guard !all.isEmpty else { return }
        let drawStart = Date()

        let W = CGFloat(imageWidth)
        let H = CGFloat(imageHeight)
        let modelSize: CGFloat = CGFloat(kModelInputSize)

        let sx = W / modelSize
        let sy = H / modelSize

        ctx.saveGState()
        ctx.translateBy(x: 0, y: H)
        ctx.scaleBy(x: 1, y: -1)

        UIGraphicsPushContext(ctx)

        let font = UIFont.boldSystemFont(ofSize: 38)

        for det in all {
            let cx = CGFloat(det.x)
            let cy = CGFloat(det.y)
            let w  = CGFloat(det.width)
            let h  = CGFloat(det.height)

            let left = (cx - w / 2) * sx
            let top  = (cy - h / 2) * sy
            let rect = CGRect(x: left, y: top, width: w * sx, height: h * sy)

            if drawBoxes {
                UIColor.cyan.setStroke()
                let b = UIBezierPath(rect: rect)
                b.lineWidth = 4
                b.stroke()
            }

            let textString = "\(det.className) \(Int(det.confidence * 100))%"
            let attributes: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: UIColor.cyan,
                .backgroundColor: UIColor.black.withAlphaComponent(0.6),
                .shadow: {
                    let sh = NSShadow()
                    sh.shadowBlurRadius = 6
                    sh.shadowOffset = CGSize(width: 2, height: -2)
                    sh.shadowColor = UIColor.black.withAlphaComponent(0.8)
                    return sh
                }()
            ]

            let text = NSAttributedString(string: textString, attributes: attributes)
            let size = text.size()

            var tx = max(0, min(left, W - size.width - 4))
            var ty = top - size.height - 6
            if ty < 0 { ty = top + 6 }

            let drawRect = CGRect(x: tx, y: ty, width: size.width, height: size.height)
            text.draw(in: drawRect)
        }

        UIGraphicsPopContext()
        ctx.restoreGState()
        if debugMode {
            let drawEnd = Date()
            print(String(format: "⏱ drawLabelsAndBoxes: %.2f ms (items: %d)", drawEnd.timeIntervalSince(drawStart) * 1000.0, all.count))
        }
    }

    // MARK: - Draw Perimeter Outline (Debug)
    func drawPerimeterOutline(
        on pixels: UnsafeMutablePointer<UInt8>,
        mask: [Float],
        maskWidth: Int,
        maskHeight: Int,
        imageWidth: Int,
        imageHeight: Int
    ) {
        let perStart = Date()
        let scaleX = Float(imageWidth) / Float(maskWidth)
        let scaleY = Float(imageHeight) / Float(maskHeight)
        
        for my in 0..<maskHeight {
            for mx in 0..<maskWidth {
                let idx = my * maskWidth + mx
                if mask[idx] > 0 {
                    var isEdge = false
                    if mx == 0 || mx == maskWidth - 1 || my == 0 || my == maskHeight - 1 {
                        isEdge = true
                    } else {
                        if mask[idx - 1] == 0 || mask[idx + 1] == 0 ||
                           mask[idx - maskWidth] == 0 || mask[idx + maskWidth] == 0 {
                            isEdge = true
                        }
                    }
                    
                    if isEdge {
                        let imgX = Int(Float(mx) * scaleX)
                        let imgY = Int(Float(my) * scaleY)
                        
                        for dy in -1...1 {
                            for dx in -1...1 {
                                let px = imgX + dx
                                let py = imgY + dy
                                if px >= 0 && px < imageWidth && py >= 0 && py < imageHeight {
                                    let pixelIdx = (py * imageWidth + px) * 4
                                    pixels[pixelIdx + 0] = 255   // B
                                    pixels[pixelIdx + 1] = 255   // G
                                    pixels[pixelIdx + 2] = 0     // R
                                    pixels[pixelIdx + 3] = 255   // A
                                }
                            }
                        }
                    }
                }
            }
        }
        if debugMode {
            let perEnd = Date()
            print(String(format: "⏱ drawPerimeterOutline: %.2f ms", perEnd.timeIntervalSince(perStart) * 1000.0))
        }
    }

    // MARK: - Save Mask to Photos (Debug)
    func saveMaskToPhotos(_ mask: [Float], width: Int, height: Int, label: String = "mask") {
        let saveStart = Date()
        let count = width * height
        
        // We assume mask is binary (0.0 or 1.0). Keep semantics:
        //  - 0.0  → 0
        //  - 1.0  → 255
        // This matches the original "mask[i] > 0 ? 255 : 0" for binary masks.
        
        // 1) Scale mask floats by 255 into an intermediate float buffer.
        var scaled = [Float](repeating: 0, count: count)
        var scale: Float = 255.0
        mask.withUnsafeBufferPointer { src in
            scaled.withUnsafeMutableBufferPointer { dst in
                if let s = src.baseAddress, let d = dst.baseAddress {
                    vDSP_vsmul(s, 1, &scale, d, 1, vDSP_Length(count))
                }
            }
        }
        
        // 2) Convert scaled floats → UInt8 (0 or 255 when input is 0/1).
        var gray = [UInt8](repeating: 0, count: count)
        scaled.withUnsafeBufferPointer { src in
            gray.withUnsafeMutableBufferPointer { dst in
                if let s = src.baseAddress, let d = dst.baseAddress {
                    vDSP_vfixu8(s, 1, d, 1, vDSP_Length(count))
                }
            }
        }
        
        // 3) Expand planar gray into RGBA buffer.
        var pixels = [UInt8](repeating: 0, count: count * 4)
        for i in 0..<count {
            let v = gray[i]
            let base = i * 4
            pixels[base + 0] = v  // R
            pixels[base + 1] = v  // G
            pixels[base + 2] = v  // B
            pixels[base + 3] = 255
        }
        
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else {
            print("❌ Failed to create data provider")
            return
        }
        
        guard let cgImage = CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ) else {
            print("❌ Failed to create CGImage")
            return
        }
        
        let uiImage = UIImage(cgImage: cgImage)
        UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil)
        print("📸 Saved \(label) (\(width)x\(height)) to Photos")
        if debugMode {
            let saveEnd = Date()
            print(String(format: "⏱ saveMaskToPhotos: %.2f ms", saveEnd.timeIntervalSince(saveStart) * 1000.0))
        }
    }

    // MARK: - Clear Outside BBox
    func clearOutsideUsingIntCorners(x0: Int, y0: Int, x1: Int, y1: Int, in image: CGImage) -> CGImage? {
        let t0 = Date()
        
        let width = image.width
        let height = image.height
        let imageRect = CGRect(x: 0, y: 0, width: width, height: height)

        let minX0 = min(x0, x1), maxX0 = max(x0, x1)
        let minY0 = min(y0, y1), maxY0 = max(y0, y1)

        var bbox = CGRect(x: CGFloat(minX0), y: CGFloat(minY0),
                          width: CGFloat(maxX0 - minX0), height: CGFloat(maxY0 - minY0))

        if bbox.isNull || bbox.width <= 0 || bbox.height <= 0 {
            return makeTransparentImage(width: width, height: height)
        }

        bbox = bbox.intersection(imageRect)
        if bbox.isNull || bbox.width <= 0 || bbox.height <= 0 {
            return makeTransparentImage(width: width, height: height)
        }

        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let dataSize = bytesPerRow * height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }

        let rawData = UnsafeMutablePointer<UInt8>.allocate(capacity: dataSize)
        rawData.initialize(repeating: 0, count: dataSize)
        defer { rawData.deinitialize(count: dataSize); rawData.deallocate() }

        guard let ctx = CGContext(data: rawData, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                  space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }

        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1.0, y: -1.0)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var kx0 = max(0, min(Int(floor(bbox.minX)), width))
        var kx1 = max(0, min(Int(ceil(bbox.maxX)), width))
        var ky0 = max(0, min(Int(floor(bbox.minY)), height))
        var ky1 = max(0, min(Int(ceil(bbox.maxY)), height))

        if kx0 > kx1 { swap(&kx0, &kx1) }
        if ky0 > ky1 { swap(&ky0, &ky1) }

        for y in 0..<height {
            let rowBase = rawData + y * bytesPerRow
            for x in 0..<width {
                let px = rowBase + x * bytesPerPixel
                let inside = (x >= kx0 && x < kx1 && y >= ky0 && y < ky1)
                if !inside {
                    px[0] = 0; px[1] = 0; px[2] = 0; px[3] = 0
                }
            }
        }

        let out = ctx.makeImage()
        if debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ clearOutsideUsingIntCorners: %.2f ms", dt))
        }
        return out
    }

    func makeTransparentImage(width: Int, height: Int) -> CGImage? {
        guard width > 0 && height > 0 else { return nil }
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let bufSize = bytesPerRow * height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        guard let data = calloc(1, bufSize) else { return nil }
        defer { free(data) }

        guard let ctx = CGContext(data: data, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                  space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }

        return ctx.makeImage()
    }

    func cutoutClearOutsideAccelerated(x0: Int, y0: Int, x1: Int, y1: Int, in image: CGImage) -> CGImage? {
        let t0 = Date()
        let width = image.width
        let height = image.height
        guard width > 0 && height > 0 else { return nil }

        var minX = max(0, min(min(x0, x1), width))
        var maxX = max(0, min(max(x0, x1), width))
        var minY = max(0, min(min(y0, y1), height))
        var maxY = max(0, min(max(y0, y1), height))

        if minX >= maxX || minY >= maxY {
            return makeTransparentImage(width: width, height: height)
        }

        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let bufSize = bytesPerRow * height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }

        guard let destData = malloc(bufSize) else { return nil }
        defer { free(destData) }

        guard let ctx = CGContext(data: destData, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                  space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }

        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1.0, y: -1.0)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let zeroRow = malloc(bytesPerRow) else { return nil }
        memset(zeroRow, 0, bytesPerRow)
        defer { free(zeroRow) }

        if minY > 0 {
            for r in 0..<minY {
                memcpy(destData.advanced(by: r * bytesPerRow), zeroRow, bytesPerRow)
            }
        }

        if maxY < height {
            for r in 0..<(height - maxY) {
                memcpy(destData.advanced(by: (maxY + r) * bytesPerRow), zeroRow, bytesPerRow)
            }
        }

        if minX > 0 || maxX < width {
            let leftBytes = minX * bytesPerPixel
            let rightBytes = (width - maxX) * bytesPerPixel
            for row in minY..<maxY {
                let rowBase = destData.advanced(by: row * bytesPerRow)
                if leftBytes > 0 { memset(rowBase, 0, leftBytes) }
                if rightBytes > 0 { memset(rowBase.advanced(by: maxX * bytesPerPixel), 0, rightBytes) }
            }
        }

        guard let outCtx = CGContext(data: destData, width: width, height: height,
                                     bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                     space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }

        let outImage = outCtx.makeImage()
        if debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ cutoutClearOutsideAccelerated: %.2f ms", dt))
        }
        return outImage
    }

    func cutoutClearOutsideAcceleratedUIImage(x0: Int, y0: Int, x1: Int, y1: Int, in image: UIImage) -> UIImage? {
        guard let cg = image.cgImage else { return nil }
        guard let outCG = cutoutClearOutsideAccelerated(x0: x0, y0: y0, x1: x1, y1: y1, in: cg) else { return nil }
        return UIImage(cgImage: outCG, scale: image.scale, orientation: image.imageOrientation)
    }
}
