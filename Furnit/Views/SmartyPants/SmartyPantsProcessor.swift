// SmartyPantsProcessor.swift
// Detection processing utilities for SmartyPants

import Foundation
import CoreML
import Accelerate
import CoreGraphics
import UIKit

// MARK: - NMS (Non-Maximum Suppression)

/// Apply Non-Maximum Suppression to filter overlapping detections
/// - Parameters:
///   - detections: Array of detections to filter
///   - iouThreshold: IoU threshold above which detections are suppressed
///   - debugMode: Whether to print debug info
/// - Returns: Filtered array of detections
func applyNMS(_ detections: [DetectionSmarty], iouThreshold: Float, debugMode: Bool = false) -> [DetectionSmarty] {
    // Guard against empty or invalid input
    guard !detections.isEmpty else { return [] }
    guard iouThreshold >= 0 && iouThreshold <= 1 else {
        if debugMode { print("⚠️ applyNMS: Invalid IoU threshold: \(iouThreshold)") }
        return detections
    }

    // Filter out detections with invalid dimensions before sorting
    let validDetections = detections.filter { det in
        guard det.width > 0, det.height > 0,
              det.width.isFinite, det.height.isFinite,
              det.x.isFinite, det.y.isFinite,
              det.confidence >= 0 && det.confidence <= 1 else {
            if debugMode {
                print("⚠️ applyNMS: Filtering invalid detection: w=\(det.width), h=\(det.height), x=\(det.x), y=\(det.y), conf=\(det.confidence)")
            }
            return false
        }
        return true
    }

    guard !validDetections.isEmpty else { return [] }

    let sorted = validDetections.sorted { $0.confidence > $1.confidence }
    var kept: [DetectionSmarty] = []
    kept.reserveCapacity(sorted.count)

    for det in sorted {
        var dominated = false
        for k in kept {
            let iou = bboxIoU(det, k)
            if iou.isFinite && iou > iouThreshold {
                dominated = true
                break
            }
        }
        if !dominated { kept.append(det) }
    }
    return kept
}

// MARK: - Bounding Box IoU

/// Calculate Intersection over Union between two detections
/// - Parameters:
///   - a: First detection
///   - b: Second detection
/// - Returns: IoU value (0.0 to 1.0)
func bboxIoU(_ a: DetectionSmarty, _ b: DetectionSmarty) -> Float {
    // Guard against invalid inputs
    guard a.width > 0 && a.height > 0 && b.width > 0 && b.height > 0 else { return 0 }
    guard a.width.isFinite && a.height.isFinite && b.width.isFinite && b.height.isFinite else { return 0 }
    guard a.x.isFinite && a.y.isFinite && b.x.isFinite && b.y.isFinite else { return 0 }

    let aLeft = a.x - a.width * 0.5
    let aRight = a.x + a.width * 0.5
    let aTop = a.y - a.height * 0.5
    let aBottom = a.y + a.height * 0.5

    let bLeft = b.x - b.width * 0.5
    let bRight = b.x + b.width * 0.5
    let bTop = b.y - b.height * 0.5
    let bBottom = b.y + b.height * 0.5

    let ix1 = max(aLeft, bLeft)
    let ix2 = min(aRight, bRight)
    let iy1 = max(aTop, bTop)
    let iy2 = min(aBottom, bBottom)

    let iw = max(0, ix2 - ix1)
    let ih = max(0, iy2 - iy1)
    let inter = iw * ih

    let areaA = a.width * a.height
    let areaB = b.width * b.height
    let union = areaA + areaB - inter

    // Prevent division by zero and ensure result is valid
    guard union > 0 && union.isFinite && inter.isFinite else { return 0 }

    let iou = inter / union
    return iou.isFinite ? iou : 0
}

// MARK: - Prototype Buffer

/// Convert MLMultiArray prototypes to Float buffer using Accelerate
/// - Parameters:
///   - array: Source MLMultiArray
///   - C: Number of channels
///   - Hp: Prototype height
///   - Wp: Prototype width
///   - debugMode: Whether to print debug info
/// - Returns: Float array of prototype values
func makePrototypeBuffer(from array: MLMultiArray, C: Int, Hp: Int, Wp: Int, debugMode: Bool = false) -> [Float] {
    let count = C * Hp * Wp
    var out = [Float](repeating: 0, count: count)

    // Validate array size matches expected count
    guard array.count >= count else {
        if debugMode {
            print("⚠️ makePrototypeBuffer: Array size mismatch! Expected: \(count), Got: \(array.count)")
        }
        return out
    }

    switch array.dataType {
    case .float32:
        // Safer memory copying with bounds checking
        array.dataPointer.withMemoryRebound(to: Float.self, capacity: array.count) { src in
            out.withUnsafeMutableBufferPointer { dst in
                guard let dstPtr = dst.baseAddress else {
                    if debugMode { print("⚠️ makePrototypeBuffer: Null destination pointer") }
                    return
                }
                let safeCopyCount = min(count, array.count)
                memcpy(dstPtr, src, safeCopyCount * MemoryLayout<Float>.size)
            }
        }
    case .float16:
        // Safer Float16 conversion with bounds checking
        let actualCount = min(count, array.count)
        let src = array.dataPointer.bindMemory(to: UInt16.self, capacity: actualCount)
        var srcBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: src),
            height: 1,
            width: vImagePixelCount(actualCount),
            rowBytes: actualCount * MemoryLayout<UInt16>.size
        )
        out.withUnsafeMutableBufferPointer { dst in
            var dstBuf = vImage_Buffer(
                data: dst.baseAddress,
                height: 1,
                width: vImagePixelCount(actualCount),
                rowBytes: actualCount * MemoryLayout<Float>.size
            )
            let result = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
            if result != kvImageNoError && debugMode {
                print("⚠️ makePrototypeBuffer: vImage conversion failed with error: \(result)")
            }
        }
    default:
        // Safe fallback with bounds checking
        let safeCopyCount = min(count, array.count)
        for i in 0..<safeCopyCount {
            out[i] = array[i].floatValue
        }
    }

    return out
}

// MARK: - Binary Mask Conversion

/// Convert float global mask to binary UInt8 mask
/// - Parameters:
///   - globalMask: Float mask (0.0 or 1.0 values)
///   - count: Number of pixels
/// - Returns: Binary UInt8 mask (0 or 255 values)
func makeBinaryMaskFromGlobalMask(_ globalMask: [Float], count: Int) -> [UInt8] {
    var scaled = [Float](repeating: 0, count: count)
    var scale255: Float = 255.0

    globalMask.withUnsafeBufferPointer { src in
        scaled.withUnsafeMutableBufferPointer { dst in
            vDSP_vsmul(src.baseAddress!, 1, &scale255, dst.baseAddress!, 1, vDSP_Length(count))
        }
    }

    var binary = [UInt8](repeating: 0, count: count)
    scaled.withUnsafeBufferPointer { src in
        binary.withUnsafeMutableBufferPointer { dst in
            vDSP_vfixu8(src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(count))
        }
    }

    return binary
}

// MARK: - Clear Outside Region

/// Clear pixels outside a rectangular region (non-accelerated version)
/// - Parameters:
///   - x0: First corner X
///   - y0: First corner Y
///   - x1: Second corner X
///   - y1: Second corner Y
///   - image: Source image
///   - debugMode: Whether to print timing info
/// - Returns: Image with region outside rectangle cleared to transparent
func clearOutsideUsingIntCorners(x0: Int, y0: Int, x1: Int, y1: Int, in image: CGImage, debugMode: Bool = false) -> CGImage? {
    let t0 = Date()

    let width = image.width
    let height = image.height
    let imageRect = CGRect(x: 0, y: 0, width: width, height: height)

    let minX0 = min(x0, x1)
    let maxX0 = max(x0, x1)
    let minY0 = min(y0, y1)
    let maxY0 = max(y0, y1)

    var bbox = CGRect(x: CGFloat(minX0),
                      y: CGFloat(minY0),
                      width: CGFloat(maxX0 - minX0),
                      height: CGFloat(maxY0 - minY0))

    if bbox.isNull || bbox.width <= 0 || bbox.height <= 0 {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bufSize = bytesPerRow * height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        guard let data = calloc(1, bufSize) else { return nil }
        defer { free(data) }

        guard let ctx = CGContext(data: data,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            return nil
        }
        if debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ clearOutsideUsingIntCorners (empty): %.2f ms", dt))
        }
        return ctx.makeImage()
    }

    bbox = bbox.intersection(imageRect)
    if bbox.isNull || bbox.width <= 0 || bbox.height <= 0 {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bufSize = bytesPerRow * height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        guard let data = calloc(1, bufSize) else { return nil }
        defer { free(data) }

        guard let ctx = CGContext(data: data,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            return nil
        }
        if debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ clearOutsideUsingIntCorners (no intersection): %.2f ms", dt))
        }
        return ctx.makeImage()
    }

    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * width
    let bufSize = bytesPerRow * height
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
    guard let data = calloc(1, bufSize) else { return nil }
    defer { free(data) }

    guard let ctx = CGContext(data: data,
                              width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bytesPerRow: bytesPerRow,
                              space: colorSpace,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else {
        return nil
    }

    ctx.translateBy(x: 0, y: CGFloat(height))
    ctx.scaleBy(x: 1.0, y: -1.0)

    ctx.saveGState()
    ctx.clip(to: bbox)
    ctx.draw(image, in: imageRect)
    ctx.restoreGState()

    let out = ctx.makeImage()
    if debugMode {
        let dt = Date().timeIntervalSince(t0) * 1000.0
        print(String(format: "⏱ clearOutsideUsingIntCorners: %.2f ms", dt))
    }
    return out
}

// MARK: - Clear Outside (Accelerated)

/// Clear pixels outside a rectangular region using vImage acceleration
/// - Parameters:
///   - x0: First corner X
///   - y0: First corner Y
///   - x1: Second corner X
///   - y1: Second corner Y
///   - image: Source image
///   - debugMode: Whether to print timing info
/// - Returns: Image with region outside rectangle cleared to transparent
func cutoutClearOutsideAccelerated(x0: Int, y0: Int, x1: Int, y1: Int, in image: CGImage, debugMode: Bool = false) -> CGImage? {
    let t0 = Date()

    let width = image.width
    let height = image.height
    guard width > 0 && height > 0 else { return nil }

    var minX = min(x0, x1)
    var maxX = max(x0, x1)
    var minY = min(y0, y1)
    var maxY = max(y0, y1)

    minX = max(0, min(minX, width))
    maxX = max(0, min(maxX, width))
    minY = max(0, min(minY, height))
    maxY = max(0, min(maxY, height))

    if minX >= maxX || minY >= maxY {
        let out = makeTransparentImage(width: width, height: height)
        if debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ cutoutClearOutsideAccelerated (empty): %.2f ms", dt))
        }
        return out
    }

    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel
    let bufSize = bytesPerRow * height
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }

    guard let destData = malloc(bufSize) else { return nil }
    defer { free(destData) }

    guard let ctx = CGContext(data: destData,
                              width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bytesPerRow: bytesPerRow,
                              space: colorSpace,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else {
        return nil
    }

    ctx.translateBy(x: 0, y: CGFloat(height))
    ctx.scaleBy(x: 1.0, y: -1.0)
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    guard let zeroRow = malloc(bytesPerRow) else { return nil }
    memset(zeroRow, 0, bytesPerRow)
    defer { free(zeroRow) }

    if minY > 0 {
        let dstPtr = destData
        for r in 0..<minY {
            let rowBase = dstPtr.advanced(by: r * bytesPerRow)
            memcpy(rowBase, zeroRow, bytesPerRow)
        }
    }

    if maxY < height {
        let dstPtr = destData.advanced(by: maxY * bytesPerRow)
        for r in 0..<(height - maxY) {
            let rowBase = dstPtr.advanced(by: r * bytesPerRow)
            memcpy(rowBase, zeroRow, bytesPerRow)
        }
    }

    if minX > 0 || maxX < width {
        let leftBytes = minX * bytesPerPixel
        let rightBytes = (width - maxX) * bytesPerPixel
        for row in minY..<maxY {
            let rowBase = destData.advanced(by: row * bytesPerRow)
            if leftBytes > 0 {
                memset(rowBase, 0, leftBytes)
            }
            if rightBytes > 0 {
                let rightPtr = rowBase.advanced(by: maxX * bytesPerPixel)
                memset(rightPtr, 0, rightBytes)
            }
        }
    }

    guard let outCtx = CGContext(data: destData,
                                 width: width,
                                 height: height,
                                 bitsPerComponent: 8,
                                 bytesPerRow: bytesPerRow,
                                 space: colorSpace,
                                 bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else {
        return nil
    }

    let outImage = outCtx.makeImage()
    if debugMode {
        let dt = Date().timeIntervalSince(t0) * 1000.0
        print(String(format: "⏱ cutoutClearOutsideAccelerated: %.2f ms", dt))
    }
    return outImage
}

// MARK: - Clear Outside (UIImage wrapper)

/// Clear pixels outside a rectangular region (UIImage convenience wrapper)
/// - Parameters:
///   - x0: First corner X
///   - y0: First corner Y
///   - x1: Second corner X
///   - y1: Second corner Y
///   - image: Source UIImage
///   - debugMode: Whether to print timing info
/// - Returns: UIImage with region outside rectangle cleared to transparent
func cutoutClearOutsideAcceleratedUIImage(x0: Int, y0: Int, x1: Int, y1: Int, in image: UIImage, debugMode: Bool = false) -> UIImage? {
    guard let cg = image.cgImage else { return nil }
    guard let outCG = cutoutClearOutsideAccelerated(x0: x0, y0: y0, x1: x1, y1: y1, in: cg, debugMode: debugMode) else { return nil }
    return UIImage(cgImage: outCG, scale: image.scale, orientation: image.imageOrientation)
}
