// SmartyPantsMaskUtils.swift
// Mask processing utilities for SmartyPants detection

import Foundation
import Accelerate
import UIKit
import Photos

// MARK: - Mask Utilities

/// Applies morphological closing (Dilate then Erode) on binary mask
/// - Parameters:
///   - mask: Float mask array (0.0 or 1.0 values)
///   - width: Mask width
///   - height: Mask height
///   - kernelSize: Size of the morphological kernel (default 9)
///   - iterations: Number of closing iterations (default 2)
func applyMorphologicalClosing(to mask: inout [Float], width: Int, height: Int, kernelSize: Int = 9, iterations: Int = 2) {
    let count = width * height
    guard count > 0, mask.count == count else { return }

    // Convert Float (0/1) -> Planar8 (0/255)
    var buf1 = [UInt8](repeating: 0, count: count)
    for i in 0..<count { buf1[i] = mask[i] > 0 ? 255 : 0 }

    var buf2 = [UInt8](repeating: 0, count: count)
    var tmpBuf = [UInt8](repeating: 0, count: count)

    // Create kernel (all ones)
    let kSize = kernelSize
    var kernel = [UInt8](repeating: 1, count: kSize * kSize)

    for iter in 0..<iterations {
        let isEven = (iter % 2 == 0)

        if isEven {
            buf1.withUnsafeMutableBytes { srcPtr in
                tmpBuf.withUnsafeMutableBytes { tmpPtr in
                    buf2.withUnsafeMutableBytes { dstPtr in
                        var srcVBuf = vImage_Buffer(data: srcPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                        var tmpVBuf = vImage_Buffer(data: tmpPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                        var dstVBuf = vImage_Buffer(data: dstPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)

                        kernel.withUnsafeMutableBufferPointer { kPtr in
                            vImageDilate_Planar8(&srcVBuf, &tmpVBuf, 0, 0, kPtr.baseAddress!, vImagePixelCount(kSize), vImagePixelCount(kSize), vImage_Flags(kvImageNoFlags))
                            vImageErode_Planar8(&tmpVBuf, &dstVBuf, 0, 0, kPtr.baseAddress!, vImagePixelCount(kSize), vImagePixelCount(kSize), vImage_Flags(kvImageNoFlags))
                        }
                    }
                }
            }
        } else {
            buf2.withUnsafeMutableBytes { srcPtr in
                tmpBuf.withUnsafeMutableBytes { tmpPtr in
                    buf1.withUnsafeMutableBytes { dstPtr in
                        var srcVBuf = vImage_Buffer(data: srcPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                        var tmpVBuf = vImage_Buffer(data: tmpPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                        var dstVBuf = vImage_Buffer(data: dstPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)

                        kernel.withUnsafeMutableBufferPointer { kPtr in
                            vImageDilate_Planar8(&srcVBuf, &tmpVBuf, 0, 0, kPtr.baseAddress!, vImagePixelCount(kSize), vImagePixelCount(kSize), vImage_Flags(kvImageNoFlags))
                            vImageErode_Planar8(&tmpVBuf, &dstVBuf, 0, 0, kPtr.baseAddress!, vImagePixelCount(kSize), vImagePixelCount(kSize), vImage_Flags(kvImageNoFlags))
                        }
                    }
                }
            }
        }
    }

    // Result is in buf2 if iterations is even, buf1 if odd
    let resultBuf = (iterations % 2 == 0) ? buf1 : buf2

    // Convert back Planar8 -> Float (0/1)
    for i in 0..<count { mask[i] = resultBuf[i] > 0 ? 1.0 : 0.0 }
}

/// Computes convex hull of mask pixels and fills the entire hull area
/// This guarantees no gaps within each detection's boundary
/// - Returns: Number of pixels added
func fillConvexHullOfMask(
    mask: [Float],
    into globalMask: inout [Float],
    width: Int,
    height: Int,
    bboxX1: Int, bboxY1: Int, bboxX2: Int, bboxY2: Int,
    threshold: Float
) -> Int {
    // Collect mask pixels above threshold within bbox
    var points = [(x: Int, y: Int)]()
    for py in bboxY1..<bboxY2 {
        let rowStart = py * width
        for px in bboxX1..<bboxX2 {
            let idx = rowStart + px
            if mask[idx] > threshold {
                points.append((px, py))
            }
        }
    }

    guard points.count >= 3 else {
        // Not enough points for hull, just copy mask directly
        var added = 0
        for py in bboxY1..<bboxY2 {
            let rowStart = py * width
            for px in bboxX1..<bboxX2 {
                let idx = rowStart + px
                if mask[idx] > threshold && globalMask[idx] == 0 {
                    globalMask[idx] = 1.0
                    added += 1
                }
            }
        }
        return added
    }

    // Compute convex hull using Graham scan
    let hull = computeConvexHull(points: points)
    guard hull.count >= 3 else {
        return 0
    }

    // Fill the convex hull polygon using scanline
    var added = 0
    let minY = hull.min(by: { $0.y < $1.y })!.y
    let maxY = hull.max(by: { $0.y < $1.y })!.y

    for y in minY...maxY {
        // Find all x-intersections with hull edges at this y
        var xIntersections = [Int]()

        for i in 0..<hull.count {
            let p1 = hull[i]
            let p2 = hull[(i + 1) % hull.count]

            // Check if edge crosses this scanline
            if (p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y) {
                // Calculate x intersection
                let t = Float(y - p1.y) / Float(p2.y - p1.y)
                let x = Int(Float(p1.x) + t * Float(p2.x - p1.x))
                xIntersections.append(x)
            }
        }

        xIntersections.sort()

        // Fill between pairs of intersections
        var i = 0
        while i + 1 < xIntersections.count {
            let x1 = max(bboxX1, xIntersections[i])
            let x2 = min(bboxX2 - 1, xIntersections[i + 1])
            if x2 >= x1 && y >= 0 && y < height {
                let rowStart = y * width
                for x in x1...x2 {
                    if x >= 0 && x < width {
                        let idx = rowStart + x
                        if globalMask[idx] == 0 {
                            globalMask[idx] = 1.0
                            added += 1
                        }
                    }
                }
            }
            i += 2
        }
    }

    return added
}

/// Graham scan convex hull algorithm
func computeConvexHull(points: [(x: Int, y: Int)]) -> [(x: Int, y: Int)] {
    guard points.count >= 3 else { return points }

    // Find lowest point (and leftmost if tie)
    var sorted = points
    let start = sorted.min { a, b in
        if a.y != b.y { return a.y < b.y }
        return a.x < b.x
    }!

    // Sort by polar angle with respect to start point
    sorted.sort { a, b in
        let ax = a.x - start.x, ay = a.y - start.y
        let bx = b.x - start.x, by = b.y - start.y
        let cross = ax * by - ay * bx
        if cross != 0 { return cross > 0 }
        // Collinear - sort by distance
        return ax * ax + ay * ay < bx * bx + by * by
    }

    // Build hull
    var hull = [(x: Int, y: Int)]()
    for p in sorted {
        while hull.count >= 2 {
            let o = hull[hull.count - 2]
            let a = hull[hull.count - 1]
            let cross = (a.x - o.x) * (p.y - o.y) - (a.y - o.y) * (p.x - o.x)
            if cross <= 0 {
                hull.removeLast()
            } else {
                break
            }
        }
        hull.append(p)
    }

    return hull
}

/// Compute edge pixels (circumference) of a binary mask using 4-neighborhood
func computeMaskEdges(mask: [Float], width: Int, height: Int) -> [(x: Int, y: Int)] {
    let w = width, h = height
    guard mask.count == w * h, w > 0, h > 0 else { return [] }
    var edges: [(Int, Int)] = []
    edges.reserveCapacity(w * h / 8)
    for y in 0..<h {
        let rowOff = y * w
        for x in 0..<w {
            let idx = rowOff + x
            if mask[idx] <= 0 { continue }
            // 4-neighborhood: if any neighbor is background or out of bounds, it's an edge
            let leftEmpty   = (x == 0)        || mask[idx - 1] <= 0
            let rightEmpty  = (x == w - 1)    || mask[idx + 1] <= 0
            let topEmpty    = (y == 0)        || mask[idx - w] <= 0
            let bottomEmpty = (y == h - 1)    || mask[idx + w] <= 0
            if leftEmpty || rightEmpty || topEmpty || bottomEmpty {
                edges.append((x, y))
            }
        }
    }
    return edges
}

/// Print 20x20 binary grid representation of mask (for debugging)
func print20x20BinaryGrid(_ title: String, mask: [UInt8], width: Int, height: Int) {
    print("\n🔢 [\(title)] (20x20 binary, * = object, . = background):")
    for gy in 0..<20 {
        var rowSymbols = ""
        for gx in 0..<20 {
            let y = gy * 8 + 7
            let x = gx * 8 + 7
            if y < height && x < width {
                let idx = y * width + x
                rowSymbols += mask[idx] > 0 ? "*" : "."
            } else {
                rowSymbols += " "
            }
        }
        print("   \(rowSymbols)")
    }
}

/// Save mask to Photos library (for debugging)
func saveMaskToFile(rawMask: [Float], width: Int, height: Int, detection: DetectionSmarty, maskThreshold: Float) {
    let timestamp = Int(Date().timeIntervalSince1970)
    let colorSpace = CGColorSpaceCreateDeviceGray()

    var minVal: Float = 0
    var maxVal: Float = 0
    vDSP_minv(rawMask, 1, &minVal, vDSP_Length(rawMask.count))
    vDSP_maxv(rawMask, 1, &maxVal, vDSP_Length(rawMask.count))
    let range = maxVal - minVal

    let count = rawMask.count
    var normalized = [Float](repeating: 0, count: count)

    if range > 0 {
        var negMin = -minVal
        rawMask.withUnsafeBufferPointer { src in
            normalized.withUnsafeMutableBufferPointer { dst in
                vDSP_vsadd(src.baseAddress!, 1, &negMin, dst.baseAddress!, 1, vDSP_Length(count))
            }
        }
        var invRange: Float = 1.0 / range
        vDSP_vsmul(normalized, 1, &invRange, &normalized, 1, vDSP_Length(count))
    } else {
        normalized = [Float](repeating: 0.5, count: count)
    }

    var scale255: Float = 255.0
    vDSP_vsmul(normalized, 1, &scale255, &normalized, 1, vDSP_Length(count))

    var clipLow: Float = 0
    var clipHigh: Float = 255
    vDSP_vclip(normalized, 1, &clipLow, &clipHigh, &normalized, 1, vDSP_Length(count))

    var grayPixels = [UInt8](repeating: 0, count: count)
    normalized.withUnsafeBufferPointer { src in
        grayPixels.withUnsafeMutableBufferPointer { dst in
            vDSP_vfixu8(src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(count))
        }
    }

    if let provider = CGDataProvider(data: Data(grayPixels) as CFData),
       let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
                              bytesPerRow: width, space: colorSpace,
                              bitmapInfo: CGBitmapInfo(rawValue: 0),
                              provider: provider, decode: nil, shouldInterpolate: false,
                              intent: .defaultIntent) {
        let grayImage = UIImage(cgImage: cgImage)
        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAsset(from: grayImage)
        }) { success, error in
            if success {
                print("💾 Saved GRAYSCALE mask to Photos @ \(timestamp)")
            } else {
                print("❌ Failed to save grayscale: \(error?.localizedDescription ?? "unknown")")
            }
        }
    }

    let scale = Float(width) / 960.0
    let mx1 = max(0, Int((detection.x - detection.width / 2) * scale))
    let my1 = max(0, Int((detection.y - detection.height / 2) * scale))
    let mx2 = min(width, Int((detection.x + detection.width / 2) * scale))
    let my2 = min(height, Int((detection.y + detection.height / 2) * scale))

    var binaryPixels = [UInt8](repeating: 0, count: width * height)
    for y in 0..<height {
        for x in 0..<width {
            let idx = y * width + x
            if x >= mx1 && x < mx2 && y >= my1 && y < my2 && rawMask[idx] > maskThreshold {
                binaryPixels[idx] = 255
            }
        }
    }

    if let provider = CGDataProvider(data: Data(binaryPixels) as CFData),
       let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
                              bytesPerRow: width, space: colorSpace,
                              bitmapInfo: CGBitmapInfo(rawValue: 0),
                              provider: provider, decode: nil, shouldInterpolate: false,
                              intent: .defaultIntent) {
        let binaryImage = UIImage(cgImage: cgImage)
        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAsset(from: binaryImage)
        }) { success, error in
            if success {
                print("💾 Saved BINARY mask to Photos (threshold: \(maskThreshold)) @ \(timestamp)")
            } else {
                print("❌ Failed to save binary: \(error?.localizedDescription ?? "unknown")")
            }
        }
    }
}
