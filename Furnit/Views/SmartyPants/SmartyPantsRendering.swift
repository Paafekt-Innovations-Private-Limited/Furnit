// SmartyPantsRendering.swift
// Mask building and rendering for SmartyPants detection

import Foundation
import CoreML
import CoreVideo
import Accelerate
import CoreImage
import UIKit

// MARK: - Mask Build Result

struct MaskBuildResult {
    let globalMask: [Float]
    let mappedStage2Detections: [DetectionSmarty]
    let maskWidth: Int
    let maskHeight: Int
    let stage1PixelCount: Int
}

// MARK: - Build Union Mask

/// Build union mask from stage 1 and stage 2 detections
/// - Parameters:
///   - stage1Detections: Detections from full frame inference
///   - stage1Prototypes: Prototype array from stage 1
///   - stage2Detections: Detections from cropped inference
///   - stage2Prototypes: Prototype array from stage 2 (optional)
///   - primaryBBox: Primary detection used for cropping
///   - edgeFillMode: Edge fill mode for mask generation
///   - enableMaskClosing: Whether to apply morphological closing
/// - Returns: MaskBuildResult with global mask and mapped detections
func buildUnionMask(
    stage1Detections: [DetectionSmarty],
    stage1Prototypes: MLMultiArray,
    stage2Detections: [DetectionSmarty],
    stage2Prototypes: MLMultiArray?,
    primaryBBox: DetectionSmarty,
    edgeFillMode: EdgeFillMode,
    enableMaskClosing: Bool
) -> MaskBuildResult {
    let funcStart = Date()

    let shape = stage1Prototypes.shape.map { $0.intValue }
    let C = shape[1]
    let Hp = shape[2]
    let Wp = shape[3]
    let spatial = Hp * Wp

    // Hardcoded threshold - all positive values are mask
    let maskThreshold: Float = 0.0

    print("\n🎨 Generating TWO-STAGE UNION cutout")
    print("   Stage 1: \(stage1Detections.count) detections")
    print("   Stage 2: \(stage2Detections.count) detections (Stage2 coords)")
    print("📐 Prototype shape: C=\(C), H=\(Hp), W=\(Wp)")

    var mappedStage2Detections: [DetectionSmarty] = []

    // Stage 1 prototype buffer
    let protoStage1Start = Date()
    let protoMatrix1 = makePrototypeBuffer(from: stage1Prototypes, C: C, Hp: Hp, Wp: Wp, debugMode: true)
    let protoStage1End = Date()
    print(String(format: "⏱ Stage1 prototype buffer build (Accelerate): %.2f ms",
                 protoStage1End.timeIntervalSince(protoStage1Start) * 1000.0))

    var globalMask = [Float](repeating: 0, count: spatial)

    print("\n🔵 Processing Stage 1 masks (full frame)...")

    var primaryRawMask: [Float]? = nil
    var primaryDet: DetectionSmarty? = nil
    var stage1PixelCount = 0

    let s1MaskStart = Date()
    for (detIndex, det) in stage1Detections.enumerated() {
        var rawMask = [Float](repeating: 0, count: spatial)
        let mmulStart = Date()

        // Validate mask coefficients before matrix multiplication
        guard det.maskCoeffs.count == C else {
            print("⚠️ Stage1 det[\(detIndex)]: Invalid mask coeffs count: \(det.maskCoeffs.count), expected: \(C)")
            continue
        }

        // Validate all coefficients are finite
        let hasInvalidCoeffs = det.maskCoeffs.contains { !$0.isFinite }
        guard !hasInvalidCoeffs else {
            print("⚠️ Stage1 det[\(detIndex)]: Non-finite mask coefficients detected")
            continue
        }

        // Validate prototype matrix dimensions
        guard protoMatrix1.count == C * spatial else {
            print("⚠️ Stage1 det[\(detIndex)]: Prototype matrix size mismatch: \(protoMatrix1.count), expected: \(C * spatial)")
            continue
        }

        // Safe matrix multiplication
        vDSP_mmul(det.maskCoeffs, 1,
                  protoMatrix1, 1,
                  &rawMask, 1,
                  1, vDSP_Length(spatial), vDSP_Length(C))
        let mmulEnd = Date()
        print(String(format: "   ⏱ vDSP_mmul Stage1 det[%d]: %.2f ms", detIndex,
                     mmulEnd.timeIntervalSince(mmulStart) * 1000.0))

        if detIndex == 0 {
            primaryRawMask = rawMask
            primaryDet = det

            var minVal: Float = 0, maxVal: Float = 0
            vDSP_minv(rawMask, 1, &minVal, vDSP_Length(spatial))
            vDSP_maxv(rawMask, 1, &maxVal, vDSP_Length(spatial))
            var mean: Float = 0
            vDSP_meanv(rawMask, 1, &mean, vDSP_Length(spatial))

            print("\n📊 PRIMARY MASK RAW VALUES (\(det.className) @ \(Int(det.confidence*100))%):")
            print("   Range: min=\(minVal), max=\(maxVal), mean=\(mean)")

            var posCount = 0, negCount = 0, zeroCount = 0
            for v in rawMask {
                if v > 0 { posCount += 1 }
                else if v < 0 { negCount += 1 }
                else { zeroCount += 1 }
            }
            print("   Distribution: \(posCount) positive, \(negCount) negative, \(zeroCount) zero")

            print("   Mask coefficients (32): [\(det.maskCoeffs.map { String(format: "%.6f", $0) }.joined(separator: ", "))]")

            let scale = Float(Wp) / 960.0
            let mx1 = max(0, Int((det.x - det.width / 2) * scale))
            let my1 = max(0, Int((det.y - det.height / 2) * scale))
            let mx2 = min(Wp, Int((det.x + det.width / 2) * scale))
            let my2 = min(Hp, Int((det.y + det.height / 2) * scale))

            print("   BBox in mask coords: (\(mx1),\(my1)) → (\(mx2),\(my2))")
        }

        let scale = Float(Wp) / 960.0
        let mx1 = max(0, Int((det.x - det.width / 2) * scale))
        let my1 = max(0, Int((det.y - det.height / 2) * scale))
        let mx2 = min(Wp, Int((det.x + det.width / 2) * scale))
        let my2 = min(Hp, Int((det.y + det.height / 2) * scale))

        var addedPixels = 0

//        switch edgeFillMode {
//        case .clothBased:
//            // Solid fill using hull - for beds, sofas, fabric items with gaps
//            addedPixels = fillConvexHullOfMask(
//                mask: rawMask,
//                into: &globalMask,
//                width: Wp,
//                height: Hp,
//                bboxX1: mx1, bboxY1: my1, bboxX2: mx2, bboxY2: my2,
//                threshold: maskThreshold
//            )
//            if detIndex < 5 {
//                print("   ✅ S1 \(det.className) @ \(Int(det.confidence*100))%: +\(addedPixels)px (hull)")
//            }
//
//        case .furniMaterial:
//            // Preserve fine edges - for chairs, tables, solid items
//            rawMask.withUnsafeBufferPointer { rPtr in
//                globalMask.withUnsafeMutableBufferPointer { gPtr in
//                    if mx2 > mx1 && my2 > my1 {
//                        for py in my1..<my2 {
//                            let rowStart = py * Wp + mx1
//                            let rowLen = mx2 - mx1
//                            for i in 0..<rowLen {
//                                let idx = rowStart + i
//                                if rPtr[idx] > maskThreshold && gPtr[idx] == 0 {
//                                    gPtr[idx] = 1.0
//                                    addedPixels += 1
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//            if detIndex < 5 {
//                print("   ✅ S1 \(det.className) @ \(Int(det.confidence*100))%: +\(addedPixels)px (fine)")
//            }
//
//        case .chairType:
//            // Morphological close on per-detection mask - fills small gaps, preserves shape
//            var closedMask = rawMask
//            applyMorphologicalClosing(to: &closedMask, width: Wp, height: Hp, kernelSize: 9, iterations: 1)
//            closedMask.withUnsafeBufferPointer { rPtr in
//                globalMask.withUnsafeMutableBufferPointer { gPtr in
//                    if mx2 > mx1 && my2 > my1 {
//                        for py in my1..<my2 {
//                            let rowStart = py * Wp + mx1
//                            let rowLen = mx2 - mx1
//                            for i in 0..<rowLen {
//                                let idx = rowStart + i
//                                if rPtr[idx] > maskThreshold && gPtr[idx] == 0 {
//                                    gPtr[idx] = 1.0
//                                    addedPixels += 1
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//            if detIndex < 5 {
//                print("   ✅ S1 \(det.className) @ \(Int(det.confidence*100))%: +\(addedPixels)px (morph)")
//            }
//        }
    }
    let s1MaskEnd = Date()

    for i in 0..<spatial { if globalMask[i] > 0 { stage1PixelCount += 1 } }
    print("   ⚙️ Mask threshold: \(maskThreshold)")
    print("   📊 After Stage 1: \(stage1PixelCount)/\(spatial) pixels (\(String(format: "%.1f", Float(stage1PixelCount)/Float(spatial)*100))%)")
    print(String(format: "⏱ Stage1 mask build+apply: %.2f ms", s1MaskEnd.timeIntervalSince(s1MaskStart) * 1000.0))

    if let rawMask = primaryRawMask, let det = primaryDet {
        saveMaskToFile(rawMask: rawMask, width: Wp, height: Hp, detection: det, maskThreshold: maskThreshold)
    }

    // Stage 2
    if let proto2 = stage2Prototypes, !stage2Detections.isEmpty {
        let s2ProtoStart = Date()
        print("\n🟢 Processing Stage 2 masks (cropped → full frame)...")

        let protoMatrix2 = makePrototypeBuffer(from: proto2, C: C, Hp: Hp, Wp: Wp, debugMode: true)
        let s2ProtoEnd = Date()
        print(String(format: "⏱ Stage2 prototype buffer build (Accelerate): %.2f ms",
                     s2ProtoEnd.timeIntervalSince(s2ProtoStart) * 1000.0))

        // Reconstruct crop region in Stage1 model coords (0..960)
        let padding: Float = 0.1
        let cropX1 = max(0, primaryBBox.x - primaryBBox.width / 2 * (1 + padding))
        let cropY1 = max(0, primaryBBox.y - primaryBBox.height / 2 * (1 + padding))
        let cropX2 = min(960, primaryBBox.x + primaryBBox.width / 2 * (1 + padding))
        let cropY2 = min(960, primaryBBox.y + primaryBBox.height / 2 * (1 + padding))
        let cropW = cropX2 - cropX1
        let cropH = cropY2 - cropY1

        print("   Crop region (model): (\(Int(cropX1)),\(Int(cropY1)))→(\(Int(cropX2)),\(Int(cropY2))) = \(Int(cropW))x\(Int(cropH))")

        let scale = Float(Wp) / 960.0
        let s2MaskStart = Date()

        let s2ToS1ScaleX = cropW / 960.0
        let s2ToS1ScaleY = cropH / 960.0

        mappedStage2Detections.removeAll(keepingCapacity: true)
        mappedStage2Detections.reserveCapacity(stage2Detections.count)

        for det in stage2Detections {
            var rawMask = [Float](repeating: 0, count: spatial)
            let mmulStart = Date()

            // Validate mask coefficients before matrix multiplication
            guard det.maskCoeffs.count == C else {
                print("⚠️ Stage2 det: Invalid mask coeffs count: \(det.maskCoeffs.count), expected: \(C)")
                continue
            }

            // Validate all coefficients are finite
            let hasInvalidCoeffs = det.maskCoeffs.contains { !$0.isFinite }
            guard !hasInvalidCoeffs else {
                print("⚠️ Stage2 det: Non-finite mask coefficients detected")
                continue
            }

            // Validate prototype matrix dimensions
            guard protoMatrix2.count == C * spatial else {
                print("⚠️ Stage2 det: Prototype matrix size mismatch: \(protoMatrix2.count), expected: \(C * spatial)")
                continue
            }

            // Safe matrix multiplication
            vDSP_mmul(det.maskCoeffs, 1,
                      protoMatrix2, 1,
                      &rawMask, 1,
                      1, vDSP_Length(spatial), vDSP_Length(C))
            let mmulEnd = Date()
            print(String(format: "   ⏱ vDSP_mmul Stage2: %.2f ms",
                         mmulEnd.timeIntervalSince(mmulStart) * 1000.0))

            let mx1_crop = max(0, Int((det.x - det.width / 2) * scale))
            let my1_crop = max(0, Int((det.y - det.height / 2) * scale))
            let mx2_crop = min(Wp, Int((det.x + det.width / 2) * scale))
            let my2_crop = min(Hp, Int((det.y + det.height / 2) * scale))

            var addedPixels = 0

            rawMask.withUnsafeBufferPointer { rPtr in
                globalMask.withUnsafeMutableBufferPointer { gPtr in
                    if mx2_crop > mx1_crop && my2_crop > my1_crop {
                        for py_crop in my1_crop..<my2_crop {
                            let base = py_crop * Wp
                            for px_crop in mx1_crop..<mx2_crop {
                                let cropIdx = base + px_crop
                                if rPtr[cropIdx] > maskThreshold {
                                    let fracX = Float(px_crop) / Float(Wp)
                                    let fracY = Float(py_crop) / Float(Hp)
                                    let fullX = cropX1 + fracX * cropW
                                    let fullY = cropY1 + fracY * cropH
                                    let mx_full = Int(fullX * scale)
                                    let my_full = Int(fullY * scale)
                                    if mx_full >= 0 && mx_full < Wp && my_full >= 0 && my_full < Hp {
                                        let fullIdx = my_full * Wp + mx_full
                                        if gPtr[fullIdx] == 0 {
                                            addedPixels += 1
                                        }
                                        gPtr[fullIdx] = 1.0
                                    }
                                }
                            }
                        }
                    }
                }
            }

            print("   ✅ S2 \(det.className) @ \(Int(det.confidence*100))%: bbox(\(mx1_crop),\(my1_crop))→(\(mx2_crop),\(my2_crop)), +\(addedPixels)px NEW")

            // Map Stage2 bbox → Stage1 bbox for drawing/summary (not for mask)
            let newX = cropX1 + det.x * s2ToS1ScaleX
            let newY = cropY1 + det.y * s2ToS1ScaleY
            let newW = det.width * s2ToS1ScaleX
            let newH = det.height * s2ToS1ScaleY

            var mapped = DetectionSmarty(
                x: newX,
                y: newY,
                width: newW,
                height: newH,
                confidence: det.confidence,
                classIdx: det.classIdx,
                className: det.className,
                maskCoeffs: det.maskCoeffs
            )
            mapped.trackId = det.trackId  // Preserve track ID
            mappedStage2Detections.append(mapped)
        }
        let s2MaskEnd = Date()
        print(String(format: "⏱ Stage2 mask build+apply: %.2f ms",
                     s2MaskEnd.timeIntervalSince(s2MaskStart) * 1000.0))
    }

    // Morphological closing to strengthen edges and close gaps (15x15 kernel, 2 iterations)
    if enableMaskClosing {
        let closeStart = Date()
        applyMorphologicalClosing(to: &globalMask, width: Wp, height: Hp, kernelSize: 15, iterations: 2)
        let dt = Date().timeIntervalSince(closeStart) * 1000.0
        print(String(format: "⏱ Morphological closing (15x15 x2): %.2f ms", dt))
    }

    var finalPixelCount = 0
    for i in 0..<spatial { if globalMask[i] > 0 { finalPixelCount += 1 } }
    let addedByStage2 = finalPixelCount - stage1PixelCount

    print("\n📊 MERGED MASK: \(finalPixelCount)/\(spatial) pixels (\(String(format: "%.1f", Float(finalPixelCount)/Float(spatial)*100))%)")
    print("   Stage 1 contributed: \(stage1PixelCount) pixels")
    print("   Stage 2 added: \(addedByStage2) NEW pixels")

    let funcEnd = Date()
    print(String(format: "⏱ buildUnionMask total: %.2f ms", funcEnd.timeIntervalSince(funcStart) * 1000.0))

    return MaskBuildResult(
        globalMask: globalMask,
        mappedStage2Detections: mappedStage2Detections,
        maskWidth: Wp,
        maskHeight: Hp,
        stage1PixelCount: stage1PixelCount
    )
}

// MARK: - Apply Mask to Image

/// Apply mask to original image by setting alpha channel
/// - Parameters:
///   - maskResult: Result from buildUnionMask
///   - originalImage: Original pixel buffer from camera
///   - stage1Detections: Stage 1 detections for bounding box regions
///   - ciContext: CIContext for image conversion
/// - Returns: CGContext with mask applied, or nil on failure
func applyMaskToImage(
    maskResult: MaskBuildResult,
    originalImage: CVPixelBuffer,
    stage1Detections: [DetectionSmarty],
    ciContext: CIContext
) -> (context: CGContext, width: Int, height: Int)? {
    let renderStart = Date()
    let ciImage = CIImage(cvPixelBuffer: originalImage)
    let width = CVPixelBufferGetWidth(originalImage)
    let height = CVPixelBufferGetHeight(originalImage)

    guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
        print("❌ Failed to create CGImage")
        return nil
    }

    guard let ctx = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        print("❌ Failed to create CGContext")
        return nil
    }

    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    guard let data = ctx.data else {
        print("❌ CGContext has no data")
        return nil
    }

    let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
    let Wp = maskResult.maskWidth
    let Hp = maskResult.maskHeight
    let globalMask = maskResult.globalMask

    let scaleX = Float(Wp) / Float(width)
    let scaleY = Float(Hp) / Float(height)

    print("🖼️ Upscaling \(Wp)×\(Hp) → \(width)×\(height)")

    var opaqueCount = 0
    var xMap = [Int](repeating: 0, count: width)
    for px in 0..<width {
        xMap[px] = min(max(Int(Float(px) * scaleX), 0), Wp - 1)
    }

    let allDetections = stage1Detections + maskResult.mappedStage2Detections

    if allDetections.isEmpty {
        memset(data, 0, width * height * 4)
        print("📊 Output: 0/\(width * height) opaque (0.0%)")
    } else {
        let modelSize: Float = 960.0
        var imageRects = [(x0: Int, y0: Int, x1: Int, y1: Int)]()
        imageRects.reserveCapacity(allDetections.count)

        for det in allDetections {
            let left = det.x - det.width / 2.0
            let right = det.x + det.width / 2.0
            let top = det.y - det.height / 2.0
            let bottom = det.y + det.height / 2.0

            let sx = Float(width) / modelSize
            let sy = Float(height) / modelSize

            var ix0 = Int(floor(left * sx))
            var ix1 = Int(ceil(right * sx))
            var iy0 = Int(floor(top * sy))
            var iy1 = Int(ceil(bottom * sy))

            ix0 = max(0, min(ix0, width))
            ix1 = max(0, min(ix1, width))
            iy0 = max(0, min(iy0, height))
            iy1 = max(0, min(iy1, height))

            if ix0 < ix1 && iy0 < iy1 {
                imageRects.append((x0: ix0, y0: iy0, x1: ix1, y1: iy1))
            }
        }

        var rowIntervals = Array(repeating: [(start:Int,end:Int)](), count: height)
        for r in imageRects {
            for y in r.y0..<r.y1 {
                rowIntervals[y].append((start: r.x0, end: r.x1))
            }
        }

        for y in 0..<height {
            if rowIntervals[y].isEmpty { continue }
            var intervals = rowIntervals[y]
            intervals.sort { $0.start < $1.start }
            var merged: [(Int,Int)] = []
            var cur = intervals[0]
            for i in 1..<intervals.count {
                let it = intervals[i]
                if it.start <= cur.end { cur.end = max(cur.end, it.end) } else { merged.append(cur); cur = it }
            }
            merged.append(cur)
            rowIntervals[y] = merged
        }

        for py in 0..<height {
            let my = min(max(Int(Float(py) * scaleY), 0), Hp - 1)
            let maskRowOffset = my * Wp
            let rowBase = pixels.advanced(by: py * width * 4)

            let intervals = rowIntervals[py]
            if intervals.isEmpty {
                memset(rowBase, 0, width * 4)
                continue
            }

            var x = 0
            var intervalIndex = 0

            while x < width {
                let nextInterval = intervalIndex < intervals.count ? intervals[intervalIndex] : (start: width, end: width)
                if x < nextInterval.start {
                    // Only clear alpha, keep RGB for potential hole filling later
                    let clearEnd = min(nextInterval.start, width)
                    for cx in x..<clearEnd {
                        rowBase.advanced(by: cx * 4 + 3).pointee = 0  // Clear alpha only
                    }
                    x = clearEnd
                    continue
                }

                let runEnd = min(nextInterval.end, width)
                var pxIdx = x
                while pxIdx < runEnd {
                    let maskIdx = maskRowOffset + xMap[pxIdx]
                    let pixelPtr = rowBase.advanced(by: pxIdx * 4)
                    if globalMask[maskIdx] > 0 {
                        pixelPtr[3] = 255
                        opaqueCount += 1
                    } else {
                        // Only clear alpha, keep RGB for potential hole filling later
                        pixelPtr[3] = 0
                    }
                    pxIdx += 1
                }
                x = runEnd
                intervalIndex += 1
            }
        }

        print("📊 Output: \(opaqueCount)/\(width * height) opaque (\(String(format: "%.1f", Float(opaqueCount)/Float(width*height)*100))%)")
    }

    let renderEnd = Date()
    print(String(format: "⏱ applyMaskToImage: %.2f ms", renderEnd.timeIntervalSince(renderStart) * 1000.0))

    return (context: ctx, width: width, height: height)
}

// MARK: - Debug Edge Overlay

/// Apply debug edge overlay and hole filling based on edge fill mode
/// - Parameters:
///   - pixels: Pointer to pixel data
///   - width: Image width
///   - height: Image height
///   - globalMask: The mask array
///   - maskWidth: Mask width (Wp)
///   - maskHeight: Mask height (Hp)
///   - edgeFillMode: Current edge fill mode
func applyDebugEdgeOverlay(
    pixels: UnsafeMutablePointer<UInt8>,
    width: Int,
    height: Int,
    globalMask: [Float],
    maskWidth: Int,
    maskHeight: Int,
    edgeFillMode: EdgeFillMode
) {
    let edgeStart = Date()
    let edges = computeMaskEdges(mask: globalMask, width: maskWidth, height: maskHeight)

    let sxOutline = Float(width) / Float(maskWidth)
    let syOutline = Float(height) / Float(maskHeight)

    switch edgeFillMode {
    case .furniMaterial:
        if !edges.isEmpty {
            // Collect edge x-positions per full-res row (fy)
            var rowXs = Array(repeating: [Int](), count: height)
            for (ex, ey) in edges {
                let fx = Int(Float(ex) * sxOutline)
                let fy = Int(Float(ey) * syOutline)
                guard fx >= 0, fx < width, fy >= 0, fy < height else { continue }
                rowXs[fy].append(fx)
            }

            // Draw edge dots for visibility (only where alpha != 0)
            let r: UInt8 = 255, g: UInt8 = 0, b: UInt8 = 200, a: UInt8 = 255
            for (ex, ey) in edges {
                let fx = Int(Float(ex) * sxOutline)
                let fy = Int(Float(ey) * syOutline)
                let y0 = max(0, fy - 1), y1 = min(height - 1, fy + 1)
                let x0 = max(0, fx - 1), x1 = min(width - 1, fx + 1)
                for yy in y0...y1 {
                    let row = pixels.advanced(by: yy * width * 4)
                    for xx in x0...x1 {
                        let p = row.advanced(by: xx * 4)
                        if p[3] != 0 {
                            p[0] = b; p[1] = g; p[2] = r; p[3] = a
                        }
                    }
                }
            }

            // Fill interior between edge crossings on each scanline (even-odd rule)
            for fy in 0..<height {
                var xs = rowXs[fy]
                if xs.count < 2 { continue }
                xs.sort()
                let row = pixels.advanced(by: fy * width * 4)
                var i = 0
                while i + 1 < xs.count {
                    let start = max(0, min(xs[i], width - 1))
                    let end   = max(0, min(xs[i + 1], width - 1))
                    if end > start {
                        var xx = start
                        while xx < end {
                            let p = row.advanced(by: xx * 4)
                            // Fill ALL pixels inside scanline bounds (fills holes)
                            p[3] = 255
                            xx += 1
                        }
                    }
                    i += 2
                }
            }
        }

    case .clothBased:
        if !edges.isEmpty {
            // MORPHOLOGICAL HOLE FILL: preserves original boundary, only fills interior holes

            // Find bounding box of the mask in full-res coordinates
            var minX = width, maxX = 0, minY = height, maxY = 0
            for (ex, ey) in edges {
                let fx = Int(Float(ex) * sxOutline)
                let fy = Int(Float(ey) * syOutline)
                if fx >= 0 && fx < width && fy >= 0 && fy < height {
                    minX = min(minX, fx)
                    maxX = max(maxX, fx)
                    minY = min(minY, fy)
                    maxY = max(maxY, fy)
                }
            }

            // Add padding for flood fill boundary
            let pad = 2
            minX = max(0, minX - pad)
            maxX = min(width - 1, maxX + pad)
            minY = max(0, minY - pad)
            maxY = min(height - 1, maxY + pad)

            let boxW = maxX - minX + 1
            let boxH = maxY - minY + 1

            if boxW > 4 && boxH > 4 {
                // Create a mask of opaque pixels (from current alpha)
                // 0 = transparent (unknown), 1 = opaque (definitely mask), 2 = exterior (flood-filled)
                var fillMask = [UInt8](repeating: 0, count: boxW * boxH)

                // Mark opaque pixels
                for ly in 0..<boxH {
                    let fy = minY + ly
                    let srcRow = pixels.advanced(by: fy * width * 4)
                    for lx in 0..<boxW {
                        let fx = minX + lx
                        if srcRow[fx * 4 + 3] > 0 {
                            fillMask[ly * boxW + lx] = 1  // Opaque
                        }
                    }
                }

                // Flood fill from edges to mark exterior pixels
                var queue = [(x: Int, y: Int)]()
                queue.reserveCapacity(2 * (boxW + boxH))

                // Add border pixels to queue
                for lx in 0..<boxW {
                    if fillMask[lx] == 0 { queue.append((lx, 0)); fillMask[lx] = 2 }
                    let bottomIdx = (boxH - 1) * boxW + lx
                    if fillMask[bottomIdx] == 0 { queue.append((lx, boxH - 1)); fillMask[bottomIdx] = 2 }
                }
                for ly in 1..<(boxH - 1) {
                    let leftIdx = ly * boxW
                    if fillMask[leftIdx] == 0 { queue.append((0, ly)); fillMask[leftIdx] = 2 }
                    let rightIdx = ly * boxW + boxW - 1
                    if fillMask[rightIdx] == 0 { queue.append((boxW - 1, ly)); fillMask[rightIdx] = 2 }
                }

                // BFS flood fill
                var qIdx = 0
                while qIdx < queue.count {
                    let (cx, cy) = queue[qIdx]
                    qIdx += 1

                    // Check 4-connected neighbors
                    let neighbors = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
                    for (nx, ny) in neighbors {
                        guard nx >= 0 && nx < boxW && ny >= 0 && ny < boxH else { continue }
                        let nIdx = ny * boxW + nx
                        if fillMask[nIdx] == 0 {  // Transparent and not yet visited
                            fillMask[nIdx] = 2  // Mark as exterior
                            queue.append((nx, ny))
                        }
                    }
                }

                // Now fill interior holes: pixels that are still 0 (not opaque, not exterior)
                var holesFilled = 0
                for ly in 0..<boxH {
                    let fy = minY + ly
                    let srcRow = pixels.advanced(by: fy * width * 4)
                    for lx in 0..<boxW {
                        let idx = ly * boxW + lx
                        if fillMask[idx] == 0 {  // Interior hole!
                            let fx = minX + lx
                            srcRow[fx * 4 + 3] = 255  // Fill with original image content
                            holesFilled += 1
                        }
                    }
                }

                print("🕳️ Hole fill: filled \(holesFilled) interior hole pixels")
            }

            // Draw original edge outline in magenta (not alpha shape)
            let r: UInt8 = 255, g: UInt8 = 0, b: UInt8 = 200, a: UInt8 = 255
            for (ex, ey) in edges {
                let fx = Int(Float(ex) * sxOutline)
                let fy = Int(Float(ey) * syOutline)
                guard fx >= 1 && fx < width - 1 && fy >= 1 && fy < height - 1 else { continue }

                // Draw 3x3 dot
                for yy in (fy - 1)...(fy + 1) {
                    let row = pixels.advanced(by: yy * width * 4)
                    for xx in (fx - 1)...(fx + 1) {
                        let p = row.advanced(by: xx * 4)
                        if p[3] != 0 {
                            p[0] = b; p[1] = g; p[2] = r; p[3] = a
                        }
                    }
                }
            }

            print("🔺 Edge outline: \(edges.count) boundary points (original mask edges preserved)")
        }

    case .chairType:
        // Same as furniMaterial - draw edges and fill with scanline
        if !edges.isEmpty {
            var rowXs = Array(repeating: [Int](), count: height)
            for (ex, ey) in edges {
                let fx = Int(Float(ex) * sxOutline)
                let fy = Int(Float(ey) * syOutline)
                guard fx >= 0, fx < width, fy >= 0, fy < height else { continue }
                rowXs[fy].append(fx)
            }

            // Draw edge dots
            let r: UInt8 = 255, g: UInt8 = 0, b: UInt8 = 200, a: UInt8 = 255
            for (ex, ey) in edges {
                let fx = Int(Float(ex) * sxOutline)
                let fy = Int(Float(ey) * syOutline)
                let y0 = max(0, fy - 1), y1 = min(height - 1, fy + 1)
                let x0 = max(0, fx - 1), x1 = min(width - 1, fx + 1)
                for yy in y0...y1 {
                    let row = pixels.advanced(by: yy * width * 4)
                    for xx in x0...x1 {
                        let p = row.advanced(by: xx * 4)
                        if p[3] != 0 {
                            p[0] = b; p[1] = g; p[2] = r; p[3] = a
                        }
                    }
                }
            }

            // Fill interior with scanline
            for fy in 0..<height {
                var xs = rowXs[fy]
                if xs.count < 2 { continue }
                xs.sort()
                let row = pixels.advanced(by: fy * width * 4)
                var i = 0
                while i + 1 < xs.count {
                    let start = max(0, min(xs[i], width - 1))
                    let end   = max(0, min(xs[i + 1], width - 1))
                    if end > start {
                        var xx = start
                        while xx < end {
                            let p = row.advanced(by: xx * 4)
                            p[3] = 255
                            xx += 1
                        }
                    }
                    i += 2
                }
            }
        }
    }

    let edgeDt = Date().timeIntervalSince(edgeStart) * 1000.0
    print(String(format: "⏱ Edge overlay: %.2f ms (%d pts)", edgeDt, edges.count))
}
