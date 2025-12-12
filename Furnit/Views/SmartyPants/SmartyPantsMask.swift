// SmartyPantsMask.swift
// Mask building, post-processing, and fill operations

import Accelerate

extension SmartyPantsContainerView {
    
    // MARK: - Build Stitched Mask (Primary Only)
    func buildStitchedMask(
        globalMask: inout [Float],
        allDetections: [DetectionSmarty],
        protoMatrix: [Float],
        primaryBBox: DetectionSmarty,
        C: Int, Wp: Int, Hp: Int
    ) {
        let spatial = Wp * Hp

        // Same guards as before – don’t build nonsense masks
        guard primaryBBox.maskCoeffs.count == C else { return }
        guard !primaryBBox.maskCoeffs.contains(where: { !$0.isFinite }) else { return }

        var rawMask = [Float](repeating: 0, count: spatial)

        primaryBBox.maskCoeffs.withUnsafeBufferPointer { coeffPtr in
            protoMatrix.withUnsafeBufferPointer { protoPtr in
                rawMask.withUnsafeMutableBufferPointer { rawPtr in
                    guard
                        let A = coeffPtr.baseAddress,      // 1×C
                        let B = protoPtr.baseAddress,      // C×spatial
                        let Cptr = rawPtr.baseAddress      // 1×spatial
                    else { return }

                    // BLAS: (1×C) * (C×spatial) = (1×spatial)
                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        1,                      // M = 1 row
                        Int32(spatial),         // N = spatial
                        Int32(C),               // K = C
                        1.0,                    // alpha
                        A, Int32(C),            // A: lda = C
                        B, Int32(spatial),      // B: ldb = spatial
                        0.0,                    // beta
                        Cptr, Int32(spatial)    // C: ldc = spatial
                    )
                }
            }
        }

        // Threshold into globalMask exactly as before
        for i in 0..<spatial {
            if rawMask[i] > maskThreshold {
                globalMask[i] = 1.0
            }
        }

        if debugMode {
            var area = 0
            for i in 0..<spatial {
                if globalMask[i] > 0 { area += 1 }
            }
            let percentage = String(format: "%.1f", Float(area) / Float(spatial) * 100)
            print("🔷 Primary-only mask: \(area)px (\(percentage)%)")
        }
    }

    // MARK: - Build Global Mask with Overlap Filter
    func buildGlobalMaskWithOverlapFilter(
        globalMask: inout [Float],
        allDetections: [DetectionSmarty],
        protoMatrix: [Float],
        primaryBBox: DetectionSmarty,
        C: Int, Wp: Int, Hp: Int,
        minOverlap: Float = 0.5
    ) {
        let spatial = Wp * Hp
        let scale = Float(Wp) / kModelInputSizeFloat

        // BBox in mask space (same as before)
        let bboxX1 = max(0, Int((primaryBBox.x - primaryBBox.width / 2) * scale))
        let bboxY1 = max(0, Int((primaryBBox.y - primaryBBox.height / 2) * scale))
        let bboxX2 = min(Wp, Int((primaryBBox.x + primaryBBox.width / 2) * scale))
        let bboxY2 = min(Hp, Int((primaryBBox.y + primaryBBox.height / 2) * scale))

        let numDet = allDetections.count
        guard numDet > 0 else { return }

        // ===== 1) Build coefficient matrix [numDet × C] =====
        var coeffMatrix = [Float](repeating: 0, count: numDet * C)

        for (i, det) in allDetections.enumerated() {
            let rowOffset = i * C
            let coeffs = det.maskCoeffs

            // Keep old guards: invalid coeffs → effectively zero mask
            if coeffs.count != C || coeffs.contains(where: { !$0.isFinite }) {
                continue
            }

            for c in 0..<C {
                coeffMatrix[rowOffset + c] = coeffs[c]
            }
        }

        // ===== 2) BLAS: A[numDet×C] × B[C×spatial] = masksFlat[numDet×spatial] =====
        var masksFlat = [Float](repeating: 0, count: numDet * spatial)

        coeffMatrix.withUnsafeBufferPointer { aPtr in
            protoMatrix.withUnsafeBufferPointer { bPtr in
                masksFlat.withUnsafeMutableBufferPointer { cPtr in
                    guard
                        let A = aPtr.baseAddress,
                        let B = bPtr.baseAddress,
                        let Cptr = cPtr.baseAddress
                    else { return }

                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        Int32(numDet),          // M
                        Int32(spatial),         // N
                        Int32(C),               // K
                        1.0,                    // alpha
                        A, Int32(C),            // lda = C
                        B, Int32(spatial),      // ldb = spatial
                        0.0,                    // beta
                        Cptr, Int32(spatial)    // ldc = spatial
                    )
                }
            }
        }

        // ===== 3) Threshold each detection’s mask inside bbox only (same semantics) =====
        var detMasks: [[Float]] = []
        var detAreas: [Int] = []
        detMasks.reserveCapacity(numDet)
        detAreas.reserveCapacity(numDet)

        for detIndex in 0..<numDet {
            var rawMask = [Float](repeating: 0, count: spatial)
            var area = 0
            let rowBase = detIndex * spatial

            // Zero everywhere except bbox; apply maskThreshold just like before
            for y in bboxY1..<bboxY2 {
                let rowOffset = y * Wp
                for x in bboxX1..<bboxX2 {
                    let pos = rowOffset + x
                    let val = masksFlat[rowBase + pos]

                    if val > maskThreshold {
                        rawMask[pos] = 1.0
                        area += 1
                    } else {
                        rawMask[pos] = 0.0
                    }
                }
            }

            detMasks.append(rawMask)
            detAreas.append(area)
        }

        // ===== 4) Original union logic: start from largest area and merge =====
        guard let primaryIdx = detAreas.indices.max(by: { detAreas[$0] < detAreas[$1] }) else { return }
        guard detAreas[primaryIdx] > 0 else {
            if debugMode { print("⚠️ No valid masks with area > 0") }
            return
        }

        // Initialize globalMask from largest mask
        for i in 0..<spatial {
            globalMask[i] = detMasks[primaryIdx][i]
        }

        var used = [Bool](repeating: false, count: numDet)
        used[primaryIdx] = true

        if debugMode {
            print("🔷 Primary (largest): \(allDetections[primaryIdx].className) @ \(Int(allDetections[primaryIdx].confidence * 100))%, area=\(detAreas[primaryIdx])px")
        }

        // Iteratively merge detections with sufficient overlap
        var changed = true
        while changed {
            changed = false

            for detIndex in 0..<numDet {
                if used[detIndex] || detAreas[detIndex] == 0 { continue }

                let detMask = detMasks[detIndex]

                // intersection = dot(detMask, globalMask) (because both are 0/1)
                var overlapFloat: Float = 0
                detMask.withUnsafeBufferPointer { detPtr in
                    globalMask.withUnsafeBufferPointer { globPtr in
                        if let d = detPtr.baseAddress, let g = globPtr.baseAddress {
                            vDSP_dotpr(d, 1, g, 1, &overlapFloat, vDSP_Length(spatial))
                        }
                    }
                }

                let overlapCount = Int(overlapFloat.rounded())
                let overlapRatio = Float(overlapCount) / Float(detAreas[detIndex])

                if overlapRatio >= minOverlap {
                    var added = 0
                    for i in 0..<spatial {
                        if detMask[i] > 0 && globalMask[i] == 0 {
                            globalMask[i] = 1.0
                            added += 1
                        }
                    }

                    used[detIndex] = true
                    changed = true

                    if debugMode {
                        print("🔗 Merged \(allDetections[detIndex].className) @ \(Int(allDetections[detIndex].confidence * 100))%: overlap=\(Int(overlapRatio * 100))%, +\(added)px")
                    }
                }
            }
        }

        if debugMode {
            let mergedCount = used.filter { $0 }.count
            var sum: Float = 0
            globalMask.withUnsafeBufferPointer { ptr in
                if let base = ptr.baseAddress {
                    vDSP_sve(base, 1, &sum, vDSP_Length(spatial))
                }
            }
            let totalArea = Int(sum.rounded())
            print("🔷 buildGlobalMaskWithOverlapFilter: \(mergedCount)/\(allDetections.count) merged, total=\(totalArea)px")
        }
    }



    // MARK: - Fill Inside Perimeter (with vImage dilation)
    func fillInsidePerimeter(_ mask: inout [Float], width: Int, height: Int) {
        let count = width * height
        
        // ========= Step 1: Dilate to seal perimeter gaps (vImage) =========
        var sealed = [Float](repeating: 0, count: count)
        let radius = 3
        let kernelSide = radius * 2 + 1      // 7×7 kernel
        let kernelCount = kernelSide * kernelSide
        var kernel = [UInt8](repeating: 1, count: kernelCount)
        
        // Convert Float mask (0/1) → Planar8 (0/255)
        var srcPlanar = [UInt8](repeating: 0, count: count)
        for i in 0..<count {
            srcPlanar[i] = mask[i] > 0 ? 255 : 0
        }
        var dstPlanar = [UInt8](repeating: 0, count: count)
        
        var srcBuf = vImage_Buffer(
            data: &srcPlanar,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width
        )
        var dstBuf = vImage_Buffer(
            data: &dstPlanar,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: width
        )
        
        let dilateStart = Date()
        let err = vImageDilate_Planar8(
            &srcBuf,
            &dstBuf,
            0, 0,
            &kernel,
            vImagePixelCount(kernelSide),   // FIXED
            vImagePixelCount(kernelSide),   // FIXED
            vImage_Flags(kvImageNoFlags)
        )

        let dilateEnd = Date()
        
        if err == kvImageNoError {
            // Convert back to Float 0/1
            var sealedCount = 0
            for i in 0..<count {
                if dstPlanar[i] > 0 {
                    sealed[i] = 1.0
                    sealedCount += 1
                } else {
                    sealed[i] = 0.0
                }
            }
            if debugMode {
                print(String(format: "⏱ fillInsidePerimeter dilation (vImage): %.2f ms, sealed=%d",
                             dilateEnd.timeIntervalSince(dilateStart) * 1000.0,
                             sealedCount))
            }
        } else {
            // Fallback to original CPU dilation if vImage fails for some reason
            if debugMode {
                print("⚠️ vImageDilate_Planar8 failed with error \(err), falling back to CPU dilation")
            }
            for y in 0..<height {
                for x in 0..<width {
                    if mask[y * width + x] > 0 {
                        for dy in -radius...radius {
                            for dx in -radius...radius {
                                let nx = x + dx, ny = y + dy
                                if nx >= 0 && nx < width && ny >= 0 && ny < height {
                                    sealed[ny * width + nx] = 1.0
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // ========= Step 2: Flood fill from edges to mark EXTERIOR =========
        var exterior = [Bool](repeating: false, count: count)
        var queue = [Int]()
        queue.reserveCapacity(count / 4)
        
        for x in 0..<width {
            if sealed[x] == 0 { queue.append(x); exterior[x] = true }
            let b = (height - 1) * width + x
            if sealed[b] == 0 && !exterior[b] { queue.append(b); exterior[b] = true }
        }
        for y in 0..<height {
            let l = y * width
            let r = y * width + width - 1
            if sealed[l] == 0 && !exterior[l] { queue.append(l); exterior[l] = true }
            if sealed[r] == 0 && !exterior[r] { queue.append(r); exterior[r] = true }
        }
        
        let floodStart = Date()
        var head = 0
        while head < queue.count {
            let idx = queue[head]; head += 1
            let x = idx % width, y = idx / width
            
            let neighbors = [idx - 1, idx + 1, idx - width, idx + width]
            let valid = [x > 0, x < width - 1, y > 0, y < height - 1]
            
            for i in 0..<4 {
                if valid[i] {
                    let n = neighbors[i]
                    if sealed[n] == 0 && !exterior[n] {
                        exterior[n] = true
                        queue.append(n)
                    }
                }
            }
        }
        if debugMode {
            let floodEnd = Date()
            print(String(format: "⏱ fillInsidePerimeter flood: %.2f ms", floodEnd.timeIntervalSince(floodStart) * 1000.0))
        }
        
        // ========= Step 3: NOT exterior = interior = fill with 1 =========
        var filled = 0
        for i in 0..<count {
            if !exterior[i] {
                if mask[i] == 0 { filled += 1 }
                mask[i] = 1.0
            }
        }
        
        if debugMode { print("🔷 fillInsidePerimeter: \(filled)px holes filled") }
    }

    // MARK: - Binary Mask Conversion
    func makeBinaryMaskFromGlobalMask(_ globalMask: [Float], count: Int) -> [UInt8] {
        var scaled = [Float](repeating: 0, count: count)
        var scale255: Float = 255.0
        
        globalMask.withUnsafeBufferPointer { src in
            scaled.withUnsafeMutableBufferPointer { dst in
                if let s = src.baseAddress, let d = dst.baseAddress {
                    vDSP_vsmul(s, 1, &scale255, d, 1, vDSP_Length(count))
                }
            }
        }
        
        var binary = [UInt8](repeating: 0, count: count)
        scaled.withUnsafeBufferPointer { src in
            binary.withUnsafeMutableBufferPointer { dst in
                if let s = src.baseAddress, let d = dst.baseAddress {
                    vDSP_vfixu8(s, 1, d, 1, vDSP_Length(count))
                }
            }
        }
        return binary
    }
}
