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
        
        guard primaryBBox.maskCoeffs.count == C else { return }
        guard !primaryBBox.maskCoeffs.contains(where: { !$0.isFinite }) else { return }
        
        var rawMask = [Float](repeating: 0, count: spatial)
        primaryBBox.maskCoeffs.withUnsafeBufferPointer { coeffPtr in
            protoMatrix.withUnsafeBufferPointer { protoPtr in
                guard let coeffBase = coeffPtr.baseAddress,
                      let protoBase = protoPtr.baseAddress else { return }
                vDSP_mmul(coeffBase, 1, protoBase, 1, &rawMask, 1,
                          1, vDSP_Length(spatial), vDSP_Length(C))
            }
        }
        
        for i in 0..<spatial {
            if rawMask[i] > maskThreshold {
                globalMask[i] = 1.0
            }
        }
        
        if debugMode {
            var sum: Float = 0
            globalMask.withUnsafeBufferPointer { ptr in
                if let base = ptr.baseAddress {
                    vDSP_sve(base, 1, &sum, vDSP_Length(spatial))
                }
            }
            let area = Int(sum.rounded())
            let percent = Float(area) / Float(spatial) * 100.0
            print("🔷 Primary-only mask: \(area)px (\(String(format: "%.1f", percent))%)")
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
        
        let bboxX1 = max(0, Int((primaryBBox.x - primaryBBox.width / 2) * scale))
        let bboxY1 = max(0, Int((primaryBBox.y - primaryBBox.height / 2) * scale))
        let bboxX2 = min(Wp, Int((primaryBBox.x + primaryBBox.width / 2) * scale))
        let bboxY2 = min(Hp, Int((primaryBBox.y + primaryBBox.height / 2) * scale))
        
        var detMasks = [[Float]]()
        var detAreas = [Int]()
        
        for det in allDetections {
            var rawMask = [Float](repeating: 0, count: spatial)
            guard det.maskCoeffs.count == C else {
                detMasks.append(rawMask)
                detAreas.append(0)
                continue
            }
            guard !det.maskCoeffs.contains(where: { !$0.isFinite }) else {
                detMasks.append(rawMask)
                detAreas.append(0)
                continue
            }
            
            det.maskCoeffs.withUnsafeBufferPointer { coeffPtr in
                protoMatrix.withUnsafeBufferPointer { protoPtr in
                    guard let coeffBase = coeffPtr.baseAddress,
                          let protoBase = protoPtr.baseAddress else { return }
                    vDSP_mmul(coeffBase, 1, protoBase, 1, &rawMask, 1,
                              1, vDSP_Length(spatial), vDSP_Length(C))
                }
            }
            
            var area = 0
            for y in 0..<Hp {
                let rowOffset = y * Wp
                for x in 0..<Wp {
                    let idx = rowOffset + x
                    if rawMask[idx] > maskThreshold &&
                       x >= bboxX1 && x < bboxX2 &&
                       y >= bboxY1 && y < bboxY2 {
                        rawMask[idx] = 1.0
                        area += 1
                    } else {
                        rawMask[idx] = 0.0
                    }
                }
            }
            detMasks.append(rawMask)
            detAreas.append(area)
        }
        
        // Start with largest area mask
        guard let primaryIdx = detAreas.indices.max(by: { detAreas[$0] < detAreas[$1] }) else { return }
        guard detAreas[primaryIdx] > 0 else {
            if debugMode { print("⚠️ No valid masks with area > 0") }
            return
        }
        
        for i in 0..<spatial { globalMask[i] = detMasks[primaryIdx][i] }
        
        var used = [Bool](repeating: false, count: allDetections.count)
        used[primaryIdx] = true
        
        if debugMode {
            print("🔷 Primary (largest): \(allDetections[primaryIdx].className) @ \(Int(allDetections[primaryIdx].confidence * 100))%, area=\(detAreas[primaryIdx])px")
        }
        
        // Iteratively add masks with >= minOverlap
        var changed = true
        while changed {
            changed = false
            for (idx, detMask) in detMasks.enumerated() {
                if used[idx] || detAreas[idx] == 0 { continue }
                
                // Overlap count via dot product on 0/1 masks
                var overlapFloat: Float = 0
                detMask.withUnsafeBufferPointer { detPtr in
                    globalMask.withUnsafeBufferPointer { globPtr in
                        if let d = detPtr.baseAddress, let g = globPtr.baseAddress {
                            vDSP_dotpr(d, 1, g, 1, &overlapFloat, vDSP_Length(spatial))
                        }
                    }
                }
                let overlapCount = Int(overlapFloat.rounded())
                
                let overlapRatio = Float(overlapCount) / Float(detAreas[idx])
                
                if overlapRatio >= minOverlap {
                    var added = 0
                    for i in 0..<spatial {
                        if detMask[i] > 0 && globalMask[i] == 0 {
                            globalMask[i] = 1.0
                            added += 1
                        }
                    }
                    used[idx] = true
                    changed = true
                    
                    if debugMode {
                        print("🔗 Merged \(allDetections[idx].className) @ \(Int(allDetections[idx].confidence * 100))%: overlap=\(Int(overlapRatio * 100))%, +\(added)px")
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
            print("🔷 buildGlobalMask: \(mergedCount)/\(allDetections.count) merged, total=\(totalArea)px")
        }
    }

    // MARK: - Fill Inside Perimeter
    func fillInsidePerimeter(_ mask: inout [Float], width: Int, height: Int) {
        let count = width * height
        
        // Step 1: Dilate to seal perimeter gaps
        var sealed = [Float](repeating: 0, count: count)
        let radius = 3
        
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
        
        // Step 2: Flood fill from edges to mark EXTERIOR
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
        
        // Step 3: NOT exterior = interior = fill with 1
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
