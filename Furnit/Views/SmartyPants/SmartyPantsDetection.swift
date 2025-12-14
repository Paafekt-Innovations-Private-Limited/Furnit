// SmartyPantsDetection.swift
// Detection extraction and NMS utilities for SmartyPants

import Foundation
import CoreML
import Accelerate

// MARK: - Detection Extraction

/// Extract detections from model output tensor
/// - Parameters:
///   - detections: MLMultiArray from model output
///   - confThreshold: Minimum confidence threshold
///   - detectAllObjects: If true, detect all classes; if false, only furniture classes
///   - furnitureClasses: Dictionary mapping class indices to furniture names
///   - debugMode: Whether to print debug info
/// - Returns: Array of DetectionSmarty objects
func extractDetections(
    from detections: MLMultiArray,
    confThreshold: Float,
    detectAllObjects: Bool,
    furnitureClasses: [Int: String],
    debugMode: Bool
) -> [DetectionSmarty] {
    let t0 = Date()
    var all: [DetectionSmarty] = []

    let numFeatures = detections.shape[1].intValue
    let numAnchors = detections.shape[2].intValue
    let numClasses = numFeatures - 4 - 32

    // Validate tensor dimensions
    guard numFeatures >= 36 && numAnchors > 0 && numClasses > 0 else {
        if debugMode {
            print("⚠️ extractDetections: Invalid tensor dimensions - features:\(numFeatures), anchors:\(numAnchors), classes:\(numClasses)")
        }
        return []
    }

    if debugMode {
        print("🔍 Tensor shape: [1, \(numFeatures), \(numAnchors)]")
        print("   → \(numClasses) classes, \(numAnchors) predictions")
        print("   → Mode: \(detectAllObjects ? "ALL OBJECTS" : "FURNITURE ONLY")")
        if numClasses == 4585 {
            print("   → Model: YOLOE (LVIS open-vocabulary)")
        } else if numClasses == 80 {
            print("   → Model: YOLO11-seg (COCO)")
        }
    }

    let totalCount = detections.count

    // Validate total count matches expected dimensions
    let expectedCount = numFeatures * numAnchors
    guard totalCount >= expectedCount else {
        if debugMode {
            print("⚠️ extractDetections: Array size mismatch - expected:\(expectedCount), got:\(totalCount)")
        }
        return []
    }

    let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
    defer { detBuf.deallocate() }

    let copyStart = Date()
    if detections.dataType == .float16 {
        let src = detections.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
        var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src),
                                   height: 1, width: vImagePixelCount(totalCount),
                                   rowBytes: totalCount * MemoryLayout<UInt16>.size)
        var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(detBuf),
                                   height: 1, width: vImagePixelCount(totalCount),
                                   rowBytes: totalCount * MemoryLayout<Float>.size)
        let result = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
        if result != kvImageNoError && debugMode {
            print("⚠️ extractDetections: vImage conversion failed: \(result)")
        }
    } else if detections.dataType == .float32 {
        let src = detections.dataPointer.assumingMemoryBound(to: Float.self)
        memcpy(detBuf, src, totalCount * MemoryLayout<Float>.size)
    } else {
        for i in 0..<totalCount {
            detBuf[i] = detections[i].floatValue
        }
    }
    let copyEnd = Date()
    if debugMode {
        print(String(format: "⏱ extractDetections copy/convert: %.2f ms",
                     copyEnd.timeIntervalSince(copyStart) * 1000.0))
    }

    let coeffOffset = 4 + numClasses
    let stride = numAnchors

    let decodeStart = Date()
    if detectAllObjects {
        for anchor in 0..<numAnchors {
            // Bounds checking for coordinate access
            guard anchor < stride,
                  1 * stride + anchor < totalCount,
                  2 * stride + anchor < totalCount,
                  3 * stride + anchor < totalCount else {
                if debugMode { print("⚠️ Coordinate bounds check failed for anchor \(anchor)") }
                continue
            }

            let x = detBuf[0 * stride + anchor]
            let y = detBuf[1 * stride + anchor]
            let w = detBuf[2 * stride + anchor]
            let h = detBuf[3 * stride + anchor]

            // Validate coordinate values
            guard x.isFinite && y.isFinite && w.isFinite && h.isFinite && w > 0 && h > 0 else {
                continue
            }

            var bestConf: Float = 0
            var bestClassIdx = -1

            let baseConfIdx = 4 * stride + anchor
            for classIdx in 0..<numClasses {
                let confIndex = baseConfIdx + classIdx * stride
                guard confIndex < totalCount else {
                    if debugMode { print("⚠️ Confidence bounds check failed for class \(classIdx), anchor \(anchor)") }
                    break
                }

                let conf = detBuf[confIndex]
                if conf > bestConf && conf.isFinite {
                    bestConf = conf
                    bestClassIdx = classIdx
                }
            }

            if bestConf > confThreshold && bestClassIdx >= 0 {
                var coeffs = [Float](repeating: 0, count: 32)
                let coeffStart = coeffOffset * stride + anchor

                // Bounds checking for mask coefficients
                var validCoeffs = true
                for k in 0..<32 {
                    let coeffIndex = coeffStart + k * stride
                    if coeffIndex < totalCount {
                        coeffs[k] = detBuf[coeffIndex]
                    } else {
                        if debugMode { print("⚠️ Coefficient bounds check failed for k=\(k), anchor=\(anchor)") }
                        validCoeffs = false
                        break
                    }
                }

                if validCoeffs {
                    let className = furnitureClasses[bestClassIdx] ?? "object_\(bestClassIdx)"
                    all.append(DetectionSmarty(
                        x: x, y: y, width: w, height: h,
                        confidence: bestConf, classIdx: bestClassIdx, className: className,
                        maskCoeffs: coeffs
                    ))
                }
            }
        }
    } else {
        let furnitureList = furnitureClasses.filter { $0.key < numClasses }

        // Debug: track best furniture confidence found
        var bestFurnitureConf: Float = 0
        var bestFurnitureClass = ""
        var bestFurnitureAnchor = -1

        for anchor in 0..<numAnchors {
            // Bounds checking for coordinate access
            guard anchor < stride,
                  1 * stride + anchor < totalCount,
                  2 * stride + anchor < totalCount,
                  3 * stride + anchor < totalCount else {
                if debugMode { print("⚠️ Coordinate bounds check failed for anchor \(anchor)") }
                continue
            }

            let x = detBuf[0 * stride + anchor]
            let y = detBuf[1 * stride + anchor]
            let w = detBuf[2 * stride + anchor]
            let h = detBuf[3 * stride + anchor]

            // Validate coordinate values
            guard x.isFinite && y.isFinite && w.isFinite && h.isFinite && w > 0 && h > 0 else {
                continue
            }

            for (classIdx, className) in furnitureList {
                let confIdx = (4 + classIdx) * stride + anchor
                guard confIdx < totalCount else {
                    if debugMode { print("⚠️ Furniture confidence bounds check failed for class \(classIdx), anchor \(anchor)") }
                    continue
                }

                let conf = detBuf[confIdx]

                // Track best furniture confidence for debug
                if conf > bestFurnitureConf && conf.isFinite {
                    bestFurnitureConf = conf
                    bestFurnitureClass = className
                    bestFurnitureAnchor = anchor
                }

                if conf > confThreshold && conf.isFinite {
                    var coeffs = [Float](repeating: 0, count: 32)
                    let coeffStart = coeffOffset * stride + anchor

                    // Bounds checking for mask coefficients
                    var validCoeffs = true
                    for k in 0..<32 {
                        let coeffIndex = coeffStart + k * stride
                        if coeffIndex < totalCount {
                            coeffs[k] = detBuf[coeffIndex]
                        } else {
                            if debugMode { print("⚠️ Coefficient bounds check failed for k=\(k), anchor=\(anchor)") }
                            validCoeffs = false
                            break
                        }
                    }

                    if validCoeffs {
                        all.append(DetectionSmarty(
                            x: x, y: y, width: w, height: h,
                            confidence: conf, classIdx: classIdx, className: className,
                            maskCoeffs: coeffs
                        ))
                    }
                }
            }
        }

        // Debug: show best furniture confidence found even if below threshold
        if debugMode && all.isEmpty {
            print("🔍 FURNITURE DEBUG: Best conf=\(String(format: "%.3f", bestFurnitureConf)) for '\(bestFurnitureClass)' at anchor \(bestFurnitureAnchor)")
            print("   Threshold was: \(confThreshold), furniture classes checked: \(furnitureList.count)")

            // Also find best confidence across ALL classes for comparison
            var bestOverallConf: Float = 0
            var bestOverallClass = -1
            for anchor in 0..<min(100, numAnchors) {  // Check first 100 anchors
                for classIdx in 0..<numClasses {
                    let confIdx = (4 + classIdx) * stride + anchor
                    if confIdx < totalCount {
                        let conf = detBuf[confIdx]
                        if conf > bestOverallConf && conf.isFinite {
                            bestOverallConf = conf
                            bestOverallClass = classIdx
                        }
                    }
                }
            }
            print("   Best OVERALL conf=\(String(format: "%.3f", bestOverallConf)) for class \(bestOverallClass)")
        }
    }
    let decodeEnd = Date()

    if debugMode {
        print(String(format: "⏱ extractDetections decode loop: %.2f ms",
                     decodeEnd.timeIntervalSince(decodeStart) * 1000.0))

        let grouped = Dictionary(grouping: all) { $0.className }
        print("\n📊 DETECTION SUMMARY: \(all.count) total")
        for (className, dets) in grouped.sorted(by: { $0.value.count > $1.value.count }).prefix(20) {
            let confidences = dets.map { Int($0.confidence * 100) }
            print("  - \(className): \(dets.count)x, conf: \(confidences)%")
        }
        if grouped.count > 20 {
            print("  ... and \(grouped.count - 20) more classes")
        }
        let tEnd = Date()
        print(String(format: "⏱ extractDetections total: %.2f ms",
                     tEnd.timeIntervalSince(t0) * 1000.0))
    }

    return all
}

// MARK: - Detection Filtering

/// Keep only detections that overlap with the primary detection
func keepOverlappingDetections(_ detections: [DetectionSmarty]) -> [DetectionSmarty] {
    guard detections.count > 0 else { return [] }
    if detections.count == 1 { return detections }

    let sorted = detections.sorted { $0.confidence > $1.confidence }
    let primary = sorted[0]
    let pLeft = primary.x - primary.width / 2
    let pRight = primary.x + primary.width / 2
    let pTop = primary.y - primary.height / 2
    let pBottom = primary.y + primary.height / 2

    var kept: [DetectionSmarty] = []
    kept.reserveCapacity(sorted.count)

    for det in sorted {
        let aLeft = det.x - det.width / 2
        let aRight = det.x + det.width / 2
        let aTop = det.y - det.height / 2
        let aBottom = det.y + det.height / 2

        if aRight < pLeft || pRight < aLeft { continue }
        if aBottom < pTop || pBottom < aTop { continue }
        kept.append(det)
    }
    return kept
}
