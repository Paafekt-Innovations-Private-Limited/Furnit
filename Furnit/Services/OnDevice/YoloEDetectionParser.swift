import Foundation
import CoreML
import Accelerate

// BLAS via C wrapper (same as FurnitureFitView.swift)
fileprivate typealias BLASInt = Int32

@inline(__always)
fileprivate func yolo_blas_scopy(_ n: BLASInt, _ x: UnsafePointer<Float>, _ incx: BLASInt, _ y: UnsafeMutablePointer<Float>, _ incy: BLASInt) {
    BlasScopy(n, x, incx, y, incy)
}

/// Shared YOLO-E detection tensor decoding for still-image calibration and optional reuse.
enum YoloEDetectionParser {

    static func parseDetections(
        detArray: MLMultiArray,
        confidenceThreshold: Float,
        classBlacklist: Set<Int>
    ) -> [FurnitureFitDetection] {
        guard detArray.dataType == .float32 else { return [] }
        let totalCount = detArray.count
        let detBuf = detArray.dataPointer.bindMemory(to: Float.self, capacity: totalCount)

        let dim1 = detArray.shape[1].intValue
        let dim2 = detArray.shape[2].intValue
        let isNewFormat = dim2 < 100

        var allDets: [FurnitureFitDetection] = []
        allDets.reserveCapacity(512)

        if isNewFormat {
            let numDetections = dim1
            let featuresPerDet = dim2
            let strides = detArray.strides.map { $0.intValue }
            let detStride = strides.count >= 2 ? strides[1] : featuresPerDet
            let featStride = strides.count >= 3 ? strides[2] : 1

            for detIdx in 0..<numDetections {
                let base = detIdx * detStride
                let x1 = detBuf[base + 0 * featStride]
                let y1 = detBuf[base + 1 * featStride]
                let x2 = detBuf[base + 2 * featStride]
                let y2 = detBuf[base + 3 * featStride]
                let confidence = detBuf[base + 4 * featStride]
                let classIdxFloat = detBuf[base + 5 * featStride]

                guard x1.isFinite, y1.isFinite, x2.isFinite, y2.isFinite, confidence.isFinite, classIdxFloat.isFinite else { continue }

                let w = x2 - x1
                let h = y2 - y1
                let x = (x1 + x2) / 2.0
                let y = (y1 + y2) / 2.0

                guard confidence >= confidenceThreshold, confidence <= 1.0 else { continue }
                guard w > 0, h > 0, w < 2000, h < 2000 else { continue }

                let classIdx = Int(classIdxFloat)
                var coeffs = [Float](repeating: 0, count: 32)
                var validCoeffs = true
                for k in 0..<32 {
                    let coeffIndex = base + (6 + k) * featStride
                    if coeffIndex < totalCount {
                        let val = detBuf[coeffIndex]
                        if val.isFinite { coeffs[k] = val } else { validCoeffs = false; break }
                    } else {
                        validCoeffs = false
                        break
                    }
                }
                guard validCoeffs else { continue }
                guard classIdx >= 0, !classBlacklist.contains(classIdx) else { continue }

                allDets.append(FurnitureFitDetection(x: x, y: y, w: w, h: h, confidence: confidence, classIdx: classIdx, coeffs: coeffs))
            }
        } else {
            let numFeatures = dim1
            let numAnchors = dim2
            let numClasses = numFeatures - 4 - 32
            guard numFeatures >= 36, numAnchors > 0, numClasses > 0 else { return [] }

            let stride = numAnchors
            let coeffOffset = 4 + numClasses
            var tempScores = [Float](repeating: 0, count: numClasses)

            for anchor in 0..<numAnchors {
                let x = detBuf[0 * stride + anchor]
                let y = detBuf[1 * stride + anchor]
                let w = detBuf[2 * stride + anchor]
                let h = detBuf[3 * stride + anchor]

                guard x.isFinite, y.isFinite, w.isFinite, h.isFinite, w > 0, h > 0 else { continue }

                let basePtr = detBuf.advanced(by: 4 * stride + anchor)
                yolo_blas_scopy(BLASInt(numClasses), basePtr, BLASInt(stride), &tempScores, 1)

                var maxVal: Float = 0
                var maxIdx: vDSP_Length = 0
                vDSP_maxvi(tempScores, 1, &maxVal, &maxIdx, vDSP_Length(numClasses))
                let classIdx = Int(maxIdx)

                guard maxVal >= confidenceThreshold, !classBlacklist.contains(classIdx) else { continue }

                var coeffs = [Float](repeating: 0, count: 32)
                let coeffBase = detBuf.advanced(by: coeffOffset * stride + anchor)
                yolo_blas_scopy(32, coeffBase, BLASInt(stride), &coeffs, 1)

                allDets.append(FurnitureFitDetection(x: x, y: y, w: w, h: h, confidence: maxVal, classIdx: classIdx, coeffs: coeffs))
            }
        }

        return allDets
    }

    static func extractDetectionAndProto(from output: MLFeatureProvider) -> (det: MLMultiArray, proto: MLMultiArray)? {
        if let det = output.featureValue(for: "var_2374")?.multiArrayValue,
           let proto = output.featureValue(for: "var_2412")?.multiArrayValue {
            return (det, proto)
        }
        if let det = output.featureValue(for: "detections")?.multiArrayValue,
           let proto = output.featureValue(for: "protos")?.multiArrayValue {
            return (det, proto)
        }
        return nil
    }
}
