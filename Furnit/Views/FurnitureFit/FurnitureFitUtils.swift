// FurnitureFitUtils.swift
// Utility functions for FurnitureFit segmentation - extracted for testability

import CoreGraphics
import Darwin
import Foundation

// MARK: - Detection Structure
/// Represents a detected object with bounding box, confidence, class, and mask coefficients
public struct FurnitureFitDetection: Equatable {
    public let x: Float       // Center x (normalized 0-1 or pixel coordinates)
    public let y: Float       // Center y
    public let w: Float       // Width
    public let h: Float       // Height
    public let confidence: Float
    public let classIdx: Int
    public let coeffs: [Float]

    public init(x: Float, y: Float, w: Float, h: Float, confidence: Float, classIdx: Int, coeffs: [Float] = []) {
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence
        self.classIdx = classIdx
        self.coeffs = coeffs
    }

    /// Convert to CGRect (x,y is center, w,h is size)
    public var boundingBox: CGRect {
        return CGRect(
            x: CGFloat(x - w * 0.5),
            y: CGFloat(y - h * 0.5),
            width: CGFloat(w),
            height: CGFloat(h)
        )
    }

    public static func == (lhs: FurnitureFitDetection, rhs: FurnitureFitDetection) -> Bool {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.w == rhs.w && lhs.h == rhs.h &&
               lhs.confidence == rhs.confidence && lhs.classIdx == rhs.classIdx
    }
}

// MARK: - IoU Calculator
/// Utility class for Intersection over Union calculations
public struct FurnitureFitIoU {

    /// Calculate IoU between two detections (center format: x,y is center, w,h is size)
    public static func calculate(_ a: FurnitureFitDetection, _ b: FurnitureFitDetection) -> Float {
        let ax1 = a.x - a.w * 0.5, ax2 = a.x + a.w * 0.5
        let ay1 = a.y - a.h * 0.5, ay2 = a.y + a.h * 0.5
        let bx1 = b.x - b.w * 0.5, bx2 = b.x + b.w * 0.5
        let by1 = b.y - b.h * 0.5, by2 = b.y + b.h * 0.5

        let ix = max(Float(0), min(ax2, bx2) - max(ax1, bx1))
        let iy = max(Float(0), min(ay2, by2) - max(ay1, by1))
        let intersection = ix * iy
        let union = a.w * a.h + b.w * b.h - intersection
        return union > 0 ? intersection / union : 0.0
    }

    /// Calculate IoU between two CGRects
    public static func calculate(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        if intersection.isNull || intersection.isEmpty {
            return 0.0
        }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - intersectionArea
        return unionArea > 0 ? intersectionArea / unionArea : 0.0
    }
}

// MARK: - NMS (Non-Maximum Suppression)
/// Utility class for Non-Maximum Suppression operations
public struct FurnitureFitNMS {

    private static func applySortedDescending(
        detections: [FurnitureFitDetection],
        iouThreshold: Float
    ) -> [FurnitureFitDetection] {
        var kept: [FurnitureFitDetection] = []
        kept.reserveCapacity(detections.count)
        var suppressed = [Bool](repeating: false, count: detections.count)

        for currentIndex in 0..<detections.count {
            if suppressed[currentIndex] { continue }
            let current = detections[currentIndex]
            kept.append(current)

            if currentIndex + 1 >= detections.count { continue }
            for candidateIndex in (currentIndex + 1)..<detections.count {
                if suppressed[candidateIndex] { continue }
                if current.classIdx != detections[candidateIndex].classIdx { continue }
                if FurnitureFitIoU.calculate(current, detections[candidateIndex]) > iouThreshold {
                    suppressed[candidateIndex] = true
                }
            }
        }

        return kept
    }

    /// Apply NMS to a list of detections
    /// - Parameters:
    ///   - detections: List of detections to filter
    ///   - iouThreshold: IoU threshold above which overlapping boxes are suppressed
    /// - Returns: Filtered list of detections
    public static func apply(detections: [FurnitureFitDetection], iouThreshold: Float) -> [FurnitureFitDetection] {
        guard !detections.isEmpty else { return [] }

        let sorted = detections.sorted { $0.confidence > $1.confidence }
        return applySortedDescending(detections: sorted, iouThreshold: iouThreshold)
    }

    public static func applySortedByConfidence(detections: [FurnitureFitDetection], iouThreshold: Float) -> [FurnitureFitDetection] {
        guard !detections.isEmpty else { return [] }
        return applySortedDescending(detections: detections, iouThreshold: iouThreshold)
    }

    /// Apply NMS using CGRect boxes and scores
    /// - Parameters:
    ///   - boxes: List of bounding boxes
    ///   - scores: Confidence scores for each box
    ///   - iouThreshold: IoU threshold
    /// - Returns: Indices of kept boxes
    public static func apply(boxes: [CGRect], scores: [Float], iouThreshold: Float) -> [Int] {
        guard boxes.count == scores.count, !boxes.isEmpty else { return [] }

        let indices = scores.enumerated()
            .sorted { $0.element > $1.element }
            .map { $0.offset }

        var kept: [Int] = []
        kept.reserveCapacity(indices.count)
        var suppressed = [Bool](repeating: false, count: indices.count)

        for currentPosition in 0..<indices.count {
            if suppressed[currentPosition] { continue }
            let currentIdx = indices[currentPosition]
            kept.append(currentIdx)

            if currentPosition + 1 >= indices.count { continue }
            for nextPosition in (currentPosition + 1)..<indices.count {
                if suppressed[nextPosition] { continue }
                let nextIdx = indices[nextPosition]
                let iou = FurnitureFitIoU.calculate(boxes[currentIdx], boxes[nextIdx])
                if iou > CGFloat(iouThreshold) {
                    suppressed[nextPosition] = true
                }
            }
        }

        return kept
    }
}

// MARK: - Confidence Filter
/// Utility for filtering detections by confidence threshold
public struct FurnitureFitFilter {

    /// Filter detections by confidence threshold
    public static func byConfidence(_ detections: [FurnitureFitDetection], threshold: Float) -> [FurnitureFitDetection] {
        return detections.filter { $0.confidence >= threshold }
    }

    /// Filter detections by class blacklist
    public static func excludingClasses(_ detections: [FurnitureFitDetection], blacklist: Set<Int>) -> [FurnitureFitDetection] {
        return detections.filter { !blacklist.contains($0.classIdx) }
    }

    /// Combined filter: confidence + class blacklist
    public static func apply(
        detections: [FurnitureFitDetection],
        confidenceThreshold: Float,
        classBlacklist: Set<Int> = []
    ) -> [FurnitureFitDetection] {
        return detections.filter { detection in
            detection.confidence >= confidenceThreshold && !classBlacklist.contains(detection.classIdx)
        }
    }
}

// MARK: - YOLO letterbox fill (Ultralytics)
/// Letterbox padding must match Ultralytics default RGB **(114, 114, 114)** (opaque BGRA on device).
/// `scripts/probe_yoloe26_coreml.py` / `visualize_yoloe26_coreml.py` use the same value; using 128×4 gray skews Core ML confidences vs PyTorch export.
public enum YoloUltralyticsLetterboxFill {
    public static func fillOpaqueBGRA114(dstBase: UnsafeMutableRawPointer, totalByteCount: Int) {
        precondition(totalByteCount >= 0 && totalByteCount % 4 == 0)
        guard totalByteCount > 0 else { return }
        var pattern = (UInt8(114), UInt8(114), UInt8(114), UInt8(255))
        withUnsafePointer(to: &pattern) {
            memset_pattern4(dstBase, UnsafeRawPointer($0), totalByteCount)
        }
    }

    public static func fillOpaqueBGRA114LetterboxStrips(
        dstBase: UnsafeMutableRawPointer,
        width: Int,
        height: Int,
        bytesPerRow: Int,
        padX: Int,
        padY: Int,
        scaledWidth: Int,
        scaledHeight: Int
    ) {
        guard width > 0, height > 0, bytesPerRow > 0 else { return }

        var pattern = (UInt8(114), UInt8(114), UInt8(114), UInt8(255))
        let clampedPadX = max(0, min(width, padX))
        let clampedPadY = max(0, min(height, padY))
        let scaledRight = max(clampedPadX, min(width, clampedPadX + scaledWidth))
        let scaledBottom = max(clampedPadY, min(height, clampedPadY + scaledHeight))
        let fullRowBytes = width * 4
        let leftPadBytes = clampedPadX * 4
        let rightPadBytes = max(0, width - scaledRight) * 4

        withUnsafePointer(to: &pattern) { patternPointer in
            let rawPattern = UnsafeRawPointer(patternPointer)
            for y in 0..<height {
                let rowBase = dstBase.advanced(by: y * bytesPerRow)
                if y < clampedPadY || y >= scaledBottom {
                    memset_pattern4(rowBase, rawPattern, fullRowBytes)
                } else {
                    if leftPadBytes > 0 {
                        memset_pattern4(rowBase, rawPattern, leftPadBytes)
                    }
                    if rightPadBytes > 0 {
                        memset_pattern4(rowBase.advanced(by: scaledRight * 4), rawPattern, rightPadBytes)
                    }
                }
            }
        }
    }
}
