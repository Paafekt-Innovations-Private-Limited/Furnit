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

    /// Apply NMS to a list of detections
    /// - Parameters:
    ///   - detections: List of detections to filter
    ///   - iouThreshold: IoU threshold above which overlapping boxes are suppressed
    /// - Returns: Filtered list of detections
    public static func apply(detections: [FurnitureFitDetection], iouThreshold: Float) -> [FurnitureFitDetection] {
        guard !detections.isEmpty else { return [] }

        // Sort by confidence descending
        var sorted = detections.sorted { $0.confidence > $1.confidence }
        var kept: [FurnitureFitDetection] = []

        while !sorted.isEmpty {
            let current = sorted.removeFirst()
            kept.append(current)

            // Remove all boxes with IoU > threshold
            sorted.removeAll { candidate in
                FurnitureFitIoU.calculate(current, candidate) > iouThreshold
            }
        }

        return kept
    }

    /// Apply NMS using CGRect boxes and scores
    /// - Parameters:
    ///   - boxes: List of bounding boxes
    ///   - scores: Confidence scores for each box
    ///   - iouThreshold: IoU threshold
    /// - Returns: Indices of kept boxes
    public static func apply(boxes: [CGRect], scores: [Float], iouThreshold: Float) -> [Int] {
        guard boxes.count == scores.count, !boxes.isEmpty else { return [] }

        // Sort indices by score descending
        var indices = scores.enumerated()
            .sorted { $0.element > $1.element }
            .map { $0.offset }

        var kept: [Int] = []

        while !indices.isEmpty {
            let currentIdx = indices.removeFirst()
            kept.append(currentIdx)

            // Remove indices with high IoU overlap
            indices.removeAll { nextIdx in
                let iou = FurnitureFitIoU.calculate(boxes[currentIdx], boxes[nextIdx])
                return iou > CGFloat(iouThreshold)
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

// MARK: - Mask Utilities
/// Utility for mask operations
public struct FurnitureFitMask {

    /// Check if mask has any positive pixels
    public static func hasContent(_ mask: [UInt8]) -> Bool {
        return mask.contains { $0 > 0 }
    }

    /// Count positive pixels in mask
    public static func positivePixelCount(_ mask: [UInt8]) -> Int {
        return mask.filter { $0 > 0 }.count
    }

    /// Calculate mask coverage ratio (positive pixels / total pixels)
    public static func coverage(_ mask: [UInt8]) -> Float {
        guard !mask.isEmpty else { return 0.0 }
        let positive = mask.filter { $0 > 0 }.count
        return Float(positive) / Float(mask.count)
    }

    /// Threshold a float mask to UInt8 (0 or 255)
    public static func threshold(_ floatMask: [Float], threshold: Float = 0.0) -> [UInt8] {
        return floatMask.map { $0 > threshold ? UInt8(255) : UInt8(0) }
    }
}

// MARK: - CGRect Extension for Area
// Note: CGRect.area is already defined in FurnitureFitView.swift

// MARK: - Rotation Decision Utility
/// Utility for determining whether to rotate segmentation output for display
public struct FurnitureFitRotation {

    /// Determines whether the segmentation output should be rotated for portrait display
    /// - Parameters:
    ///   - isLandscapeBuffer: Whether the camera buffer is in landscape orientation (width > height)
    ///   - isLandscapeRoom: Whether the room is naturally in landscape orientation
    /// - Returns: True if rotation is needed for portrait display
    public static func shouldRotateForPortrait(isLandscapeBuffer: Bool, isLandscapeRoom: Bool) -> Bool {
        // Only rotate if:
        // 1. The buffer is landscape (camera captured horizontally)
        // 2. AND the room is NOT naturally landscape (would need rotation for portrait UI)
        return isLandscapeBuffer && !isLandscapeRoom
    }

    /// Determines the rotation direction based on device orientation
    /// - Parameter deviceOrientation: The current device orientation (UIDeviceOrientation.rawValue)
    /// - Returns: True for clockwise rotation, false for counter-clockwise
    public static func rotateClockwise(deviceOrientationRawValue: Int) -> Bool {
        // UIDeviceOrientation.landscapeLeft.rawValue == 4
        return deviceOrientationRawValue == 4
    }

    /// Check if buffer dimensions indicate landscape orientation
    /// - Parameters:
    ///   - width: Buffer width in pixels
    ///   - height: Buffer height in pixels
    /// - Returns: True if width > height (landscape)
    public static func isLandscape(width: Int, height: Int) -> Bool {
        return width > height
    }
}

// MARK: - BBox Intersection Utility
/// Utility for bounding box intersection checks used in multi-candidate filtering
public struct FurnitureFitBBox {

    /// Check if two bounding boxes intersect
    /// - Parameters:
    ///   - box1: First bounding box (x1, y1, x2, y2)
    ///   - box2: Second bounding box (x1, y1, x2, y2)
    /// - Returns: True if the boxes intersect
    public static func intersects(
        _ box1: (x1: Float, y1: Float, x2: Float, y2: Float),
        _ box2: (x1: Float, y1: Float, x2: Float, y2: Float)
    ) -> Bool {
        // No intersection if one box is completely to the left, right, above, or below the other
        let noIntersection = box1.x2 < box2.x1 || box1.x1 > box2.x2 || box1.y2 < box2.y1 || box1.y1 > box2.y2
        return !noIntersection
    }

    /// Check if candidate box encompasses primary box (background detection check)
    /// A candidate encompasses primary if it's larger and fully contains the primary
    /// - Parameters:
    ///   - candidate: The candidate detection bounding box
    ///   - primary: The primary detection bounding box
    ///   - marginRatio: How much larger the candidate must be (default 1.1 = 10% larger)
    /// - Returns: True if candidate likely encompasses primary (potential background)
    public static func encompasses(
        candidate: (x1: Float, y1: Float, x2: Float, y2: Float),
        primary: (x1: Float, y1: Float, x2: Float, y2: Float),
        marginRatio: Float = 1.1
    ) -> Bool {
        let candidateW = candidate.x2 - candidate.x1
        let candidateH = candidate.y2 - candidate.y1
        let primaryW = primary.x2 - primary.x1
        let primaryH = primary.y2 - primary.y1

        // Check if candidate is larger than primary
        let isLarger = candidateW > primaryW * marginRatio && candidateH > primaryH * marginRatio

        // Check if candidate contains primary
        let contains = candidate.x1 <= primary.x1 && candidate.y1 <= primary.y1 &&
                      candidate.x2 >= primary.x2 && candidate.y2 >= primary.y2

        return isLarger && contains
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
}
