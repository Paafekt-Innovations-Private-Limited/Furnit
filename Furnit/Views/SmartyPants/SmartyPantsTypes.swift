// SmartyPantsTypes.swift
// Basic types, enums, and data structures for SmartyPants detection

import Foundation

// MARK: - Edge Fill Mode (user selectable)
enum EdgeFillMode {
    case furniMaterial  // Preserve fine edges using scanline - for chairs, tables, solid items
    case clothBased     // Solid fill using hull - for beds, sofas, fabric items with gaps
    case chairType      // Morphological close - fills small gaps, preserves general shape
}

// MARK: - Detection Struct
struct DetectionSmarty {
    let x: Float
    let y: Float
    let width: Float
    let height: Float
    let confidence: Float
    let classIdx: Int
    let className: String
    let maskCoeffs: [Float]
    var trackId: Int? = nil  // Assigned by tracker
}

// MARK: - Track State for SORT-style Tracking
struct Track {
    let id: Int
    var x: Float           // Center x
    var y: Float           // Center y
    var width: Float
    var height: Float
    var classIdx: Int
    var className: String
    var confidence: Float
    var age: Int           // Frames since track was created
    var hitStreak: Int     // Consecutive frames with detection match
    var timeSinceUpdate: Int  // Frames since last matched detection

    // Velocity estimates for simple prediction
    var vx: Float = 0
    var vy: Float = 0

    // Get predicted position
    func predicted() -> (x: Float, y: Float, w: Float, h: Float) {
        return (x + vx, y + vy, width, height)
    }

    // Calculate IoU with a detection
    func iou(with det: DetectionSmarty) -> Float {
        let pred = predicted()

        let aLeft = pred.x - pred.w * 0.5
        let aRight = pred.x + pred.w * 0.5
        let aTop = pred.y - pred.h * 0.5
        let aBottom = pred.y + pred.h * 0.5

        let bLeft = det.x - det.width * 0.5
        let bRight = det.x + det.width * 0.5
        let bTop = det.y - det.height * 0.5
        let bBottom = det.y + det.height * 0.5

        let ix1 = max(aLeft, bLeft)
        let ix2 = min(aRight, bRight)
        let iy1 = max(aTop, bTop)
        let iy2 = min(aBottom, bBottom)

        let iw = max(0, ix2 - ix1)
        let ih = max(0, iy2 - iy1)
        let inter = iw * ih

        let areaA = pred.w * pred.h
        let areaB = det.width * det.height
        let union = areaA + areaB - inter

        guard union > 0 else { return 0 }
        return inter / union
    }

    // Update track with matched detection
    mutating func update(with det: DetectionSmarty) {
        // Update velocity (simple exponential smoothing)
        let alpha: Float = 0.3
        let newVx = det.x - x
        let newVy = det.y - y
        vx = alpha * newVx + (1 - alpha) * vx
        vy = alpha * newVy + (1 - alpha) * vy

        // Update position and size
        x = det.x
        y = det.y
        width = det.width
        height = det.height
        confidence = det.confidence
        classIdx = det.classIdx
        className = det.className

        // Update counters
        age += 1
        hitStreak += 1
        timeSinceUpdate = 0
    }

    // Mark as unmatched this frame
    mutating func markMissed() {
        age += 1
        hitStreak = 0
        timeSinceUpdate += 1

        // Apply velocity prediction
        x += vx
        y += vy
    }
}
