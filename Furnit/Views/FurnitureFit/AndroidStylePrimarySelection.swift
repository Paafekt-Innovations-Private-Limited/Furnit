// MARK: - Android-style primary selection (new blue icon path)
//
// Mirrors Android ONNX path: after NMS, primary = highest-confidence detection
// (first when sorted by confidence descending). Used only when the new blue icon
// is active; brain icon keeps the existing composite score (confidence × area × center).

import Foundation

enum AndroidStylePrimarySelection {

    /// Returns the index of the primary detection using Android rule:
    /// sort by confidence descending → primary = first (highest confidence).
    /// - Parameter candidates: Detections after NMS (same as Android keepDets).
    /// - Returns: Index of the detection with highest confidence, or nil if empty.
    static func primaryIndex(candidates: [FurnitureFitDetection]) -> Int? {
        guard !candidates.isEmpty else { return nil }
        return candidates.enumerated()
            .max(by: { $0.element.confidence < $1.element.confidence })?
            .offset
    }
}
