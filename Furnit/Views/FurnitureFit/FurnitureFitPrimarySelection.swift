// FurnitureFitPrimarySelection.swift
// Primary-candidate scoring and auto-primary hysteresis for Furniture Fit.

import Foundation

struct FurnitureFitPrimarySelectionConfig {
    let minimumConfidence: Float
    let preferHighestConfidence: Bool
    let areaShortlistCount: Int
    let persistenceIoUThreshold: Float
    let switchRequiredFrames: Int
    let confidenceSwitchGain: Float
    let confidenceSwitchMargin: Float

    var clampedMinimumConfidence: Float {
        min(max(minimumConfidence, 0.05), 0.99)
    }
}

struct FurnitureFitAutoPrimarySelectionState {
    var stableDetection: FurnitureFitDetection?
    var pendingDetection: FurnitureFitDetection?
    var pendingFrameCount: Int = 0

    mutating func reset() {
        stableDetection = nil
        pendingDetection = nil
        pendingFrameCount = 0
    }
}

enum FurnitureFitPrimarySelection {
    static func selectPrimaryIndex(
        candidates: [FurnitureFitDetection],
        config: FurnitureFitPrimarySelectionConfig
    ) -> Int? {
        let minimumConfidence = config.clampedMinimumConfidence
        if config.preferHighestConfidence {
            return highestConfidencePrimaryIndex(
                candidates: candidates,
                minimumConfidence: minimumConfidence
            )
        }

        let shortlistedIndices = areaShortlistedCandidateIndices(
            candidates: candidates,
            minimumConfidence: minimumConfidence,
            shortlistCount: config.areaShortlistCount
        )
        let shortlistAreaReference = maxArea(
            candidates: candidates,
            indices: shortlistedIndices
        )

        return shortlistedIndices.max { leftIndex, rightIndex in
            let leftDetection = candidates[leftIndex]
            let rightDetection = candidates[rightIndex]
            let leftScore = weightedSelectionScore(
                detection: leftDetection,
                areaNormalizationReference: shortlistAreaReference
            )
            let rightScore = weightedSelectionScore(
                detection: rightDetection,
                areaNormalizationReference: shortlistAreaReference
            )
            if abs(leftScore - rightScore) > 1e-6 {
                return leftScore < rightScore
            }
            if abs(leftDetection.confidence - rightDetection.confidence) > 1e-6 {
                return leftDetection.confidence < rightDetection.confidence
            }
            let leftArea = leftDetection.w * leftDetection.h
            let rightArea = rightDetection.w * rightDetection.h
            if abs(leftArea - rightArea) > 1e-6 {
                return leftArea < rightArea
            }
            return leftIndex > rightIndex
        }
    }

    static func selectStableAutoPrimaryIndex(
        candidates: [FurnitureFitDetection],
        config: FurnitureFitPrimarySelectionConfig,
        state: inout FurnitureFitAutoPrimarySelectionState
    ) -> Int? {
        guard let preferredIndex = selectPrimaryIndex(candidates: candidates, config: config) else {
            state.reset()
            return nil
        }

        let preferredCandidate = candidates[preferredIndex]
        let minimumConfidence = config.clampedMinimumConfidence

        guard let stableReference = state.stableDetection,
              let stableIndex = matchedCandidateIndex(
                reference: stableReference,
                candidates: candidates,
                persistenceIoUThreshold: config.persistenceIoUThreshold
              ) else {
            state.stableDetection = preferredCandidate
            state.pendingDetection = nil
            state.pendingFrameCount = 0
            return preferredIndex
        }

        let stableCandidate = candidates[stableIndex]
        guard stableCandidate.confidence >= minimumConfidence else {
            state.stableDetection = preferredCandidate
            state.pendingDetection = nil
            state.pendingFrameCount = 0
            return preferredIndex
        }

        if !config.preferHighestConfidence {
            let shortlistedIndices = Set(
                areaShortlistedCandidateIndices(
                    candidates: candidates,
                    minimumConfidence: minimumConfidence,
                    shortlistCount: config.areaShortlistCount
                )
            )
            guard shortlistedIndices.contains(stableIndex) else {
                state.stableDetection = preferredCandidate
                state.pendingDetection = nil
                state.pendingFrameCount = 0
                return preferredIndex
            }
        }

        if stableIndex == preferredIndex {
            state.stableDetection = stableCandidate
            state.pendingDetection = nil
            state.pendingFrameCount = 0
            return stableIndex
        }

        let shortlistAreaReference: Float
        if config.preferHighestConfidence {
            shortlistAreaReference = 0
        } else {
            let shortlistedIndices = Set(
                areaShortlistedCandidateIndices(
                    candidates: candidates,
                    minimumConfidence: minimumConfidence,
                    shortlistCount: config.areaShortlistCount
                )
            )
            shortlistAreaReference = maxArea(
                candidates: candidates,
                indices: Array(shortlistedIndices)
            )
        }

        let stableScore = config.preferHighestConfidence
            ? stableCandidate.confidence
            : weightedSelectionScore(
                detection: stableCandidate,
                areaNormalizationReference: shortlistAreaReference
            )
        let preferredScore = config.preferHighestConfidence
            ? preferredCandidate.confidence
            : weightedSelectionScore(
                detection: preferredCandidate,
                areaNormalizationReference: shortlistAreaReference
            )
        let switchThreshold = max(
            stableScore * config.confidenceSwitchGain,
            stableScore + config.confidenceSwitchMargin
        )

        guard preferredScore >= switchThreshold else {
            state.stableDetection = stableCandidate
            state.pendingDetection = nil
            state.pendingFrameCount = 0
            return stableIndex
        }

        if let pendingCandidate = state.pendingDetection,
           isSameTrack(
                preferredCandidate,
                pendingCandidate,
                persistenceIoUThreshold: config.persistenceIoUThreshold
           ) {
            state.pendingFrameCount += 1
        } else {
            state.pendingDetection = preferredCandidate
            state.pendingFrameCount = 1
        }

        if state.pendingFrameCount >= config.switchRequiredFrames {
            state.stableDetection = preferredCandidate
            state.pendingDetection = nil
            state.pendingFrameCount = 0
            return preferredIndex
        }

        state.stableDetection = stableCandidate
        return stableIndex
    }

    private static func areaShortlistedCandidateIndices(
        candidates: [FurnitureFitDetection],
        minimumConfidence: Float,
        shortlistCount: Int
    ) -> [Int] {
        Array(
            candidates.enumerated()
                .filter { _, detection in detection.confidence >= minimumConfidence }
                .sorted { leftEntry, rightEntry in
                    let leftArea = leftEntry.element.w * leftEntry.element.h
                    let rightArea = rightEntry.element.w * rightEntry.element.h
                    if abs(leftArea - rightArea) > 1e-6 {
                        return leftArea > rightArea
                    }
                    if abs(leftEntry.element.confidence - rightEntry.element.confidence) > 1e-6 {
                        return leftEntry.element.confidence > rightEntry.element.confidence
                    }
                    return leftEntry.offset < rightEntry.offset
                }
                .prefix(max(0, shortlistCount))
                .map(\.offset)
        )
    }

    private static func highestConfidencePrimaryIndex(
        candidates: [FurnitureFitDetection],
        minimumConfidence: Float
    ) -> Int? {
        var bestIndex: Int?
        var bestConfidence: Float = -1
        var bestAreaAtBestConfidence: Float = 0

        for (index, detection) in candidates.enumerated() {
            guard detection.confidence >= minimumConfidence else { continue }
            let area = detection.w * detection.h
            if detection.confidence > bestConfidence {
                bestConfidence = detection.confidence
                bestAreaAtBestConfidence = area
                bestIndex = index
            } else if abs(detection.confidence - bestConfidence) <= 1e-6, area > bestAreaAtBestConfidence {
                bestAreaAtBestConfidence = area
                bestIndex = index
            }
        }

        return bestIndex
    }

    private static func weightedSelectionScore(
        detection: FurnitureFitDetection,
        areaNormalizationReference: Float
    ) -> Float {
        let normalizedArea = areaNormalizationReference > 1e-6
            ? (detection.w * detection.h) / areaNormalizationReference
            : 0
        return 0.5 * detection.confidence + 0.5 * normalizedArea
    }

    private static func matchedCandidateIndex(
        reference: FurnitureFitDetection,
        candidates: [FurnitureFitDetection],
        persistenceIoUThreshold: Float
    ) -> Int? {
        var bestIndex: Int?
        var bestIoU: Float = 0

        for (index, candidate) in candidates.enumerated() {
            guard candidate.classIdx == reference.classIdx else { continue }
            let intersectionOverUnion = FurnitureFitIoU.calculate(candidate, reference)
            if intersectionOverUnion > bestIoU {
                bestIoU = intersectionOverUnion
                bestIndex = index
            }
        }

        guard let bestIndex, bestIoU >= persistenceIoUThreshold else { return nil }
        return bestIndex
    }

    private static func isSameTrack(
        _ leftDetection: FurnitureFitDetection,
        _ rightDetection: FurnitureFitDetection,
        persistenceIoUThreshold: Float
    ) -> Bool {
        leftDetection.classIdx == rightDetection.classIdx
            && FurnitureFitIoU.calculate(leftDetection, rightDetection) >= persistenceIoUThreshold
    }

    private static func maxArea(
        candidates: [FurnitureFitDetection],
        indices: [Int]
    ) -> Float {
        indices
            .map { candidates[$0].w * candidates[$0].h }
            .max() ?? 0
    }
}
