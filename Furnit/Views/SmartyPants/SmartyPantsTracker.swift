// SmartyPantsTracker.swift
// Simple SORT-style Tracker for object tracking across frames

import Foundation

// MARK: - Simple SORT-style Tracker
final class SimpleTracker {
    private var tracks: [Track] = []
    private var nextId: Int = 1

    // Configuration
    let iouThreshold: Float = 0.3      // Minimum IoU to match
    let maxAge: Int = 30               // Max frames to keep unmatched track
    let minHits: Int = 3               // Min hits before track is confirmed

    // Temporal mask smoothing
    private var maskHistory: [Int: [[Float]]] = [:]  // trackId -> recent masks
    let maskHistorySize: Int = 5                      // Number of frames to keep
    let temporalSmoothingAlpha: Float = 0.6           // Weight for current frame (0.6 = 60% current, 40% history)

    init() {}

    /// Store mask for a track (for temporal smoothing)
    func storeMask(_ mask: [Float], forTrackId trackId: Int) {
        if maskHistory[trackId] == nil {
            maskHistory[trackId] = []
        }
        maskHistory[trackId]?.append(mask)
        // Keep only recent masks
        if let count = maskHistory[trackId]?.count, count > maskHistorySize {
            maskHistory[trackId]?.removeFirst()
        }
    }

    /// Get temporally stable mask for a track using VOTING approach
    /// A pixel is included only if detected in at least `minVotes` of recent frames
    /// This prevents gradual accumulation while providing frame-to-frame stability
    let minVotesForPixel: Int = 2  // Pixel must appear in at least 2 of last N frames

    func getSmoothedMask(_ currentMask: [Float], forTrackId trackId: Int) -> [Float] {
        guard let history = maskHistory[trackId], !history.isEmpty else {
            return currentMask
        }

        let count = currentMask.count
        var voteCounts = [Int](repeating: 0, count: count)

        // Count votes from historical masks
        for histMask in history {
            for i in 0..<min(count, histMask.count) {
                if histMask[i] > 0 {
                    voteCounts[i] += 1
                }
            }
        }

        // Count vote from current frame
        for i in 0..<count {
            if currentMask[i] > 0 {
                voteCounts[i] += 1
            }
        }

        // Pixel is ON only if it has enough votes
        var resultMask = [Float](repeating: 0, count: count)
        for i in 0..<count {
            if voteCounts[i] >= minVotesForPixel {
                resultMask[i] = 1.0
            }
        }

        return resultMask
    }

    /// Clean up mask history for removed tracks
    private func cleanupMaskHistory() {
        let activeTrackIds = Set(tracks.map { $0.id })
        maskHistory = maskHistory.filter { activeTrackIds.contains($0.key) }
    }

    /// Update tracker with new detections, returns detections with assigned track IDs
    func update(detections: [DetectionSmarty]) -> [DetectionSmarty] {
        // Step 1: Build cost matrix using track IDs (not indices)
        var matchedDetIdx = Set<Int>()
        var matchedTrackIds = Set<Int>()
        var detToTrackId: [Int: Int] = [:]  // detIdx -> trackId

        // Calculate all IoU pairs and store with track ID
        var matches: [(trackId: Int, trackIdx: Int, detIdx: Int, iou: Float)] = []
        for (ti, track) in tracks.enumerated() {
            for (di, det) in detections.enumerated() {
                let iouVal = track.iou(with: det)
                if iouVal >= iouThreshold {
                    matches.append((track.id, ti, di, iouVal))
                }
            }
        }

        // Greedy matching: sort by IoU descending, assign greedily
        matches.sort { $0.iou > $1.iou }

        for match in matches {
            if matchedTrackIds.contains(match.trackId) || matchedDetIdx.contains(match.detIdx) {
                continue
            }
            matchedTrackIds.insert(match.trackId)
            matchedDetIdx.insert(match.detIdx)
            detToTrackId[match.detIdx] = match.trackId

            // Update track with detection (use index here, before any removal)
            tracks[match.trackIdx].update(with: detections[match.detIdx])
        }

        // Step 2: Mark unmatched tracks as missed
        for i in 0..<tracks.count {
            if !matchedTrackIds.contains(tracks[i].id) {
                tracks[i].markMissed()
            }
        }

        // Step 3: Create new tracks for unmatched detections
        for (di, det) in detections.enumerated() {
            if !matchedDetIdx.contains(di) {
                let newTrack = Track(
                    id: nextId,
                    x: det.x,
                    y: det.y,
                    width: det.width,
                    height: det.height,
                    classIdx: det.classIdx,
                    className: det.className,
                    confidence: det.confidence,
                    age: 1,
                    hitStreak: 1,
                    timeSinceUpdate: 0,
                    vx: 0,
                    vy: 0
                )
                tracks.append(newTrack)
                detToTrackId[di] = nextId  // Map new detection to new track
                nextId += 1
            }
        }

        // Step 4: Remove dead tracks
        tracks.removeAll { $0.timeSinceUpdate > maxAge }
        cleanupMaskHistory()  // Remove mask history for deleted tracks

        // Step 5: Build output - detections with track IDs assigned
        // Use track IDs (not indices) to find tracks safely
        var result: [DetectionSmarty] = []

        for (di, det) in detections.enumerated() {
            var detWithId = det

            if let trackId = detToTrackId[di],
               let track = tracks.first(where: { $0.id == trackId }) {
                // Only assign ID if track is confirmed (enough hits)
                if track.hitStreak >= minHits || track.age >= minHits {
                    detWithId.trackId = track.id
                }
            }

            result.append(detWithId)
        }

        return result
    }

    /// Get all confirmed tracks (for visualization)
    func getConfirmedTracks() -> [Track] {
        return tracks.filter { $0.hitStreak >= minHits || $0.age >= minHits }
    }

    /// Reset tracker state
    func reset() {
        tracks.removeAll()
        maskHistory.removeAll()
        nextId = 1
    }
}
