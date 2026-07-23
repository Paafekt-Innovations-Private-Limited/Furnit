// FurnitureFitTapSelection.swift
// Thread-safe tap mask snapshots and candidate selection for Furniture Fit bbox taps.

import CoreGraphics
import Foundation

struct FurnitureFitTapMaskSnapshot {
    let planes: [Float]
    let protoWidth: Int
    let protoHeight: Int
    let modelSide: Int
    let imageWidth: Int
    let imageHeight: Int
    let usesLetterbox: Bool

    var hasUsableState: Bool {
        !planes.isEmpty &&
        protoWidth > 0 &&
        protoHeight > 0 &&
        modelSide > 0 &&
        imageWidth > 0 &&
        imageHeight > 0
    }
}

final class FurnitureFitTapMaskState {
    private let lock = NSLock()
    private var latestSnapshot = FurnitureFitTapMaskSnapshot(
        planes: [],
        protoWidth: 0,
        protoHeight: 0,
        modelSide: 0,
        imageWidth: 0,
        imageHeight: 0,
        usesLetterbox: false
    )

    func update(
        planes: [Float],
        protoWidth: Int,
        protoHeight: Int,
        modelSide: Int,
        imageWidth: Int,
        imageHeight: Int,
        usesLetterbox: Bool
    ) {
        lock.lock()
        latestSnapshot = FurnitureFitTapMaskSnapshot(
            planes: planes,
            protoWidth: protoWidth,
            protoHeight: protoHeight,
            modelSide: modelSide,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            usesLetterbox: usesLetterbox
        )
        lock.unlock()
    }

    func clear() {
        lock.lock()
        latestSnapshot = FurnitureFitTapMaskSnapshot(
            planes: [],
            protoWidth: 0,
            protoHeight: 0,
            modelSide: 0,
            imageWidth: 0,
            imageHeight: 0,
            usesLetterbox: false
        )
        lock.unlock()
    }

    func snapshot() -> FurnitureFitTapMaskSnapshot {
        lock.lock()
        let snapshot = latestSnapshot
        lock.unlock()
        return snapshot
    }
}

struct FurnitureFitTapSelectionContext {
    let pointInMaskView: CGPoint
    let maskViewBounds: CGRect
    let candidateRectsInView: [CGRect]
    let candidates: [FurnitureFitDetection]
    let tapMaskSnapshot: FurnitureFitTapMaskSnapshot
    let isShowingLiveVideoIdentifications: Bool
    let imageContentRectInView: CGRect?

    var tapHitPadding: CGFloat {
        isShowingLiveVideoIdentifications ? 22 : 10
    }
}

enum FurnitureFitTapSelection {
    static func candidateIndex(context: FurnitureFitTapSelectionContext) -> Int? {
        guard !context.candidateRectsInView.isEmpty,
              context.candidateRectsInView.count == context.candidates.count else {
            return nil
        }

        let paddedMatches = context.candidateRectsInView.enumerated().filter { _, rect in
            rect.insetBy(dx: -context.tapHitPadding, dy: -context.tapHitPadding).contains(context.pointInMaskView)
        }
        guard !paddedMatches.isEmpty else { return nil }

        let maskMatches = paddedMatches.compactMap { match -> (offset: Int, rect: CGRect, score: Float)? in
            guard match.offset < context.candidates.count else { return nil }
            guard let maskScore = maskPresenceScore(
                detection: context.candidates[match.offset],
                context: context
            ) else {
                return nil
            }
            return (offset: match.offset, rect: match.element, score: maskScore)
        }

        let shouldRequireMaskHit = context.tapMaskSnapshot.hasUsableState && !context.isShowingLiveVideoIdentifications
        if shouldRequireMaskHit, maskMatches.isEmpty {
            return nil
        }

        let candidatesForSelection = maskMatches.isEmpty
            ? paddedMatches.map { (offset: $0.offset, rect: $0.element, score: Float.leastNormalMagnitude) }
            : maskMatches

        return candidatesForSelection.max { leftMatch, rightMatch in
            if abs(leftMatch.score - rightMatch.score) > 0.0001 {
                return leftMatch.score < rightMatch.score
            }
            let leftArea = leftMatch.rect.width * leftMatch.rect.height
            let rightArea = rightMatch.rect.width * rightMatch.rect.height
            if abs(leftArea - rightArea) > 1 {
                return leftArea > rightArea
            }
            return context.candidates[leftMatch.offset].confidence < context.candidates[rightMatch.offset].confidence
        }?.offset
    }

    private static func maskPresenceScore(
        detection: FurnitureFitDetection,
        context: FurnitureFitTapSelectionContext
    ) -> Float? {
        let snapshot = context.tapMaskSnapshot
        guard snapshot.hasUsableState,
              detection.coeffs.count >= 32,
              context.maskViewBounds.width > 0,
              context.maskViewBounds.height > 0 else {
            return nil
        }

        let imageRect = context.imageContentRectInView ?? context.maskViewBounds
        guard imageRect.width > 0,
              imageRect.height > 0,
              imageRect.contains(context.pointInMaskView) else {
            return nil
        }

        let imageX = Float((context.pointInMaskView.x - imageRect.minX) / imageRect.width) * Float(snapshot.imageWidth)
        let imageY = Float((context.pointInMaskView.y - imageRect.minY) / imageRect.height) * Float(snapshot.imageHeight)

        let bboxHalfWidth = detection.w * 0.5
        let bboxHalfHeight = detection.h * 0.5
        let bboxMinX = detection.x - bboxHalfWidth
        let bboxMaxX = detection.x + bboxHalfWidth
        let bboxMinY = detection.y - bboxHalfHeight
        let bboxMaxY = detection.y + bboxHalfHeight
        guard imageX >= bboxMinX,
              imageX <= bboxMaxX,
              imageY >= bboxMinY,
              imageY <= bboxMaxY else {
            return nil
        }

        let modelX: Float
        let modelY: Float
        if snapshot.usesLetterbox {
            let gain = min(
                Float(snapshot.modelSide) / Float(snapshot.imageWidth),
                Float(snapshot.modelSide) / Float(snapshot.imageHeight)
            )
            let padX = (Float(snapshot.modelSide) - Float(snapshot.imageWidth) * gain) * 0.5
            let padY = (Float(snapshot.modelSide) - Float(snapshot.imageHeight) * gain) * 0.5
            modelX = imageX * gain + padX
            modelY = imageY * gain + padY
        } else {
            modelX = imageX * Float(snapshot.modelSide) / Float(snapshot.imageWidth)
            modelY = imageY * Float(snapshot.modelSide) / Float(snapshot.imageHeight)
        }

        let protoX = min(
            snapshot.protoWidth - 1,
            max(0, Int(floor(modelX * Float(snapshot.protoWidth) / Float(snapshot.modelSide))))
        )
        let protoY = min(
            snapshot.protoHeight - 1,
            max(0, Int(floor(modelY * Float(snapshot.protoHeight) / Float(snapshot.modelSide))))
        )
        let prototypePixelCount = snapshot.protoWidth * snapshot.protoHeight
        let protoPixelIndex = protoY * snapshot.protoWidth + protoX
        guard snapshot.planes.count >= 32 * prototypePixelCount else { return nil }

        var dotProductSum: Float = 0
        for coefficientIndex in 0..<32 {
            let planeIndex = coefficientIndex * prototypePixelCount + protoPixelIndex
            dotProductSum += detection.coeffs[coefficientIndex] * snapshot.planes[planeIndex]
        }
        return dotProductSum > 0 ? dotProductSum : nil
    }
}
