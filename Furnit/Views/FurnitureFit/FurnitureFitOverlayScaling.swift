// FurnitureFitOverlayScaling.swift
// Overlay scale fallback, presentation mode, and final transform computation for Furniture Fit.

import CoreGraphics
import Foundation

enum FurnitureFitOverlayPresentationMode {
    case deferredCentered
    case measuredPlacement
}

struct FurnitureFitOverlayPresentationUpdate {
    let mode: FurnitureFitOverlayPresentationMode
    let stableMeasurementFrameCount: Int
    let lastStableHeightMeters: Float?
    let lastStableScale: CGFloat?
}

struct FurnitureFitOverlayTransformResult {
    let transform: CGAffineTransform
    let loggedOverlayScale: CGFloat
    let assistedLabel: String?
    let logMessage: String?
}

enum FurnitureFitOverlayScaling {
    static func resolvedRoomScale(
        currentAutoScaleFromRoom: CGFloat,
        currentAutoScaleFromAR: CGFloat,
        arAssistedSizingEnabled: Bool,
        hasARKitAssistedSizingPayload: Bool,
        arAssistedScaleValid: Bool,
        normalizedARFurnitureHeightMeters: Float?,
        allowRoomProportionFallback: Bool,
        shouldFreezeAutomaticOverlaySizing: Bool,
        primaryBboxInView: CGRect,
        bounds: CGRect,
        primaryBx1: Int,
        primaryBy1: Int,
        primaryBx2: Int,
        primaryBy2: Int,
        imageWidth: Int,
        imageHeight: Int
    ) -> CGFloat {
        if shouldFreezeAutomaticOverlaySizing {
            return currentAutoScaleFromRoom
        }

        let arSizingReady =
            arAssistedSizingEnabled &&
            hasARKitAssistedSizingPayload &&
            arAssistedScaleValid &&
            normalizedARFurnitureHeightMeters != nil &&
            currentAutoScaleFromAR.isFinite &&
            currentAutoScaleFromAR > 0

        guard !arSizingReady else {
            return 1.0
        }

        guard allowRoomProportionFallback else {
            return 1.0
        }

        guard imageWidth > 0, imageHeight > 0,
              bounds.width > 1, bounds.height > 1,
              primaryBboxInView.width > 1, primaryBboxInView.height > 1 else {
            return 1.0
        }

        let targetWidthFraction = CGFloat(max(1, primaryBx2 - primaryBx1)) / CGFloat(imageWidth)
        let targetHeightFraction = CGFloat(max(1, primaryBy2 - primaryBy1)) / CGFloat(imageHeight)
        let currentWidthFraction = primaryBboxInView.width / bounds.width
        let currentHeightFraction = primaryBboxInView.height / bounds.height

        var scaleCandidates: [CGFloat] = []
        if currentWidthFraction > 0.0001, targetWidthFraction.isFinite {
            let widthScale = targetWidthFraction / currentWidthFraction
            if widthScale.isFinite, widthScale > 0 {
                scaleCandidates.append(widthScale)
            }
        }
        if currentHeightFraction > 0.0001, targetHeightFraction.isFinite {
            let heightScale = targetHeightFraction / currentHeightFraction
            if heightScale.isFinite, heightScale > 0 {
                scaleCandidates.append(heightScale)
            }
        }

        guard !scaleCandidates.isEmpty else {
            return 1.0
        }

        let proportionalScale = scaleCandidates.reduce(0, +) / CGFloat(scaleCandidates.count)
        return min(max(proportionalScale, 0.25), 2.5)
    }

    static func updatedPresentation(
        currentMode: FurnitureFitOverlayPresentationMode,
        currentStableMeasurementFrameCount: Int,
        currentLastStableHeightMeters: Float?,
        currentLastStableScale: CGFloat?,
        primaryClassChanged: Bool,
        shouldFreezeAutomaticOverlaySizing: Bool,
        arSizingReady: Bool,
        normalizedARFurnitureHeightMeters: Float?,
        autoScaleFromAR: CGFloat,
        requiredStableOverlayMeasurementFrames: Int,
        maxStableOverlayHeightDriftFraction: Float,
        maxStableOverlayScaleDrift: CGFloat
    ) -> FurnitureFitOverlayPresentationUpdate {
        if shouldFreezeAutomaticOverlaySizing {
            return FurnitureFitOverlayPresentationUpdate(
                mode: currentMode,
                stableMeasurementFrameCount: currentStableMeasurementFrameCount,
                lastStableHeightMeters: currentLastStableHeightMeters,
                lastStableScale: currentLastStableScale
            )
        }

        var mode = currentMode
        var stableMeasurementFrameCount = currentStableMeasurementFrameCount
        var lastStableHeightMeters = currentLastStableHeightMeters
        var lastStableScale = currentLastStableScale

        if primaryClassChanged {
            mode = .deferredCentered
            stableMeasurementFrameCount = 0
            lastStableHeightMeters = nil
            lastStableScale = nil
        }

        if arSizingReady, let arHeight = normalizedARFurnitureHeightMeters {
            let scaleDrift = lastStableScale.map { abs(autoScaleFromAR - $0) } ?? 0
            let heightDriftFraction: Float
            if let lastHeight = lastStableHeightMeters, lastHeight > 0 {
                heightDriftFraction = abs(arHeight - lastHeight) / lastHeight
            } else {
                heightDriftFraction = 0
            }

            let isStable = scaleDrift <= maxStableOverlayScaleDrift &&
                heightDriftFraction <= maxStableOverlayHeightDriftFraction

            stableMeasurementFrameCount = isStable ? (stableMeasurementFrameCount + 1) : 1
            lastStableHeightMeters = arHeight
            lastStableScale = autoScaleFromAR
            mode = stableMeasurementFrameCount >= requiredStableOverlayMeasurementFrames
                ? .measuredPlacement
                : .deferredCentered
        } else {
            mode = .deferredCentered
        }

        return FurnitureFitOverlayPresentationUpdate(
            mode: mode,
            stableMeasurementFrameCount: stableMeasurementFrameCount,
            lastStableHeightMeters: lastStableHeightMeters,
            lastStableScale: lastStableScale
        )
    }

    static func resolvedTransform(
        currentLastAssistedLabel: String,
        currentLastCombinedScale: CGFloat,
        autoScaleFromRoom: CGFloat,
        autoScaleFromAR: CGFloat,
        userPinchScale: CGFloat,
        userPanOffset: CGPoint,
        userLockedAssistedOverlayScale: Bool,
        arAssistedSizingEnabled: Bool,
        hasARKitAssistedSizingPayload: Bool,
        arAssistedScaleValid: Bool,
        allowRoomProportionFallback: Bool,
        defaultStaticOverlayScale: CGFloat,
        minCombinedOverlayScale: CGFloat,
        maxCombinedOverlayScale: CGFloat,
        isShowingLiveVideoIdentifications: Bool,
        overlayPresentationMode: FurnitureFitOverlayPresentationMode,
        bounds: CGRect,
        primaryBboxInView: CGRect
    ) -> FurnitureFitOverlayTransformResult {
        let arOn = arAssistedSizingEnabled && hasARKitAssistedSizingPayload && arAssistedScaleValid
        let roomFactor: CGFloat = arOn ? 1.0 : (allowRoomProportionFallback ? autoScaleFromRoom : defaultStaticOverlayScale)
        let assistedScale: CGFloat = arOn ? autoScaleFromAR : 1.0
        let product = roomFactor * assistedScale * userPinchScale
        let clamped = min(max(product, minCombinedOverlayScale), maxCombinedOverlayScale)

        let finalTransform: CGAffineTransform
        if isShowingLiveVideoIdentifications {
            finalTransform = .identity
        } else {
            switch overlayPresentationMode {
            case .deferredCentered:
                let bboxCenter = CGPoint(x: primaryBboxInView.midX, y: primaryBboxInView.midY)
                let viewCenter = CGPoint(x: bounds.midX, y: bounds.midY)
                let autoTX = primaryBboxInView.width > 0 ? (viewCenter.x - bboxCenter.x) : 0
                let autoTY = primaryBboxInView.height > 0 ? (viewCenter.y - bboxCenter.y) : 0
                finalTransform = CGAffineTransform(scaleX: clamped, y: clamped)
                    .concatenating(CGAffineTransform(
                        translationX: autoTX + userPanOffset.x,
                        y: autoTY + userPanOffset.y
                    ))
            case .measuredPlacement:
                finalTransform = CGAffineTransform(scaleX: clamped, y: clamped)
                    .concatenating(CGAffineTransform(
                        translationX: userPanOffset.x,
                        y: userPanOffset.y
                    ))
            }
        }

        let wantAR = arAssistedSizingEnabled && hasARKitAssistedSizingPayload && !userLockedAssistedOverlayScale
        let assistedLabel: String
        if arOn {
            assistedLabel = "AR"
        } else if wantAR {
            assistedLabel = "ROOM_PROP_AR_unavailable"
        } else if abs(autoScaleFromRoom - 1.0) > 0.02 {
            assistedLabel = "ROOM_PROP"
        } else {
            assistedLabel = "STATIC_DEFAULT"
        }

        let jump = currentLastCombinedScale < 0 || abs(clamped - currentLastCombinedScale) > 0.02
        let labelChange = assistedLabel != currentLastAssistedLabel
        guard jump || labelChange else {
            return FurnitureFitOverlayTransformResult(
                transform: finalTransform,
                loggedOverlayScale: isShowingLiveVideoIdentifications ? 1.0 : clamped,
                assistedLabel: nil,
                logMessage: nil
            )
        }

        let loggedOverlayScale = isShowingLiveVideoIdentifications ? CGFloat(1.0) : clamped
        let modeLabel: String
        if isShowingLiveVideoIdentifications {
            modeLabel = "full_video_identifications"
        } else {
            modeLabel = overlayPresentationMode == .deferredCentered ? "centered_pending" : "measured"
        }
        let logMessage =
            "mode=\(modeLabel) assist=\(assistedLabel) " +
            "roomStored=\(String(format: "%.3f", autoScaleFromRoom)) " +
            "roomUsed=\(String(format: "%.3f", roomFactor)) " +
            "ar=\(String(format: "%.3f", autoScaleFromAR)) " +
            "pinch=\(String(format: "%.3f", userPinchScale)) " +
            "→ overlay=\(String(format: "%.3f", loggedOverlayScale)) " +
            "wantAR=\(wantAR) arValid=\(arAssistedScaleValid)"

        return FurnitureFitOverlayTransformResult(
            transform: finalTransform,
            loggedOverlayScale: loggedOverlayScale,
            assistedLabel: assistedLabel,
            logMessage: logMessage
        )
    }
}
