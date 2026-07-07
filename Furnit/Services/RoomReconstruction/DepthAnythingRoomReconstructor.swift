import CoreGraphics
import CoreML
import Foundation
import ImageIO
import ModelIO
import SceneKit
import simd
import UIKit
@preconcurrency import Vision

struct DepthAnythingRoomResult: Sendable {
    let usdzURL: URL
    let vertexCount: Int
    let triangleCount: Int
    let imageWidth: Int
    let imageHeight: Int
    /// Display/saved dimensions from Depth Anything metric depth scaled with a 1.7 m camera-height prior.
    /// Width/height are projected from depth-unprojected spread using GeoCalib focal when available.
    /// Never derived from USDZ mesh, point cloud, or RealityKit bounds.
    let roomWidthMeters: Float
    let roomHeightMeters: Float
    let roomDepthMeters: Float
    let calibrationMetadata: [String: Double]

    /// Compact on-screen diagnostic (Settings → Debug Mode): shows the inputs behind W×H×D
    /// so calibration issues can be read off a screenshot without console access.
    var measurementDebugLine: String? {
        guard let scale = calibrationMetadata["measurementDepthScale"],
              let focal = calibrationMetadata["measurementFocalLengthPx"] else {
            return nil
        }
        let rawHeight = calibrationMetadata["cameraHeightRawM"].map { String(format: "%.2f", $0) } ?? "?"
        let focalSource: String
        switch calibrationMetadata["measurementFocalSourceCode"] {
        case 2: focalSource = "capture"
        case 1: focalSource = "geocalib"
        default: focalSource = "fallback"
        }
        let gravitySource: String
        switch calibrationMetadata["measurementGravitySourceCode"] {
        case 2: gravitySource = "arkit"
        case 1: gravitySource = "geo"
        default: gravitySource = "none"
        }
        let prior = calibrationMetadata["cameraHeightPriorM"] ?? 1.7
        let priorSource: String
        switch calibrationMetadata["cameraHeightPriorSourceCode"] {
        case 2: priorSource = "est"
        case 1: priorSource = "floor"
        default: priorSource = "fixed"
        }
        return String(
            format: "camH %@m · prior %.2fm (%@) · scale %.2f · f %.0fpx (%@) · grav %@",
            rawHeight,
            prior,
            priorSource,
            scale,
            focal,
            focalSource,
            gravitySource
        )
    }

    var photoOrientation: PhotoOrientation {
        if imageWidth == imageHeight { return .square }
        return imageWidth > imageHeight ? .landscape : .portrait
    }

    var summary: String {
        String(
            format: "vertices=%d triangles=%d image=%dx%d dims=%.2fx%.2fx%.2fm usdz=%@",
            vertexCount,
            triangleCount,
            imageWidth,
            imageHeight,
            roomWidthMeters,
            roomHeightMeters,
            roomDepthMeters,
            usdzURL.lastPathComponent
        )
    }
}

final class DepthAnythingRoomReconstructor {
    let model: VNCoreMLModel

    private let modelName: String
    private let pixelStep: Int
    private let depthDiscontinuityThresholdMeters: Float
    private let maxReconstructionImageDimension: Int
    private let outputDirectory: URL?
    private let wallMargin: Float

    /// Matches `scripts/depthanything_measure_room.py --flat-mesh` (photo on a flat plane).
    private static let usesFlatMesh = true
    private static let minimumRoomWidthMeters: Float = 2.0
    private static let fallbackFocal35mmEquivalent: Float = 28.0
    private static let objectBBoxConfidenceThreshold: Float = 0.30
    private static let geoExifFocalMatchRatioRange: ClosedRange<Float> = 0.85...1.15
    private static let depthMetricScaleRange: ClosedRange<Float> = 0.55...1.45
    /// Eye-level anchor for room measurement (matches Python `scripts/*measure_room.py`).
    private static let cameraHeightPriorMeters: Float = 1.70
    /// Python gate: trust raw camera height only inside this range (scripts/structure_box_measure_room.py).
    private static let cameraHeightRawValidRange: ClosedRange<Float> = 0.45...5.0
    /// Bottom band used to estimate camera height above the floor (no M-LSD quad on iOS).
    private static let floorBandStartFraction: Float = 0.78
    /// Skip bottom-right chair clutter when sampling floor height (matches Python quad mask).
    private static let floorChairExcludeU: Float = 0.58
    private static let floorChairExcludeV: Float = 0.55
    private static let plausibleRoomSpanRange: ClosedRange<Float> = 1.2...8.0
    /// Plausible fy/imageHeight for phone photos (vertical FOV roughly 25°–110°).
    /// Width-based gating breaks on tall crops, so gate on the vertical ratio.
    private static let measurementFocalHeightRatioRange: ClosedRange<Float> = 0.35...2.30
    /// If the on-device GeoCalib roll/pitch is implausible for a handheld room photo,
    /// skip gravity leveling rather than corrupt the floor-height estimate.
    private static let maxPlausibleRollRadians: Float = 0.6   // ~34°
    private static let maxPlausiblePitchRadians: Float = 0.9  // ~52°
    /// Ceiling sampling for floor+ceiling anchored room height: candidates come from the top
    /// band of the image and must sit clearly above the camera after leveling.
    private static let ceilingBandRowFraction: Float = 0.18
    private static let minimumCeilingClearanceMeters: Float = 0.3
    /// Plausible indoor ceiling-anchored heights; outside this the vertical spread is kept.
    private static let ceilingAnchoredHeightRange: ClosedRange<Float> = 1.9...4.2

    /// COCO class index → expected physical height in meters for metric depth anchoring.
    private static let objectAnchorHeightMeters: [Int: Float] = [
        56: 1.15, // chair
    ]

    init(
        pixelStep: Int = 4,
        depthDiscontinuityThresholdMeters: Float = 0.15,
        maxReconstructionImageDimension: Int = 1600,
        outputDirectory: URL? = nil,
        wallMargin: Float = 0.05
    ) throws {
        self.pixelStep = max(1, pixelStep)
        self.depthDiscontinuityThresholdMeters = depthDiscontinuityThresholdMeters
        self.maxReconstructionImageDimension = max(256, maxReconstructionImageDimension)
        self.outputDirectory = outputDirectory
        self.wallMargin = min(max(wallMargin, 0), 0.45)

        let loaded = try Self.sharedDepthModel()
        self.modelName = loaded.name
        self.model = loaded.model
    }

    private static func timedStage<T>(_ label: String, _ work: () async throws -> T) async rethrows -> T {
        let start = CFAbsoluteTimeGetCurrent()
        let result = try await work()
        logDebug(String(format: "[DepthAnythingRoom][Timing] %@ %.0f ms", label, (CFAbsoluteTimeGetCurrent() - start) * 1000))
        return result
    }

    private static let depthModelLock = NSLock()
    private static var cachedDepthModel: (model: VNCoreMLModel, name: String)?

    /// A reconstructor is created per room; loading/compiling Depth Anything each time
    /// is wasted seconds, so share one Vision model per process.
    private static func sharedDepthModel() throws -> (model: VNCoreMLModel, name: String) {
        depthModelLock.lock()
        defer { depthModelLock.unlock() }
        if let cachedDepthModel {
            return cachedDepthModel
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let loaded = try loadDepthAnythingModel(configuration: config)
        let vision = try VNCoreMLModel(for: loaded.model)
        let entry = (model: vision, name: loaded.name)
        cachedDepthModel = entry
        return entry
    }

    func reconstruct(image: UIImage) async throws -> URL {
        let result = try await reconstructWithResult(image: image)
        return result.usdzURL
    }

    func reconstructWithResult(
        image: UIImage,
        cameraMetadata: [String: Double]? = nil
    ) async throws -> DepthAnythingRoomResult {
        let pipelineStart = CFAbsoluteTimeGetCurrent()
        let sourcePixelWidth = max(1, Int(ceil(Double(image.size.width * image.scale))))
        let sourcePixelHeight = max(1, Int(ceil(Double(image.size.height * image.scale))))
        let fixedImage = image.fixedOrientation()
        let workingImage = try Self.downsampledImage(fixedImage, maxDimension: maxReconstructionImageDimension)
        let fixedPixelWidth = fixedImage.cgImage?.width ?? sourcePixelWidth
        let fixedPixelHeight = fixedImage.cgImage?.height ?? sourcePixelHeight
        let workingPixelWidth = workingImage.cgImage?.width ?? fixedPixelWidth
        let workingPixelHeight = workingImage.cgImage?.height ?? fixedPixelHeight
        let preprocessEnd = CFAbsoluteTimeGetCurrent()
        // 1–3: one measurement grid — GeoCalib and Depth Anything both use the full working frame.
        // The three model inferences (GeoCalib, Depth Anything, RTMDet) only need the image,
        // so they run concurrently; all cross-model math happens after the joins below.
        async let geoCalibTask = Self.timedStage("geocalib") {
            await GeoCalibCalibrationService.shared.estimateCalibration(image: workingImage)
        }
        async let objectRectTask = Self.timedStage("rtmdet") {
            await Self.detectMeasurementObjectRect(
                image: workingImage,
                imageWidth: workingPixelWidth,
                imageHeight: workingPixelHeight
            )
        }
        let depthMap = try await Self.timedStage("depth_anything") {
            try await self.inferDepth(image: workingImage)
        }
        let geoCalibCalibration = await geoCalibTask
        let objectRect = await objectRectTask
        let inferenceEnd = CFAbsoluteTimeGetCurrent()
        logDebug(String(
            format: "[DepthAnythingRoom][Timing] preprocess %.0f ms · inference_join %.0f ms (parallel geocalib+depth+rtmdet)",
            (preprocessEnd - pipelineStart) * 1000,
            (inferenceEnd - preprocessEnd) * 1000
        ))
        let imageWidth = depthMap.first?.count ?? 0
        let imageHeight = depthMap.count
        guard imageWidth == workingPixelWidth, imageHeight == workingPixelHeight else {
            throw DepthAnythingRoomError.depthImageSizeMismatch
        }
        let calibrationMetadata = Self.mergedCameraMetadata(
            cameraMetadata,
            geoCalibCalibration?.metadata
        )
        let focal = Self.focalPixelDetails(
            image: workingImage,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            cameraMetadata: calibrationMetadata
        )
        var focalPx = focal.fx
        let rawObjectMeasured = Self.measureObjectBBox(
            objectRect: objectRect,
            depthMap: depthMap,
            fx: focalPx,
            fy: focalPx
        )
        let metricCalibration = Self.resolveMetricCalibration(
            geoFocalPx: focalPx,
            cameraMetadata: calibrationMetadata,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            rawObjectMeasurement: rawObjectMeasured
        )
        focalPx = metricCalibration.focalPx
        let measurementFocal = Self.measurementFocalPixels(
            geoCalib: geoCalibCalibration,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            trustedFocal: focal,
            fallbackFx: focalPx,
            fallbackFy: focalPx
        )
        let calibratedDepthMap = Self.scaleDepthMap(depthMap, scale: metricCalibration.depthScale)
        let measurementCalibration = Self.resolveMeasurementCameraHeightScale(
            depthMap: depthMap,
            workingImage: workingImage,
            geoCalib: geoCalibCalibration,
            arkitGravityDown: Self.arkitGravityDownVector(from: calibrationMetadata),
            arkitCameraHeightMeters: calibrationMetadata?["arkitCameraHeightM"].map(Float.init),
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            fx: measurementFocal.fx,
            fy: measurementFocal.fy,
            wallMargin: wallMargin,
            objectRect: objectRect,
            furnitureExcludeBBox: rawObjectMeasured.map {
                (leftX: $0.bboxLeftX, rightX: $0.bboxRightX, topY: $0.bboxTopY, bottomY: $0.bboxBottomY)
            }
        )
        let measurementDepthMap = Self.scaleDepthMap(depthMap, scale: measurementCalibration.depthScale)
        var enrichedCalibrationMetadata = calibrationMetadata ?? [:]
        enrichedCalibrationMetadata["depthMetricScale"] = Double(metricCalibration.depthScale)
        enrichedCalibrationMetadata["depthMetricCalibrationSource"] = Double(metricCalibration.sourceCode)
        enrichedCalibrationMetadata["measurementDepthScale"] = Double(measurementCalibration.depthScale)
        enrichedCalibrationMetadata["measurementCalibrationSourceCode"] = Double(measurementCalibration.sourceCode)
        enrichedCalibrationMetadata["measurementFocalLengthPx"] = Double(measurementFocal.fx)
        enrichedCalibrationMetadata["measurementFocalLengthYPx"] = Double(measurementFocal.fy)
        // 0 = fallback metric focal, 1 = geocalib CNN, 2 = trusted sidecar/EXIF capture intrinsics.
        let measurementFocalSourceCode: Double
        if measurementFocal.source == "geocalib" {
            measurementFocalSourceCode = 1
        } else if measurementFocal.source == "fallback_metric_focal" {
            measurementFocalSourceCode = 0
        } else {
            measurementFocalSourceCode = 2
        }
        enrichedCalibrationMetadata["measurementFocalSourceCode"] = measurementFocalSourceCode
        enrichedCalibrationMetadata["measurementGravitySourceCode"] = Double(measurementCalibration.gravitySourceCode)
        enrichedCalibrationMetadata["cameraHeightPriorSourceCode"] = Double(measurementCalibration.cameraHeightPriorSourceCode)
        enrichedCalibrationMetadata["cameraHeightPriorM"] = Double(measurementCalibration.cameraHeightPriorMeters)
        enrichedCalibrationMetadata["scaleEstimatorConfidence"] = Double(measurementCalibration.scaleEstimatorConfidence)
        if let cameraHeightRaw = measurementCalibration.cameraHeightRawMeters {
            enrichedCalibrationMetadata["cameraHeightRawM"] = Double(cameraHeightRaw)
        }
        enrichedCalibrationMetadata["metricFocalLengthPx"] = Double(focalPx)
        if let exifFocalPx = metricCalibration.exifFocalPx {
            enrichedCalibrationMetadata["exifFocalLengthPx"] = Double(exifFocalPx)
        }
        if let anchorClassIdx = metricCalibration.anchorClassIdx {
            enrichedCalibrationMetadata["depthAnchorClassIdx"] = Double(anchorClassIdx)
        }
        if let anchorExpectedHeight = metricCalibration.anchorExpectedHeightMeters {
            enrichedCalibrationMetadata["depthAnchorExpectedHeightM"] = Double(anchorExpectedHeight)
        }
        if let anchorMeasuredHeight = metricCalibration.anchorMeasuredHeightMeters {
            enrichedCalibrationMetadata["depthAnchorMeasuredHeightM"] = Double(anchorMeasuredHeight)
        }

        let rawStats = Self.depthMapStats(depthMap)
        let stats = Self.depthMapStats(calibratedDepthMap)
        let measurementStats = Self.depthMapStats(measurementDepthMap)
        logDebug(
            "[DepthAnythingRoom][MetricCalib] source=\(metricCalibration.sourceLabel) " +
            "geo_focal_px=\(String(format: "%.1f", metricCalibration.geoFocalPx)) " +
            "exif_focal_px=\(metricCalibration.exifFocalPx.map { String(format: "%.1f", $0) } ?? "nil") " +
            "final_focal_px=\(String(format: "%.1f", focalPx)) " +
            "measurement_focal_px=\(String(format: "%.1f", measurementFocal.fx)) " +
            "measurement_fy_px=\(String(format: "%.1f", measurementFocal.fy)) " +
            "measurement_focal_source=\(measurementFocal.source) " +
            "depth_scale=\(String(format: "%.4f", metricCalibration.depthScale)) " +
            "measurement_scale=\(String(format: "%.4f", measurementCalibration.depthScale)) " +
            "measurement_source=\(measurementCalibration.sourceLabel) " +
            "gravity_source_code=\(measurementCalibration.gravitySourceCode) " +
            "camera_height_prior_m=\(String(format: "%.2f", measurementCalibration.cameraHeightPriorMeters)) " +
            "camera_height_raw_m=\(measurementCalibration.cameraHeightRawMeters.map { String(format: "%.3f", $0) } ?? "nil") " +
            "anchor_cls=\(metricCalibration.anchorClassIdx.map(String.init) ?? "nil") " +
            "anchor_h_expected=\(metricCalibration.anchorExpectedHeightMeters.map { String(format: "%.3f", $0) } ?? "nil") " +
            "anchor_h_raw=\(metricCalibration.anchorMeasuredHeightMeters.map { String(format: "%.3f", $0) } ?? "nil") " +
            "depth_median_raw=\(Self.formatMeters(rawStats.median)) depth_median_cal=\(Self.formatMeters(stats.median)) " +
            "depth_median_measurement=\(Self.formatMeters(measurementStats.median))"
        )

        let wallMeasured = Self.measureWall(
            depthMap: measurementDepthMap,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            fx: measurementFocal.fx,
            fy: measurementFocal.fy,
            wallMargin: wallMargin
        )
        let depthSpreadMeasured = Self.measureDepthSpread(
            depthMap: measurementDepthMap,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            fx: measurementFocal.fx,
            fy: measurementFocal.fy,
            wallMargin: wallMargin,
            levelingRotation: measurementCalibration.levelingRotation,
            cameraHeightPriorMeters: measurementCalibration.cameraHeightPriorMeters
        ) ?? wallMeasured
        // Same detection, calibrated depth — RTMDet does not need to run again.
        let objectMeasured = Self.measureObjectBBox(
            objectRect: objectRect,
            depthMap: calibratedDepthMap,
            fx: focalPx,
            fy: focalPx
        )
        // Depth-unprojected spread is the room estimate; wall rect is frustum-at-center-depth only.
        let measured = Self.sanitizeRoomMeasurement(
            depthSpreadMeasured,
            wallFallback: wallMeasured
        )
        let rect = Self.measureWallSampleRect(imageWidth: imageWidth, imageHeight: imageHeight, wallMargin: wallMargin)
        let rectWidthPixels = max(0, rect.rightX - rect.leftX)
        let rectHeightPixels = max(0, rect.bottomY - rect.topY)
        let wallWidthFormula = "\(rectWidthPixels)px*\(String(format: "%.4f", wallMeasured.depth))m/\(String(format: "%.2f", focalPx))f"
        let wallHeightFormula = "\(rectHeightPixels)px*\(String(format: "%.4f", wallMeasured.depth))m/\(String(format: "%.2f", focalPx))f"
        let objectDimsSummary: String
        if let objectMeasured {
            objectDimsSummary = "W:\(String(format: "%.4f", objectMeasured.width)),H:\(String(format: "%.4f", objectMeasured.height)),D:\(String(format: "%.4f", objectMeasured.depth))"
        } else {
            objectDimsSummary = "nil"
        }
        let rawObjectDimsSummary: String
        if let rawObjectMeasured {
            rawObjectDimsSummary = "W:\(String(format: "%.4f", rawObjectMeasured.width)),H:\(String(format: "%.4f", rawObjectMeasured.height)),D:\(String(format: "%.4f", rawObjectMeasured.depth))"
        } else {
            rawObjectDimsSummary = "nil"
        }
        logDebug(
            "[DepthAnythingRoom][InferenceDims] model=\(modelName) " +
            "source_px=\(sourcePixelWidth)x\(sourcePixelHeight) fixed_px=\(fixedPixelWidth)x\(fixedPixelHeight) " +
            "working_px=\(workingPixelWidth)x\(workingPixelHeight) measurement_grid=\(imageWidth)x\(imageHeight) grid_aligned=true " +
            "geocalib=\(geoCalibCalibration == nil ? "nil" : "available") " +
            "valid_depths=\(stats.validCount) invalid_depths=\(stats.invalidCount) " +
            "depth_min=\(Self.formatMeters(stats.min)) depth_p05=\(Self.formatMeters(stats.p05)) " +
            "depth_median=\(Self.formatMeters(stats.median)) depth_p95=\(Self.formatMeters(stats.p95)) " +
            "depth_max=\(Self.formatMeters(stats.max)) center_depth=\(Self.formatMeters(stats.centerDepth)) " +
            "focal35mm=\(String(format: "%.2f", focal.focal35mm)) focal_source=\(focal.source) " +
            "focal_px=\(String(format: "%.2f", focalPx)) fx=fy=\(String(format: "%.2f", focalPx)) " +
            "wall_margin=\(String(format: "%.3f", wallMargin)) " +
            "rect_px=x:\(rect.leftX)-\(rect.rightX),y:\(rect.topY)-\(rect.bottomY),sample:\(rect.sampleCenterX),\(rect.sampleCenterY) " +
            "wall_dims_source=depth_map_center_depth_plus_projected_wall_rect " +
            "wall_width_formula=\(wallWidthFormula) wall_height_formula=\(wallHeightFormula) " +
            "wall_dims_m=W:\(String(format: "%.4f", wallMeasured.width)),H:\(String(format: "%.4f", wallMeasured.height)),D:\(String(format: "%.4f", wallMeasured.depth)) " +
            "depth_spread_dims_m=W:\(String(format: "%.4f", depthSpreadMeasured.width)),H:\(String(format: "%.4f", depthSpreadMeasured.height)),D:\(String(format: "%.4f", depthSpreadMeasured.depth)) " +
            "object_bbox_dims_raw_m=\(rawObjectDimsSummary) object_bbox_dims_m=\(objectDimsSummary) " +
            "depth_metric_scale=\(String(format: "%.4f", metricCalibration.depthScale)) " +
            "measurement_depth_scale=\(String(format: "%.4f", measurementCalibration.depthScale)) " +
            "measurement_source=\(measurementCalibration.sourceLabel) " +
            "depth_metric_source=\(metricCalibration.sourceLabel) " +
            "result_dims_source=leveled_wall_spread_wd_ceiling_anchored_h_camera_height_1.7m " +
            "result_dims_m=W:\(String(format: "%.4f", measured.width)),H:\(String(format: "%.4f", measured.height)),D:\(String(format: "%.4f", measured.depth))"
        )
        logDebug(
            "[DepthAnythingRoom] depth_spread image=\(imageWidth)x\(imageHeight) " +
            "focal_px=\(String(format: "%.1f", focalPx)) " +
            "W=\(String(format: "%.3f", measured.width)) " +
            "H=\(String(format: "%.3f", measured.height)) " +
            "D=\(String(format: "%.3f", measured.depth)) m"
        )
        let meshRoomWidthMeters = max(measured.width, Self.minimumRoomWidthMeters)
        let meshRoomHeightMeters = max(measured.height, Self.minimumRoomWidthMeters * Float(imageHeight) / Float(max(imageWidth, 1)))
        logDebug(
            "[DepthAnythingRoom][MeshScale] source=depth_spread_dims " +
            "mesh_width_m=\(String(format: "%.4f", meshRoomWidthMeters)) " +
            "mesh_height_m=\(String(format: "%.4f", meshRoomHeightMeters)) " +
            "result_dims_source=depth_unprojected_spread"
        )
        let measurementEnd = CFAbsoluteTimeGetCurrent()
        let meshDepthMap = Self.usesFlatMesh ? Self.flattenDepthForMesh(calibratedDepthMap) : calibratedDepthMap
        let mesh = try buildMesh(
            image: workingImage,
            depthMap: meshDepthMap,
            roomWidthMeters: meshRoomWidthMeters,
            roomHeightMeters: meshRoomHeightMeters
        )
        let meshEnd = CFAbsoluteTimeGetCurrent()
        let url = try exportUSDZ(mesh: mesh, textureImage: workingImage)
        let exportEnd = CFAbsoluteTimeGetCurrent()
        logDebug(String(
            format: "[DepthAnythingRoom][Timing] measurement_math %.0f ms · mesh_build %.0f ms · usdz_export %.0f ms · total %.0f ms",
            (measurementEnd - inferenceEnd) * 1000,
            (meshEnd - measurementEnd) * 1000,
            (exportEnd - meshEnd) * 1000,
            (exportEnd - pipelineStart) * 1000
        ))
        return DepthAnythingRoomResult(
            usdzURL: url,
            vertexCount: mesh.vertexCount,
            triangleCount: mesh.submeshes?.compactMap { $0 as? MDLSubmesh }.reduce(0) { $0 + $1.indexCount / 3 } ?? 0,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            roomWidthMeters: measured.width,
            roomHeightMeters: measured.height,
            roomDepthMeters: measured.depth,
            calibrationMetadata: enrichedCalibrationMetadata
        )
    }

    func inferDepth(image: UIImage) async throws -> [[Float]] {
        let fixedImage = image.fixedOrientation()
        guard let cgImage = fixedImage.cgImage else {
            throw DepthAnythingRoomError.invalidImage
        }

        let targetWidth = cgImage.width
        let targetHeight = cgImage.height
        let observation = try await runVisionDepthRequest(cgImage: cgImage)
        let dense = try Self.depthGrid(from: observation)
        let rawStats = Self.depthValueStats(dense.values)
        // Python parity: the desktop pipeline stretches to 518x518 (scaleFill), so the
        // whole canvas is content — no letterbox crop.
        let contentValues = dense.values
        let contentWidth = dense.width
        let contentHeight = dense.height
        let remapped = Self.resizeBilinear(
            values: contentValues,
            width: contentWidth,
            height: contentHeight,
            targetWidth: targetWidth,
            targetHeight: targetHeight
        )
        let resizedStats = Self.depthValueStats(remapped)
        logDebug(
            "[DepthAnythingRoom][InferenceRaw] observation=\(String(describing: type(of: observation))) " +
            "raw_depth_map=\(dense.width)x\(dense.height) letterbox_content=\(contentWidth)x\(contentHeight) " +
            "target_px=\(targetWidth)x\(targetHeight) " +
            "raw_valid=\(rawStats.validCount) raw_invalid=\(rawStats.invalidCount) " +
            "raw_min=\(Self.formatMeters(rawStats.min)) raw_median=\(Self.formatMeters(rawStats.median)) raw_max=\(Self.formatMeters(rawStats.max)) " +
            "resized_valid=\(resizedStats.validCount) resized_invalid=\(resizedStats.invalidCount) " +
            "resized_min=\(Self.formatMeters(resizedStats.min)) resized_median=\(Self.formatMeters(resizedStats.median)) resized_max=\(Self.formatMeters(resizedStats.max))"
        )

        var rows: [[Float]] = []
        rows.reserveCapacity(targetHeight)
        for row in 0..<targetHeight {
            let start = row * targetWidth
            rows.append(Array(remapped[start..<start + targetWidth]))
        }
        return rows
    }

    func buildMesh(
        image: UIImage,
        depthMap: [[Float]],
        roomWidthMeters: Float,
        roomHeightMeters: Float
    ) throws -> MDLMesh {
        let fixedImage = image.fixedOrientation()
        let raster = try DepthAnythingRasterImage(image: fixedImage)
        let imageWidth = raster.width
        let imageHeight = raster.height
        guard depthMap.count == imageHeight,
              depthMap.allSatisfy({ $0.count == imageWidth }) else {
            throw DepthAnythingRoomError.depthImageSizeMismatch
        }

        var depthMax: Float = 0
        for row in depthMap {
            for depth in row where depth.isFinite {
                depthMax = max(depthMax, depth)
            }
        }
        guard depthMax > 0 else {
            throw DepthAnythingRoomError.invalidDepthOutput
        }

        let sampledRows = Array(stride(from: 0, to: imageHeight, by: pixelStep))
        let sampledColumns = Array(stride(from: 0, to: imageWidth, by: pixelStep))
        let rowCount = sampledRows.count
        let columnCount = sampledColumns.count
        var vertexIndices = [Int32](repeating: -1, count: rowCount * columnCount)
        var vertexData = Data()
        vertexData.reserveCapacity(rowCount * columnCount * DepthAnythingVertex.byteStride)
        var vertexCount = 0

        let pixelScaleX = roomWidthMeters / Float(imageWidth)
        let pixelScaleY = roomHeightMeters / Float(imageHeight)
        let centerX = Float(imageWidth) / 2.0
        let centerY = Float(imageHeight) / 2.0

        for (sampledRowIndex, row) in sampledRows.enumerated() {
            for (sampledColumnIndex, column) in sampledColumns.enumerated() {
                let depth = depthMap[row][column]
                guard depth.isFinite, depth > 0 else { continue }

                let x = -(Float(column) - centerX) * pixelScaleX
                let y = (Float(row) - centerY) * pixelScaleY
                let z = -(depthMax - depth)
                let color = raster.color(x: column, y: row).floatRGB
                let u = Float(column) / Float(max(imageWidth - 1, 1))
                let v = 1.0 - Float(row) / Float(max(imageHeight - 1, 1))

                vertexData.appendFloat32LE(x)
                vertexData.appendFloat32LE(y)
                vertexData.appendFloat32LE(z)
                vertexData.appendFloat32LE(color.x)
                vertexData.appendFloat32LE(color.y)
                vertexData.appendFloat32LE(color.z)
                vertexData.appendFloat32LE(u)
                vertexData.appendFloat32LE(v)
                vertexIndices[sampledRowIndex * columnCount + sampledColumnIndex] = Int32(vertexCount)
                vertexCount += 1
            }
        }

        guard vertexCount > 0 else {
            throw DepthAnythingRoomError.emptyMesh
        }

        var indexData = Data()
        indexData.reserveCapacity(max(0, (rowCount - 1) * (columnCount - 1) * 6 * MemoryLayout<UInt32>.size))
        var indexCount = 0

        func sampledIndex(_ row: Int, _ column: Int) -> Int {
            row * columnCount + column
        }

        for rowIndex in 0..<(rowCount - 1) {
            for columnIndex in 0..<(columnCount - 1) {
                let i00 = sampledIndex(rowIndex, columnIndex)
                let i10 = sampledIndex(rowIndex, columnIndex + 1)
                let i01 = sampledIndex(rowIndex + 1, columnIndex)
                let i11 = sampledIndex(rowIndex + 1, columnIndex + 1)
                let v00 = vertexIndices[i00]
                let v10 = vertexIndices[i10]
                let v01 = vertexIndices[i01]
                let v11 = vertexIndices[i11]
                guard v00 >= 0, v10 >= 0, v01 >= 0, v11 >= 0 else { continue }

                let r0 = sampledRows[rowIndex]
                let r1 = sampledRows[rowIndex + 1]
                let c0 = sampledColumns[columnIndex]
                let c1 = sampledColumns[columnIndex + 1]
                let d00 = depthMap[r0][c0]
                let d10 = depthMap[r0][c1]
                let d01 = depthMap[r1][c0]
                let d11 = depthMap[r1][c1]
                guard Self.depthsAreContinuous(
                    d00,
                    d10,
                    d01,
                    d11,
                    threshold: depthDiscontinuityThresholdMeters
                ) else {
                    continue
                }

                indexData.appendUInt32LE(UInt32(v00))
                indexData.appendUInt32LE(UInt32(v10))
                indexData.appendUInt32LE(UInt32(v11))
                indexData.appendUInt32LE(UInt32(v00))
                indexData.appendUInt32LE(UInt32(v11))
                indexData.appendUInt32LE(UInt32(v01))
                indexCount += 6
            }
        }

        guard indexCount >= 3 else {
            throw DepthAnythingRoomError.emptyMesh
        }

        let allocator = MDLMeshBufferDataAllocator()
        let vertexBuffer = allocator.newBuffer(with: vertexData, type: .vertex)
        let indexBuffer = allocator.newBuffer(with: indexData, type: .index)

        let descriptor = MDLVertexDescriptor()
        descriptor.attributes[0] = MDLVertexAttribute(
            name: MDLVertexAttributePosition,
            format: .float3,
            offset: 0,
            bufferIndex: 0
        )
        descriptor.attributes[1] = MDLVertexAttribute(
            name: MDLVertexAttributeColor,
            format: .float3,
            offset: 12,
            bufferIndex: 0
        )
        descriptor.attributes[2] = MDLVertexAttribute(
            name: MDLVertexAttributeTextureCoordinate,
            format: .float2,
            offset: 24,
            bufferIndex: 0
        )
        descriptor.layouts[0] = MDLVertexBufferLayout(stride: DepthAnythingVertex.byteStride)

        let submesh = MDLSubmesh(
            indexBuffer: indexBuffer,
            indexCount: indexCount,
            indexType: .uInt32,
            geometryType: .triangles,
            material: nil
        )
        let mesh = MDLMesh(
            vertexBuffer: vertexBuffer,
            vertexCount: vertexCount,
            descriptor: descriptor,
            submeshes: [submesh]
        )
        mesh.name = "DepthAnythingMetricRoom"
        return mesh
    }

    func exportUSDZ(mesh: MDLMesh, textureImage: UIImage) throws -> URL {
        let directory = try resolvedOutputDirectory()
        let url = directory.appendingPathComponent("DepthAnythingRoom_\(Self.outputStamp()).usdz")

        var exportErrors: [String] = []
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            let scene = try Self.makeSceneKitScene(from: mesh, textureImage: textureImage)
            let didWrite = scene.write(to: url, options: nil, delegate: nil, progressHandler: nil)
            guard didWrite, FileManager.default.fileExists(atPath: url.path) else {
                throw DepthAnythingRoomError.exportFailed("SceneKit writer returned false.")
            }
            return url
        } catch {
            exportErrors.append("SceneKit: \(error.localizedDescription)")
        }

        if MDLAsset.canExportFileExtension("usdz") {
            do {
                let asset = MDLAsset(bufferAllocator: MDLMeshBufferDataAllocator())
                asset.add(mesh)
                try asset.export(to: url)
                return url
            } catch {
                exportErrors.append("ModelIO: \(error.localizedDescription)")
            }
        } else {
            exportErrors.append("ModelIO cannot export USDZ on this platform")
        }

        throw DepthAnythingRoomError.exportFailed(exportErrors.joined(separator: "; "))
    }

    private static func makeSceneKitScene(from mesh: MDLMesh, textureImage: UIImage) throws -> SCNScene {
        guard let vertexBuffer = mesh.vertexBuffers.first else {
            throw DepthAnythingRoomError.emptyMesh
        }
        let vertexData = Data(
            bytes: vertexBuffer.map().bytes,
            count: mesh.vertexCount * DepthAnythingVertex.byteStride
        )
        let positionSource = SCNGeometrySource(
            data: vertexData,
            semantic: .vertex,
            vectorCount: mesh.vertexCount,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: DepthAnythingVertex.byteStride
        )
        let colorSource = SCNGeometrySource(
            data: vertexData,
            semantic: .color,
            vectorCount: mesh.vertexCount,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 3 * MemoryLayout<Float>.size,
            dataStride: DepthAnythingVertex.byteStride
        )
        let texcoordSource = SCNGeometrySource(
            data: vertexData,
            semantic: .texcoord,
            vectorCount: mesh.vertexCount,
            usesFloatComponents: true,
            componentsPerVector: 2,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 6 * MemoryLayout<Float>.size,
            dataStride: DepthAnythingVertex.byteStride
        )

        let elements = try (mesh.submeshes as? [MDLSubmesh] ?? []).map { submesh -> SCNGeometryElement in
            guard submesh.indexType == .uInt32 else {
                throw DepthAnythingRoomError.exportFailed("SceneKit fallback expected UInt32 mesh indices.")
            }
            let indexData = Data(
                bytes: submesh.indexBuffer.map().bytes,
                count: submesh.indexCount * MemoryLayout<UInt32>.size
            )
            return SCNGeometryElement(
                data: indexData,
                primitiveType: .triangles,
                primitiveCount: submesh.indexCount / 3,
                bytesPerIndex: MemoryLayout<UInt32>.size
            )
        }
        guard !elements.isEmpty else {
            throw DepthAnythingRoomError.emptyMesh
        }

        let geometry = SCNGeometry(sources: [positionSource, colorSource, texcoordSource], elements: elements)
        let material = SCNMaterial()
        material.diffuse.contents = textureImage
        material.emission.contents = textureImage
        material.lightingModel = .constant
        material.isDoubleSided = true
        material.diffuse.wrapS = .clamp
        material.diffuse.wrapT = .clamp
        geometry.materials = [material]

        let scene = SCNScene()
        let node = SCNNode(geometry: geometry)
        node.name = "DepthAnythingMetricRoom"
        scene.rootNode.addChildNode(node)
        return scene
    }

    private func runVisionDepthRequest(cgImage: CGImage) async throws -> VNObservation {
        try await withCheckedThrowingContinuation { continuation in
            let request = VNCoreMLRequest(model: model) { request, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let observation = request.results?.first else {
                    continuation.resume(throwing: DepthAnythingRoomError.invalidDepthOutput)
                    return
                }
                continuation.resume(returning: observation)
            }
            // Python parity: desktop preprocess stretches to the square model input
            // (PIL resize to 518x518), which is Vision's scaleFill, not scaleFit.
            request.imageCropAndScaleOption = .scaleFill

            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:]).perform([request])
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func loadDepthAnythingModel(configuration: MLModelConfiguration) throws -> (model: MLModel, name: String) {
        let candidates = [
            "DepthAnythingV2MetricIndoorSmall",
            "DepthAnythingV2MetricIndoorSmallF16",
            "depthanythingv2metricindoorsmall",
            "depthanythingv2metricindoorsmallf16",
        ]
        let extensions = ["mlmodelc", "mlpackage", "mlmodel"]

        for sourceURL in candidateModelURLs(baseNames: candidates, extensions: extensions) {
            let ext = sourceURL.pathExtension
            guard !ext.isEmpty else { continue }
            do {
                let modelURL = ext == "mlpackage" || ext == "mlmodel"
                    ? try MLModel.compileModel(at: sourceURL)
                    : sourceURL
                return (try MLModel(contentsOf: modelURL, configuration: configuration), sourceURL.lastPathComponent)
            } catch {
                continue
            }
        }

        throw DepthAnythingRoomError.modelNotFound
    }

    private static func candidateModelURLs(baseNames: [String], extensions: [String]) -> [URL] {
        var urls: [URL] = []
        var seen = Set<URL>()

        func append(_ url: URL) {
            let standardized = url.standardizedFileURL
            guard !seen.contains(standardized) else { return }
            seen.insert(standardized)
            urls.append(standardized)
        }

        for baseName in baseNames {
            for ext in extensions {
                if let url = Bundle.main.url(forResource: baseName, withExtension: ext) {
                    append(url)
                }
            }
        }

        guard let resourceURL = Bundle.main.resourceURL,
              let enumerator = FileManager.default.enumerator(
                at: resourceURL,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
              ) else {
            return urls
        }

        let allowedNames = Set(baseNames.map { $0.lowercased() })
        let allowedExtensions = Set(extensions.map { $0.lowercased() })
        for case let url as URL in enumerator {
            let ext = url.pathExtension.lowercased()
            guard allowedExtensions.contains(ext),
                  allowedNames.contains(url.deletingPathExtension().lastPathComponent.lowercased()) else {
                continue
            }
            append(url)
            if ext == "mlpackage" || ext == "mlmodelc" {
                enumerator.skipDescendants()
            }
        }

        return urls
    }

    private static func depthGrid(from observation: VNObservation) throws -> DenseDepthGrid {
        if let featureObservation = observation as? VNCoreMLFeatureValueObservation,
           let multiArray = featureObservation.featureValue.multiArrayValue {
            let dense = try denseArray(from: multiArray)
            guard dense.shape.count >= 2 else {
                throw DepthAnythingRoomError.invalidDepthOutput
            }
            let height = dense.shape[dense.shape.count - 2]
            let width = dense.shape[dense.shape.count - 1]
            guard width > 0, height > 0, dense.values.count >= width * height else {
                throw DepthAnythingRoomError.invalidDepthOutput
            }
            let suffix = Array(dense.values.suffix(width * height))
            return DenseDepthGrid(width: width, height: height, values: suffix)
        }

        if let pixelObservation = observation as? VNPixelBufferObservation {
            return try denseImage(from: pixelObservation.pixelBuffer)
        }

        throw DepthAnythingRoomError.invalidDepthOutput
    }

    private static func denseArray(from multiArray: MLMultiArray) throws -> DenseDepthArray {
        let shape = multiArray.shape.map(\.intValue)
        let count = multiArray.count
        guard count > 0 else {
            throw DepthAnythingRoomError.invalidDepthOutput
        }

        if isRowMajorContiguous(multiArray) {
            switch multiArray.dataType {
            case .float32:
                let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
                return DenseDepthArray(shape: shape, values: Array(UnsafeBufferPointer(start: ptr, count: count)))
            case .float16:
                let ptr = multiArray.dataPointer.bindMemory(to: UInt16.self, capacity: count)
                let values = UnsafeBufferPointer(start: ptr, count: count).map { Float(Float16(bitPattern: $0)) }
                return DenseDepthArray(shape: shape, values: values)
            case .double:
                let ptr = multiArray.dataPointer.bindMemory(to: Double.self, capacity: count)
                return DenseDepthArray(shape: shape, values: UnsafeBufferPointer(start: ptr, count: count).map(Float.init))
            default:
                break
            }
        }

        var indices = [Int](repeating: 0, count: shape.count)
        var values: [Float] = []
        values.reserveCapacity(count)

        func visit(_ dimension: Int) {
            if dimension == shape.count {
                values.append(multiArray[indices.map(NSNumber.init(value:))].floatValue)
                return
            }
            for index in 0..<shape[dimension] {
                indices[dimension] = index
                visit(dimension + 1)
            }
        }
        visit(0)
        return DenseDepthArray(shape: shape, values: values)
    }

    private static func isRowMajorContiguous(_ multiArray: MLMultiArray) -> Bool {
        let shape = multiArray.shape.map(\.intValue)
        let strides = multiArray.strides.map(\.intValue)
        guard shape.count == strides.count else { return false }
        var expected = 1
        for index in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[index] != expected { return false }
            expected *= max(shape[index], 1)
        }
        return true
    }

    private static func denseImage(from pixelBuffer: CVPixelBuffer) throws -> DenseDepthGrid {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width > 0, height > 0,
              let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DepthAnythingRoomError.invalidDepthOutput
        }

        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        var values = [Float](repeating: 0, count: width * height)

        switch format {
        case kCVPixelFormatType_DepthFloat32, kCVPixelFormatType_OneComponent32Float:
            for y in 0..<height {
                let row = baseAddress.advanced(by: y * rowBytes).assumingMemoryBound(to: Float.self)
                for x in 0..<width {
                    values[y * width + x] = row[x]
                }
            }
        case kCVPixelFormatType_DepthFloat16, kCVPixelFormatType_OneComponent16Half:
            for y in 0..<height {
                let row = baseAddress.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt16.self)
                for x in 0..<width {
                    values[y * width + x] = Float(Float16(bitPattern: row[x]))
                }
            }
        default:
            throw DepthAnythingRoomError.invalidDepthOutput
        }

        return DenseDepthGrid(width: width, height: height, values: values)
    }

    private static func resizeBilinear(
        values: [Float],
        width: Int,
        height: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> [Float] {
        guard width != targetWidth || height != targetHeight else { return values }
        var output = [Float](repeating: 0, count: targetWidth * targetHeight)
        for y in 0..<targetHeight {
            let sourceY = (Float(y) + 0.5) * Float(height) / Float(targetHeight) - 0.5
            for x in 0..<targetWidth {
                let sourceX = (Float(x) + 0.5) * Float(width) / Float(targetWidth) - 0.5
                output[y * targetWidth + x] = bilinearSample(
                    values: values,
                    width: width,
                    height: height,
                    x: sourceX,
                    y: sourceY
                )
            }
        }
        return output
    }

    private static func downsampledImage(_ image: UIImage, maxDimension: Int) throws -> UIImage {
        guard let cgImage = image.cgImage else {
            throw DepthAnythingRoomError.invalidImage
        }

        let width = cgImage.width
        let height = cgImage.height
        let longestSide = max(width, height)
        guard longestSide > maxDimension else {
            return image
        }

        let scale = CGFloat(maxDimension) / CGFloat(longestSide)
        let targetWidth = max(1, Int((CGFloat(width) * scale).rounded()))
        let targetHeight = max(1, Int((CGFloat(height) * scale).rounded()))
        guard let context = CGContext(
            data: nil,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw DepthAnythingRoomError.invalidImage
        }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
        guard let resized = context.makeImage() else {
            throw DepthAnythingRoomError.invalidImage
        }
        return UIImage(cgImage: resized, scale: 1.0, orientation: .up)
    }

    private static func bilinearSample(values: [Float], width: Int, height: Int, x: Float, y: Float) -> Float {
        let clampedX = min(max(x, 0), Float(width - 1))
        let clampedY = min(max(y, 0), Float(height - 1))
        let x0 = Int(floor(clampedX))
        let y0 = Int(floor(clampedY))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        let tx = clampedX - Float(x0)
        let ty = clampedY - Float(y0)
        let v00 = values[y0 * width + x0]
        let v10 = values[y0 * width + x1]
        let v01 = values[y1 * width + x0]
        let v11 = values[y1 * width + x1]
        let top = v00 * (1 - tx) + v10 * tx
        let bottom = v01 * (1 - tx) + v11 * tx
        return top * (1 - ty) + bottom * ty
    }

    private static func depthsAreContinuous(
        _ d00: Float,
        _ d10: Float,
        _ d01: Float,
        _ d11: Float,
        threshold: Float
    ) -> Bool {
        let depths = [d00, d10, d01, d11]
        guard depths.allSatisfy({ $0.isFinite && $0 > 0 }) else { return false }
        for lhs in 0..<depths.count {
            for rhs in (lhs + 1)..<depths.count {
                if abs(depths[lhs] - depths[rhs]) > threshold {
                    return false
                }
            }
        }
        return true
    }

    private func resolvedOutputDirectory() throws -> URL {
        let directory = outputDirectory ?? FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DepthAnythingRooms", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func outputStamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss_SSS"
        return formatter.string(from: Date())
    }

    private static func flattenDepthForMesh(_ depthMap: [[Float]]) -> [[Float]] {
        var validDepths: [Float] = []
        validDepths.reserveCapacity(depthMap.count * max(depthMap.first?.count ?? 0, 0))
        for row in depthMap {
            for depth in row where depth.isFinite && depth > 0 {
                validDepths.append(depth)
            }
        }
        guard !validDepths.isEmpty else { return depthMap }
        let planeDepth = median(validDepths)
        return depthMap.map { row in
            row.map { depth in
                depth.isFinite && depth > 0 ? planeDepth : depth
            }
        }
    }

    private static func measureWall(
        depthMap: [[Float]],
        imageWidth: Int,
        imageHeight: Int,
        fx: Float,
        fy: Float,
        wallMargin: Float
    ) -> (width: Float, height: Float, depth: Float) {
        let margin = min(max(wallMargin, 0), 0.45)
        let rectX = margin * Float(imageWidth)
        let rectY = margin * Float(imageHeight)
        let rectWidth = (1.0 - 2.0 * margin) * Float(imageWidth)
        let rectHeight = (1.0 - 2.0 * margin) * Float(imageHeight)

        let centerX = Float(imageWidth - 1) * 0.5
        let centerY = Float(imageHeight - 1) * 0.5
        let leftX = Int(round(rectX))
        let rightX = Int(round(rectX + rectWidth - 1))
        let topY = Int(round(rectY))
        let bottomY = Int(round(rectY + rectHeight - 1))
        let sampleCenterX = Int(round(rectX + rectWidth * 0.5))
        let sampleCenterY = Int(round(rectY + rectHeight * 0.5))

        guard let centerDepth = medianAt(
            depthMap: depthMap,
            x: sampleCenterX,
            y: sampleCenterY,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        ) else {
            let fallbackDepth = centerDepthFallback(depthMap)
            let fallbackWidth = max(minimumRoomWidthMeters, Float(imageWidth) / fx * fallbackDepth)
            let fallbackHeight = fallbackWidth * Float(imageHeight) / Float(max(imageWidth, 1))
            return (fallbackWidth, fallbackHeight, fallbackDepth)
        }

        let leftPlane = (Float(leftX) - centerX) * centerDepth / fx
        let rightPlane = (Float(rightX) - centerX) * centerDepth / fx
        let topPlane = (Float(topY) - centerY) * centerDepth / fy
        let bottomPlane = (Float(bottomY) - centerY) * centerDepth / fy
        return (
            abs(rightPlane - leftPlane),
            abs(bottomPlane - topPlane),
            centerDepth
        )
    }

    /// Wall-anchored, gravity-leveled room dims (mirrors the Python back-wall quad idea):
    /// unproject and level all pixels, take room depth at the far wall (p80 of leveled Z),
    /// then measure W/H only from points near that wall. Robust to camera pitch — a
    /// camera-frame full-image spread overshoots height when the phone is tilted up.
    private static func measureDepthSpread(
        depthMap: [[Float]],
        imageWidth: Int,
        imageHeight: Int,
        fx: Float,
        fy: Float,
        wallMargin: Float,
        levelingRotation: simd_float3x3,
        cameraHeightPriorMeters: Float,
        sampleStep: Int = 8
    ) -> (width: Float, height: Float, depth: Float)? {
        guard imageWidth > 1, imageHeight > 1, fx > 1, fy > 1 else { return nil }

        let margin = min(max(wallMargin, 0), 0.45)
        let leftX = Int(round(margin * Float(imageWidth)))
        let rightX = Int(round((1.0 - margin) * Float(imageWidth))) - 1
        let topY = Int(round(margin * Float(imageHeight)))
        let bottomY = Int(round((1.0 - margin) * Float(imageHeight))) - 1
        guard leftX < rightX, topY < bottomY else { return nil }

        let centerX = Float(imageWidth - 1) * 0.5
        let centerY = Float(imageHeight - 1) * 0.5
        var leveledX: [Float] = []
        var leveledY: [Float] = []
        var leveledZ: [Float] = []
        var ceilingClearances: [Float] = []
        var ceilingDepths: [Float] = []
        let estimatedCount = (rightX - leftX) * (bottomY - topY) / max(1, sampleStep * sampleStep)
        leveledX.reserveCapacity(estimatedCount)
        leveledY.reserveCapacity(estimatedCount)
        leveledZ.reserveCapacity(estimatedCount)
        let ceilingRowCutoff = Float(topY) + ceilingBandRowFraction * Float(bottomY - topY)

        var y = topY
        while y <= bottomY {
            var x = leftX
            while x <= rightX {
                let depth = depthMap[y][x]
                if depth.isFinite, depth > 0 {
                    let point = levelingRotation * unprojectCameraPoint(
                        column: Float(x),
                        row: Float(y),
                        depth: depth,
                        fx: fx,
                        fy: fy,
                        centerX: centerX,
                        centerY: centerY
                    )
                    if point.z > 0 {
                        leveledX.append(point.x)
                        leveledY.append(point.y)
                        leveledZ.append(point.z)
                        // Ceiling candidates: top image band, clearly above the camera.
                        if Float(y) < ceilingRowCutoff, point.y < -minimumCeilingClearanceMeters {
                            ceilingClearances.append(-point.y)
                            ceilingDepths.append(point.z)
                        }
                    }
                }
                x += sampleStep
            }
            y += sampleStep
        }

        guard leveledZ.count >= 32 else { return nil }
        let sortedZ = leveledZ.sorted()
        guard let roomDepth = percentile(sorted: sortedZ, fraction: 0.80), roomDepth > 0 else { return nil }

        // Keep points near the back wall so near-camera floor/ceiling doesn't stretch W/H.
        var wallX: [Float] = []
        var wallY: [Float] = []
        wallX.reserveCapacity(leveledZ.count)
        wallY.reserveCapacity(leveledZ.count)
        let farCutoff = 0.6 * roomDepth
        for index in 0..<leveledZ.count where leveledZ[index] > farCutoff {
            wallX.append(leveledX[index])
            wallY.append(leveledY[index])
        }
        if wallX.count < 64 {
            wallX = leveledX
            wallY = leveledY
        }
        wallX.sort()
        wallY.sort()

        guard let xLow = percentile(sorted: wallX, fraction: 0.04),
              let xHigh = percentile(sorted: wallX, fraction: 0.96),
              let yLow = percentile(sorted: wallY, fraction: 0.03),
              let yHigh = percentile(sorted: wallY, fraction: 0.97) else {
            return nil
        }
        let width = xHigh - xLow
        var height = yHigh - yLow
        // Floor+ceiling anchored height (Python parity: median(floor_y) - median(ceiling_y)).
        // The 1.7 m camera-height scale pins the floor, so height = 1.7 + ceiling clearance.
        // Far more stable than the vertical spread, which inflates under camera pitch.
        if ceilingClearances.count >= 32 {
            var clearance = median(ceilingClearances)
            // A flat ceiling has constant leveled clearance; residual pitch error δ adds
            // ~z·sin(δ), tilting clearance-vs-depth. The z=0 intercept of a robust line
            // fit recovers the true clearance. Correct only in the inflation direction.
            if let fit = robustLineFit(x: ceilingDepths, y: ceilingClearances),
               fit.slope > 0.02, fit.intercept > minimumCeilingClearanceMeters {
                clearance = fit.intercept
            }
            let ceilingHeight = cameraHeightPriorMeters + clearance
            if ceilingAnchoredHeightRange.contains(ceilingHeight) {
                height = ceilingHeight
            }
        }
        guard width.isFinite, height.isFinite, roomDepth.isFinite, width > 0, height > 0 else {
            return nil
        }
        return (width, height, roomDepth)
    }

    /// Robust slope/intercept: sort by x, pair the first half with the second half, take the
    /// median pair slope, then the median intercept (Theil–Sen flavored, outlier tolerant).
    private static func robustLineFit(x: [Float], y: [Float]) -> (slope: Float, intercept: Float)? {
        guard x.count == y.count, x.count >= 16 else { return nil }
        let order = (0..<x.count).sorted { x[$0] < x[$1] }
        let half = x.count / 2
        var slopes: [Float] = []
        slopes.reserveCapacity(half)
        for index in 0..<half {
            let a = order[index]
            let b = order[index + half]
            let dx = x[b] - x[a]
            if dx > 1e-3 {
                slopes.append((y[b] - y[a]) / dx)
            }
        }
        guard slopes.count >= 8 else { return nil }
        let slope = median(slopes)
        var residuals: [Float] = []
        residuals.reserveCapacity(x.count)
        for index in 0..<x.count {
            residuals.append(y[index] - slope * x[index])
        }
        return (slope, median(residuals))
    }

    private static func unprojectCameraPoint(
        column: Float,
        row: Float,
        depth: Float,
        fx: Float,
        fy: Float,
        centerX: Float,
        centerY: Float
    ) -> SIMD3<Float> {
        SIMD3(
            (column - centerX) * depth / fx,
            (row - centerY) * depth / fy,
            depth
        )
    }

    private struct ObjectBBoxMeasurement {
        let classIdx: Int
        let width: Float
        let height: Float
        let depth: Float
        /// Pixel bbox in working-image coordinates, used to exclude furniture from floor sampling
        /// (mirrors the RTMDet exclude mask in `scripts/structure_box_measure_room.py`).
        let bboxLeftX: Int
        let bboxRightX: Int
        let bboxTopY: Int
        let bboxBottomY: Int
    }

    private struct DepthMetricCalibration {
        let depthScale: Float
        let focalPx: Float
        let geoFocalPx: Float
        let exifFocalPx: Float?
        let sourceLabel: String
        let sourceCode: Int
        let anchorClassIdx: Int?
        let anchorExpectedHeightMeters: Float?
        let anchorMeasuredHeightMeters: Float?
    }

    /// Room W×H×D uses camera-height scaling only; mesh preview keeps legacy anchor/exif calibration.
    private struct MeasurementDepthCalibration {
        let depthScale: Float
        let cameraHeightPriorMeters: Float
        let cameraHeightRawMeters: Float?
        let sourceLabel: String
        let sourceCode: Int
        /// Gravity leveling used for floor sampling; reused by the wall-anchored spread measurement.
        let levelingRotation: simd_float3x3
        /// 0 = identity (no leveling), 1 = GeoCalib CNN estimate, 2 = ARKit device gravity.
        let gravitySourceCode: Int
        /// 0 = fixed 1.7 m eye-level assumption, 1 = measured ARKit floor-plane height, 2 = fused imported-photo anchor.
        let cameraHeightPriorSourceCode: Int
        let scaleEstimatorConfidence: Float
        let scaleEstimatorDebug: String
    }

    private static func measurementFocalPixels(
        geoCalib: GeoCalibCalibrationResult?,
        imageWidth: Int,
        imageHeight: Int,
        trustedFocal: (fx: Float, fy: Float, focal35mm: Float, source: String)?,
        fallbackFx: Float,
        fallbackFy: Float
    ) -> (fx: Float, fy: Float, source: String) {
        // In-app captures carry real intrinsics (sidecar/ARKit/EXIF); those beat the
        // on-device mini GeoCalib CNN, which desktop Python only used because WhatsApp
        // imports have no EXIF. Prefer trusted capture focal; GeoCalib is the fallback.
        if let trustedFocal,
           trustedFocal.source == "exif" || trustedFocal.source.hasPrefix("sidecar"),
           trustedFocal.fx.isFinite, trustedFocal.fx > 1 {
            return (trustedFocal.fx, trustedFocal.fy, trustedFocal.source)
        }
        // Python parity: geocalib focal is square (fx == fy) in source-image pixels.
        // Rescale uniformly, and reject implausible values (CNN misfires).
        if let geoCalib, geoCalib.sourceWidth > 0 {
            let uniformScale = Float(imageWidth) / Float(max(geoCalib.sourceWidth, 1))
            let focal = geoCalib.focalLengthXPixels * uniformScale
            let heightRatio = focal / Float(max(imageHeight, 1))
            if focal.isFinite, focal > 1, measurementFocalHeightRatioRange.contains(heightRatio) {
                return (focal, focal, "geocalib")
            }
        }
        return (fallbackFx, fallbackFy, "fallback_metric_focal")
    }

    private static func sanitizeRoomMeasurement(
        _ spread: (width: Float, height: Float, depth: Float),
        wallFallback: (width: Float, height: Float, depth: Float)
    ) -> (width: Float, height: Float, depth: Float) {
        let spreadPlausible =
            plausibleRoomSpanRange.contains(spread.width) &&
            plausibleRoomSpanRange.contains(spread.height) &&
            plausibleRoomSpanRange.contains(spread.depth)
        if spreadPlausible {
            return spread
        }
        let wallPlausible =
            plausibleRoomSpanRange.contains(wallFallback.width) &&
            plausibleRoomSpanRange.contains(wallFallback.height) &&
            plausibleRoomSpanRange.contains(wallFallback.depth)
        if wallPlausible {
            return wallFallback
        }
        return spread
    }

    private static func resolveMeasurementCameraHeightScale(
        depthMap: [[Float]],
        workingImage: UIImage,
        geoCalib: GeoCalibCalibrationResult?,
        arkitGravityDown: SIMD3<Float>?,
        arkitCameraHeightMeters: Float?,
        imageWidth: Int,
        imageHeight: Int,
        fx: Float,
        fy: Float,
        wallMargin: Float,
        objectRect: ObjectDetectionRect?,
        furnitureExcludeBBox: (leftX: Int, rightX: Int, topY: Int, bottomY: Int)?
    ) -> MeasurementDepthCalibration {
        // Real ARKit floor-plane height beats the fixed eye-level assumption: a photo
        // taken from 1.2 m with a 1.7 m prior inflates every dimension by ~40%.
        var cameraHeightPrior: Float
        var priorSourceCode: Int
        if let arkitCameraHeightMeters, (0.5...2.2).contains(arkitCameraHeightMeters) {
            cameraHeightPrior = arkitCameraHeightMeters
            priorSourceCode = 1
        } else {
            cameraHeightPrior = cameraHeightPriorMeters
            priorSourceCode = 0
        }
        let levelingRotation: simd_float3x3
        let gravitySourceCode: Int
        if let arkitGravityDown {
            // Exact device gravity from the ARKit capture beats any CNN estimate.
            // `levelingRotationMatrix(gravityDown:)` expects the (0,-1,0)-at-level reference,
            // which in the +Y-down pinhole frame is the *up* vector.
            levelingRotation = GeoCalibCalibrationResult.levelingRotationMatrix(gravityDown: -arkitGravityDown)
            gravitySourceCode = 2
        } else if let geoCalib,
           abs(geoCalib.rollRadians) <= maxPlausibleRollRadians,
           abs(geoCalib.pitchRadians) <= maxPlausiblePitchRadians {
            levelingRotation = geoCalib.levelingRotationMatrix()
            gravitySourceCode = 1
        } else {
            if let geoCalib {
                logDebug(
                    "[DepthAnythingRoom][MetricCalib] geocalib_gravity_rejected " +
                    "roll=\(String(format: "%.3f", geoCalib.rollRadians)) " +
                    "pitch=\(String(format: "%.3f", geoCalib.pitchRadians))"
                )
            }
            levelingRotation = matrix_identity_float3x3
            gravitySourceCode = 0
        }
        let rawCameraHeight = cameraHeightFromFloorSamples(
            depthMap: depthMap,
            levelingRotation: levelingRotation,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            fx: fx,
            fy: fy,
            wallMargin: wallMargin,
            furnitureExcludeBBox: furnitureExcludeBBox
        )

        var depthScale: Float = 1.0
        var sourceLabel = "camera_height_unavailable"
        var sourceCode = 20
        // Python parity: no scale clamp — trust gate is the raw camera-height range itself
        // (scripts/structure_box_measure_room.py: `0.45 <= camera_height_raw <= 5.0`).
        if let rawCameraHeight, cameraHeightRawValidRange.contains(rawCameraHeight) {
            depthScale = cameraHeightPrior / rawCameraHeight
            sourceLabel = priorSourceCode == 1
                ? "camera_height_arkit_floor_plane"
                : "camera_height_prior_1.7m"
            sourceCode = 21
        }
        var scaleEstimatorConfidence: Float = 0
        var scaleEstimatorDebug = "not_run"
        if priorSourceCode == 0 {
            let vpGravity = VanishingPointGravity.refine(
                levelingRotation: levelingRotation,
                image: workingImage,
                focalPx: Double(fx)
            )
            let objectBoxes: [ScaleObjectBox] = objectRect.map { rect in
                [
                    ScaleObjectBox(
                        classIdx: rect.classIdx,
                        confidence: rect.confidence,
                        rect: CGRect(
                            x: CGFloat(rect.leftX),
                            y: CGFloat(rect.topY),
                            width: CGFloat(max(0, rect.rightX - rect.leftX)),
                            height: CGFloat(max(0, rect.bottomY - rect.topY))
                        )
                    )
                ]
            } ?? []
            let furnitureBoxes = furnitureExcludeBBox.map { bbox in
                [
                    CGRect(
                        x: CGFloat(bbox.leftX),
                        y: CGFloat(bbox.topY),
                        width: CGFloat(max(0, bbox.rightX - bbox.leftX)),
                        height: CGFloat(max(0, bbox.bottomY - bbox.topY))
                    )
                ]
            } ?? []
            let impliedRoomHeightForScale: (Double) -> Double? = { candidateScale in
                guard candidateScale.isFinite, candidateScale > 0 else { return nil }
                let scaledDepthMap = scaleDepthMap(depthMap, scale: Float(candidateScale))
                let impliedCameraHeightPrior = rawCameraHeight.map { $0 * Float(candidateScale) } ?? cameraHeightPrior
                if let spread = measureDepthSpread(
                    depthMap: scaledDepthMap,
                    imageWidth: imageWidth,
                    imageHeight: imageHeight,
                    fx: fx,
                    fy: fy,
                    wallMargin: wallMargin,
                    levelingRotation: vpGravity.levelingRotation,
                    cameraHeightPriorMeters: impliedCameraHeightPrior
                ) {
                    return Double(spread.height)
                }
                return nil
            }
            let ctx = SceneContext(
                rawDepth: depthMap,
                focalPx: Double(fx),
                levelingRotation: vpGravity.levelingRotation,
                furnitureBoxes: furnitureBoxes,
                objectBoxes: objectBoxes,
                workingImage: workingImage,
                imageSize: CGSize(width: imageWidth, height: imageHeight),
                rawCameraHeightMeters: rawCameraHeight.map(Double.init),
                fallbackDepthScale: Double(depthScale),
                impliedRoomHeightForScale: impliedRoomHeightForScale
            )
            let estimatorResult = ScaleEstimator().estimate(ctx: ctx)
            depthScale = Float(estimatorResult.depthScale)
            scaleEstimatorConfidence = Float(estimatorResult.confidence)
            scaleEstimatorDebug = estimatorResult.debugSummary + " vp=\(vpGravity.debug)"
            if let rawCameraHeight, scaleEstimatorConfidence > 0.05 {
                cameraHeightPrior = rawCameraHeight * depthScale
                priorSourceCode = 2
            }
            sourceLabel = estimatorResult.source
            sourceCode = 22
            logDebug("[ScaleResolver] \(scaleEstimatorDebug)")
        }

        return MeasurementDepthCalibration(
            depthScale: depthScale,
            cameraHeightPriorMeters: cameraHeightPrior,
            cameraHeightRawMeters: rawCameraHeight,
            sourceLabel: sourceLabel,
            sourceCode: sourceCode,
            levelingRotation: levelingRotation,
            gravitySourceCode: gravitySourceCode,
            cameraHeightPriorSourceCode: priorSourceCode,
            scaleEstimatorConfidence: scaleEstimatorConfidence,
            scaleEstimatorDebug: scaleEstimatorDebug
        )
    }

    /// Device gravity captured by ``ARRoomPhotoCaptureViewController`` (fixed-orientation pinhole frame).
    private static func arkitGravityDownVector(from metadata: [String: Double]?) -> SIMD3<Float>? {
        guard let metadata,
              let x = metadata["arkitGravityDownImageX"],
              let y = metadata["arkitGravityDownImageY"],
              let z = metadata["arkitGravityDownImageZ"] else {
            return nil
        }
        let vector = SIMD3(Float(x), Float(y), Float(z))
        guard vector.x.isFinite, vector.y.isFinite, vector.z.isFinite,
              simd_length(vector) > 0.5 else {
            return nil
        }
        return simd_normalize(vector)
    }

    /// Python parity (`camera_height_from_floor_samples` in scripts/structure_box_measure_room.py):
    /// unproject bottom-band floor pixels, level with GeoCalib gravity, take median of positive
    /// leveled Y. Furniture is excluded via the RTMDet bbox (mirrors the Python exclude mask);
    /// when RTMDet is unavailable, a fixed bottom-right corner heuristic is used instead.
    private static func cameraHeightFromFloorSamples(
        depthMap: [[Float]],
        levelingRotation: simd_float3x3,
        imageWidth: Int,
        imageHeight: Int,
        fx: Float,
        fy: Float,
        wallMargin: Float,
        furnitureExcludeBBox: (leftX: Int, rightX: Int, topY: Int, bottomY: Int)?
    ) -> Float? {
        guard imageWidth > 1, imageHeight > 1, fx > 1, fy > 1 else { return nil }

        let margin = min(max(wallMargin, 0), 0.45)
        let leftX = Int(round(margin * Float(imageWidth)))
        let rightX = Int(round((1.0 - margin) * Float(imageWidth))) - 1
        let floorStartY = Int(round(Float(imageHeight) * floorBandStartFraction))
        let bottomY = imageHeight - 1
        guard leftX < rightX, floorStartY < bottomY else { return nil }

        let centerX = Float(imageWidth - 1) * 0.5
        let centerY = Float(imageHeight - 1) * 0.5
        let step = max(4, (bottomY - floorStartY) / 32)
        let bandWidth = max(Float(rightX - leftX), 1)
        let bandHeight = max(Float(bottomY - floorStartY), 1)

        // Expand the RTMDet bbox slightly, like the dilated exclude mask in Python.
        let excludeRect: (leftX: Int, rightX: Int, topY: Int, bottomY: Int)? = furnitureExcludeBBox.map { bbox in
            let padX = Int(round(Float(bbox.rightX - bbox.leftX) * 0.05)) + 2
            let padY = Int(round(Float(bbox.bottomY - bbox.topY) * 0.05)) + 2
            return (
                leftX: max(0, bbox.leftX - padX),
                rightX: min(imageWidth - 1, bbox.rightX + padX),
                topY: max(0, bbox.topY - padY),
                bottomY: min(imageHeight - 1, bbox.bottomY + padY)
            )
        }

        func isExcluded(row: Int, column: Int) -> Bool {
            if let excludeRect {
                return column >= excludeRect.leftX && column <= excludeRect.rightX &&
                    row >= excludeRect.topY && row <= excludeRect.bottomY
            }
            let u = (Float(column) - Float(leftX)) / bandWidth
            let v = (Float(row) - Float(floorStartY)) / bandHeight
            return u > floorChairExcludeU && v > floorChairExcludeV
        }

        var cameraHeights: [Float] = []
        cameraHeights.reserveCapacity(512)
        var row = floorStartY
        while row <= bottomY {
            var column = leftX
            while column <= rightX {
                if isExcluded(row: row, column: column) {
                    column += step
                    continue
                }
                let depth = depthMap[row][column]
                guard depth.isFinite, depth > 0 else {
                    column += step
                    continue
                }
                let cameraPoint = unprojectCameraPoint(
                    column: Float(column),
                    row: Float(row),
                    depth: depth,
                    fx: fx,
                    fy: fy,
                    centerX: centerX,
                    centerY: centerY
                )
                let leveledPoint = levelingRotation * cameraPoint
                // Python: ys = leveled[floor_mask, 1]; keep ys > 0 (below camera), median.
                if leveledPoint.y > 0.05 {
                    cameraHeights.append(leveledPoint.y)
                }
                column += step
            }
            row += step
        }

        guard cameraHeights.count >= 32 else { return nil }
        return median(cameraHeights)
    }

    struct ObjectDetectionRect {
        let classIdx: Int
        let confidence: Float
        let leftX: Int
        let rightX: Int
        let topY: Int
        let bottomY: Int
    }

    /// RTMDet inference is image-only, so it can run concurrently with GeoCalib and
    /// Depth Anything. Runs detached to keep CoreML off the cooperative pool.
    private static func detectMeasurementObjectRect(
        image: UIImage,
        imageWidth: Int,
        imageHeight: Int
    ) async -> ObjectDetectionRect? {
        await Task.detached(priority: .userInitiated) {
            detectMeasurementObjectRectSync(image: image, imageWidth: imageWidth, imageHeight: imageHeight)
        }.value
    }

    private static func detectMeasurementObjectRectSync(
        image: UIImage,
        imageWidth: Int,
        imageHeight: Int
    ) -> ObjectDetectionRect? {
        guard let objectModel = loadObjectBBoxModel() else {
            logDebug("[DepthAnythingRoom][ObjectBBoxDims] unavailable reason=rtmdet_model_not_found")
            return nil
        }

        let inference: RTMDetInferenceResult
        do {
            inference = try RTMDetImageInference.runInstanceSegmentation(
                image: image,
                model: objectModel,
                confidenceThreshold: objectBBoxConfidenceThreshold,
                maxMaskCount: 1,
                maxDetectionCount: 25,
                buildInstanceMasks: false,
                cacheMaskBuildInputs: false,
                debug: false
            )
        } catch {
            logDebug("[DepthAnythingRoom][ObjectBBoxDims] unavailable reason=rtmdet_failed error=\(error.localizedDescription)")
            return nil
        }

        guard let detection = selectMeasurementObjectBBox(
            from: inference.detections,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        ) else {
            logDebug("[DepthAnythingRoom][ObjectBBoxDims] unavailable reason=no_valid_object_bbox detections=\(inference.detections.count)")
            return nil
        }

        let rect = clampedBBox(detection, imageWidth: imageWidth, imageHeight: imageHeight)
        guard rect.leftX < rect.rightX, rect.topY < rect.bottomY else {
            logDebug("[DepthAnythingRoom][ObjectBBoxDims] unavailable reason=empty_bbox cls=\(detection.classIdx)")
            return nil
        }
        return ObjectDetectionRect(
            classIdx: detection.classIdx,
            confidence: detection.confidence,
            leftX: rect.leftX,
            rightX: rect.rightX,
            topY: rect.topY,
            bottomY: rect.bottomY
        )
    }

    /// Depth math on an already-detected bbox (fast; no model inference).
    private static func measureObjectBBox(
        objectRect: ObjectDetectionRect?,
        depthMap: [[Float]],
        fx: Float,
        fy: Float
    ) -> ObjectBBoxMeasurement? {
        guard let rect = objectRect else { return nil }
        guard let depthSample = depthPercentile(
            depthMap: depthMap,
            leftX: rect.leftX,
            rightX: rect.rightX,
            topY: rect.topY,
            bottomY: rect.bottomY,
            fraction: 0.20
        ) else {
            logDebug("[DepthAnythingRoom][ObjectBBoxDims] unavailable reason=no_bbox_depth cls=\(rect.classIdx)")
            return nil
        }

        let bboxWidthPixels = Float(rect.rightX - rect.leftX)
        let bboxHeightPixels = Float(rect.bottomY - rect.topY)
        let widthMeters = bboxWidthPixels * depthSample.value / fx
        let heightMeters = bboxHeightPixels * depthSample.value / fy
        guard widthMeters.isFinite, heightMeters.isFinite, depthSample.value.isFinite,
              widthMeters > 0.03, heightMeters > 0.03, depthSample.value > 0.1 else {
            logDebug(
                "[DepthAnythingRoom][ObjectBBoxDims] unavailable reason=invalid_projected_dims " +
                "cls=\(rect.classIdx) bbox_px=\(Int(bboxWidthPixels))x\(Int(bboxHeightPixels)) " +
                "depth=\(String(format: "%.4f", depthSample.value)) " +
                "W=\(String(format: "%.4f", widthMeters)) H=\(String(format: "%.4f", heightMeters))"
            )
            return nil
        }

        logDebug(
            "[DepthAnythingRoom][ObjectBBoxDims] selected cls=\(rect.classIdx) " +
            "conf=\(String(format: "%.3f", rect.confidence)) " +
            "bbox_px=x:\(rect.leftX)-\(rect.rightX),y:\(rect.topY)-\(rect.bottomY),size:\(Int(bboxWidthPixels))x\(Int(bboxHeightPixels)) " +
            "depth_p20=\(String(format: "%.4f", depthSample.value)) depth_samples=\(depthSample.count) " +
            "fx=\(String(format: "%.2f", fx)) fy=\(String(format: "%.2f", fy)) " +
            "formula=W:\(Int(bboxWidthPixels))px*\(String(format: "%.4f", depthSample.value))m/\(String(format: "%.2f", fx))fx," +
            "H:\(Int(bboxHeightPixels))px*\(String(format: "%.4f", depthSample.value))m/\(String(format: "%.2f", fy))fy " +
            "dims_m=W:\(String(format: "%.4f", widthMeters)),H:\(String(format: "%.4f", heightMeters)),D:\(String(format: "%.4f", depthSample.value))"
        )

        return ObjectBBoxMeasurement(
            classIdx: rect.classIdx,
            width: widthMeters,
            height: heightMeters,
            depth: depthSample.value,
            bboxLeftX: rect.leftX,
            bboxRightX: rect.rightX,
            bboxTopY: rect.topY,
            bboxBottomY: rect.bottomY
        )
    }

    private static func resolveMetricCalibration(
        geoFocalPx: Float,
        cameraMetadata: [String: Double]?,
        imageWidth: Int,
        imageHeight: Int,
        rawObjectMeasurement: ObjectBBoxMeasurement?
    ) -> DepthMetricCalibration {
        let exifFocalPx = exifFocalPixels(from: cameraMetadata, imageWidth: imageWidth, imageHeight: imageHeight)
        var focalPx = geoFocalPx
        var depthScale: Float = 1.0
        var sourceLabel = "unchanged"
        var sourceCode = 0

        let anchorClassIdx = rawObjectMeasurement?.classIdx
        let anchorExpectedHeight = anchorClassIdx.flatMap { objectAnchorHeightMeters[$0] }
        let anchorMeasuredHeight = rawObjectMeasurement?.height
        let anchorDepthScale: Float? = {
            guard let expected = anchorExpectedHeight,
                  let measured = anchorMeasuredHeight,
                  measured > 0.2 else {
                return nil
            }
            return (expected / measured).clamped(to: depthMetricScaleRange)
        }()

        if let exifFocalPx, exifFocalPx > 1 {
            let focalRatio = geoFocalPx / exifFocalPx
            if geoExifFocalMatchRatioRange.contains(focalRatio) {
                if let anchorDepthScale {
                    depthScale = anchorDepthScale
                    sourceLabel = "depth_anchor_exif_confirms_focal"
                    sourceCode = 1
                } else {
                    sourceLabel = "exif_confirms_focal_no_anchor"
                    sourceCode = 2
                }
            } else if exifFocalPx > geoFocalPx * geoExifFocalMatchRatioRange.upperBound {
                focalPx = exifFocalPx
                sourceLabel = "exif_focal_override"
                sourceCode = 3
            } else if let anchorDepthScale {
                depthScale = anchorDepthScale
                sourceLabel = "depth_anchor_focal_mismatch"
                sourceCode = 4
            }
        } else if let anchorDepthScale {
            depthScale = anchorDepthScale
            sourceLabel = "depth_anchor_no_exif"
            sourceCode = 5
        }

        return DepthMetricCalibration(
            depthScale: depthScale,
            focalPx: focalPx,
            geoFocalPx: geoFocalPx,
            exifFocalPx: exifFocalPx,
            sourceLabel: sourceLabel,
            sourceCode: sourceCode,
            anchorClassIdx: anchorClassIdx,
            anchorExpectedHeightMeters: anchorExpectedHeight,
            anchorMeasuredHeightMeters: anchorMeasuredHeight
        )
    }

    private static func exifFocalPixels(from metadata: [String: Double]?, imageWidth: Int, imageHeight: Int) -> Float? {
        metadataTrustedFocalDetails(from: metadata, imageWidth: imageWidth, imageHeight: imageHeight)?.fx
    }

    private static func scaleDepthMap(_ depthMap: [[Float]], scale: Float) -> [[Float]] {
        guard abs(scale - 1.0) > 1e-4 else { return depthMap }
        return depthMap.map { row in
            row.map { depth in
                depth.isFinite && depth > 0 ? depth * scale : depth
            }
        }
    }

    private static func selectMeasurementObjectBBox(
        from detections: [FurnitureFitDetection],
        imageWidth: Int,
        imageHeight: Int
    ) -> FurnitureFitDetection? {
        let imageArea = max(1, Float(imageWidth * imageHeight))
        let centerX = Float(imageWidth) * 0.5
        let centerY = Float(imageHeight) * 0.5
        let maxCenterDistance = max(1, (centerX * centerX + centerY * centerY).squareRoot())

        let filtered = detections.filter { detection in
            let area = detection.w * detection.h
            let areaFraction = area / imageArea
            return detection.confidence >= objectBBoxConfidenceThreshold &&
                detection.w >= 8 &&
                detection.h >= 8 &&
                areaFraction >= 0.002 &&
                areaFraction <= 0.85
        }
        let anchorPool = filtered.filter { objectAnchorHeightMeters[$0.classIdx] != nil }
        let pool = anchorPool.isEmpty ? filtered : anchorPool

        return pool.max { lhs, rhs in
            objectBBoxScore(lhs, imageArea: imageArea, centerX: centerX, centerY: centerY, maxCenterDistance: maxCenterDistance) <
                objectBBoxScore(rhs, imageArea: imageArea, centerX: centerX, centerY: centerY, maxCenterDistance: maxCenterDistance)
        }
    }

    private static func objectBBoxScore(
        _ detection: FurnitureFitDetection,
        imageArea: Float,
        centerX: Float,
        centerY: Float,
        maxCenterDistance: Float
    ) -> Float {
        let areaFraction = max(0, min((detection.w * detection.h) / imageArea, 1))
        let dx = detection.x - centerX
        let dy = detection.y - centerY
        let centerDistance = (dx * dx + dy * dy).squareRoot() / maxCenterDistance
        var score = detection.confidence * 0.55 + areaFraction.squareRoot() * 0.45 - centerDistance * 0.10
        if objectAnchorHeightMeters[detection.classIdx] != nil {
            score += 0.20
        }
        return score
    }

    private static func clampedBBox(
        _ detection: FurnitureFitDetection,
        imageWidth: Int,
        imageHeight: Int
    ) -> (leftX: Int, rightX: Int, topY: Int, bottomY: Int) {
        let maxX = max(0, imageWidth - 1)
        let maxY = max(0, imageHeight - 1)
        let leftX = min(max(Int(floor(detection.x - detection.w * 0.5)), 0), maxX)
        let rightX = min(max(Int(ceil(detection.x + detection.w * 0.5)), 0), maxX)
        let topY = min(max(Int(floor(detection.y - detection.h * 0.5)), 0), maxY)
        let bottomY = min(max(Int(ceil(detection.y + detection.h * 0.5)), 0), maxY)
        return (leftX, rightX, topY, bottomY)
    }

    private static func depthPercentile(
        depthMap: [[Float]],
        leftX: Int,
        rightX: Int,
        topY: Int,
        bottomY: Int,
        fraction: Float
    ) -> (value: Float, count: Int)? {
        let imageHeight = depthMap.count
        let imageWidth = depthMap.first?.count ?? 0
        guard imageWidth > 0, imageHeight > 0 else { return nil }
        let x0 = min(max(leftX, 0), imageWidth - 1)
        let x1 = min(max(rightX, 0), imageWidth - 1)
        let y0 = min(max(topY, 0), imageHeight - 1)
        let y1 = min(max(bottomY, 0), imageHeight - 1)
        guard x0 < x1, y0 < y1 else { return nil }

        let maxSpan = max(x1 - x0, y1 - y0)
        let step = max(1, maxSpan / 160)
        var samples: [Float] = []
        samples.reserveCapacity(((y1 - y0) / step + 1) * ((x1 - x0) / step + 1))
        for y in stride(from: y0, through: y1, by: step) {
            for x in stride(from: x0, through: x1, by: step) {
                let depth = depthMap[y][x]
                if depth.isFinite, depth > 0.1, depth < 50 {
                    samples.append(depth)
                }
            }
        }
        guard !samples.isEmpty else { return nil }
        samples.sort()
        return (percentile(sorted: samples, fraction: fraction) ?? samples[samples.count / 2], samples.count)
    }

    private static let objectModelLock = NSLock()
    private static var cachedObjectModel: MLModel?
    private static var objectModelLoadAttempted = false

    /// Compiling the RTMDet mlpackage takes seconds — load once per process.
    private static func loadObjectBBoxModel() -> MLModel? {
        objectModelLock.lock()
        if let cachedObjectModel {
            objectModelLock.unlock()
            return cachedObjectModel
        }
        if objectModelLoadAttempted {
            objectModelLock.unlock()
            return nil
        }
        objectModelLock.unlock()
        let loaded = loadObjectBBoxModelUncached()
        objectModelLock.lock()
        cachedObjectModel = loaded
        objectModelLoadAttempted = true
        objectModelLock.unlock()
        return loaded
    }

    private static func loadObjectBBoxModelUncached() -> MLModel? {
        let candidates: [(name: String, ext: String)] = [
            ("rtmdet-ins-m", "mlmodelc"),
            ("rtmdet-ins-m", "mlpackage"),
            ("rtmdet_ins_m", "mlmodelc"),
            ("rtmdet_ins_m", "mlpackage"),
            ("rtmdet-ins-m-coreml", "mlmodelc"),
            ("rtmdet-ins-m-coreml", "mlpackage"),
        ]
        let subdirectories: [String?] = [
            nil,
            "Models/RTMDet",
            "Furnit/Models/RTMDet",
        ]
        let computeUnitFallbacks: [MLComputeUnits] = [
            .cpuAndNeuralEngine,
            .cpuAndGPU,
            .all,
            .cpuOnly,
        ]
        for computeUnits in computeUnitFallbacks {
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits
            for subdirectory in subdirectories {
                for candidate in candidates {
                    guard let url = Bundle.main.url(
                        forResource: candidate.name,
                        withExtension: candidate.ext,
                        subdirectory: subdirectory
                    ) else {
                        continue
                    }
                    do {
                        let loadURL = candidate.ext == "mlpackage"
                            ? try MLModel.compileModel(at: url)
                            : url
                        return try MLModel(contentsOf: loadURL, configuration: config)
                    } catch {
                        logDebug("[DepthAnythingRoom][ObjectBBoxDims] rtmdet_load_failed compute=\(computeUnits) file=\(candidate.name).\(candidate.ext) error=\(error.localizedDescription)")
                    }
                }
            }
        }
        return nil
    }

    private static func measureWallSampleRect(
        imageWidth: Int,
        imageHeight: Int,
        wallMargin: Float
    ) -> (leftX: Int, rightX: Int, topY: Int, bottomY: Int, sampleCenterX: Int, sampleCenterY: Int) {
        let margin = min(max(wallMargin, 0), 0.45)
        let rectX = margin * Float(imageWidth)
        let rectY = margin * Float(imageHeight)
        let rectWidth = (1.0 - 2.0 * margin) * Float(imageWidth)
        let rectHeight = (1.0 - 2.0 * margin) * Float(imageHeight)
        return (
            Int(round(rectX)),
            Int(round(rectX + rectWidth - 1)),
            Int(round(rectY)),
            Int(round(rectY + rectHeight - 1)),
            Int(round(rectX + rectWidth * 0.5)),
            Int(round(rectY + rectHeight * 0.5))
        )
    }

    private struct DepthMapStats {
        let validCount: Int
        let invalidCount: Int
        let min: Float?
        let p05: Float?
        let median: Float?
        let p95: Float?
        let max: Float?
        let centerDepth: Float?
    }

    private struct DepthValueStats {
        let validCount: Int
        let invalidCount: Int
        let min: Float?
        let median: Float?
        let max: Float?
    }

    private static func depthValueStats(_ values: [Float]) -> DepthValueStats {
        var validDepths: [Float] = []
        validDepths.reserveCapacity(values.count)
        var invalidCount = 0
        for depth in values {
            if depth.isFinite, depth > 0 {
                validDepths.append(depth)
            } else {
                invalidCount += 1
            }
        }
        guard !validDepths.isEmpty else {
            return DepthValueStats(
                validCount: 0,
                invalidCount: invalidCount,
                min: nil,
                median: nil,
                max: nil
            )
        }
        validDepths.sort()
        return DepthValueStats(
            validCount: validDepths.count,
            invalidCount: invalidCount,
            min: validDepths.first,
            median: percentile(sorted: validDepths, fraction: 0.5),
            max: validDepths.last
        )
    }

    private static func depthMapStats(_ depthMap: [[Float]]) -> DepthMapStats {
        let imageHeight = depthMap.count
        let imageWidth = depthMap.first?.count ?? 0
        var validDepths: [Float] = []
        validDepths.reserveCapacity(imageHeight * max(imageWidth, 0))
        var invalidCount = 0
        for row in depthMap {
            for depth in row {
                if depth.isFinite, depth > 0 {
                    validDepths.append(depth)
                } else {
                    invalidCount += 1
                }
            }
        }
        guard !validDepths.isEmpty else {
            return DepthMapStats(
                validCount: 0,
                invalidCount: invalidCount,
                min: nil,
                p05: nil,
                median: nil,
                p95: nil,
                max: nil,
                centerDepth: nil
            )
        }
        validDepths.sort()
        let centerDepth: Float?
        if imageWidth > 0, imageHeight > 0 {
            centerDepth = medianAt(
                depthMap: depthMap,
                x: imageWidth / 2,
                y: imageHeight / 2,
                imageWidth: imageWidth,
                imageHeight: imageHeight
            )
        } else {
            centerDepth = nil
        }
        return DepthMapStats(
            validCount: validDepths.count,
            invalidCount: invalidCount,
            min: validDepths.first,
            p05: percentile(sorted: validDepths, fraction: 0.05),
            median: percentile(sorted: validDepths, fraction: 0.5),
            p95: percentile(sorted: validDepths, fraction: 0.95),
            max: validDepths.last,
            centerDepth: centerDepth
        )
    }

    private static func percentile(sorted values: [Float], fraction: Float) -> Float? {
        guard !values.isEmpty else { return nil }
        let clamped = min(max(fraction, 0), 1)
        let index = Int(round(clamped * Float(values.count - 1)))
        return values[index]
    }

    private static func formatMeters(_ value: Float?) -> String {
        guard let value, value.isFinite else { return "nil" }
        return String(format: "%.4f", value)
    }

    private static func centerDepthFallback(_ depthMap: [[Float]]) -> Float {
        var validDepths: [Float] = []
        for row in depthMap {
            for depth in row where depth.isFinite && depth > 0 {
                validDepths.append(depth)
            }
        }
        return validDepths.isEmpty ? 3.0 : median(validDepths)
    }

    private static func medianAt(
        depthMap: [[Float]],
        x: Int,
        y: Int,
        imageWidth: Int,
        imageHeight: Int,
        radius: Int = 5
    ) -> Float? {
        let clampedX = min(max(x, 0), imageWidth - 1)
        let clampedY = min(max(y, 0), imageHeight - 1)
        var samples: [Float] = []
        let y0 = max(0, clampedY - radius)
        let y1 = min(imageHeight, clampedY + radius + 1)
        let x0 = max(0, clampedX - radius)
        let x1 = min(imageWidth, clampedX + radius + 1)
        for row in y0..<y1 {
            for column in x0..<x1 {
                let depth = depthMap[row][column]
                if depth.isFinite, depth > 0 {
                    samples.append(depth)
                }
            }
        }
        guard !samples.isEmpty else { return nil }
        return median(samples)
    }

    private static func focalPixels(image: UIImage, imageWidth: Int, imageHeight: Int) -> (fx: Float, fy: Float) {
        let details = focalPixelDetails(
            image: image,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            cameraMetadata: nil
        )
        return (details.fx, details.fy)
    }

    private static func focalPixelDetails(
        image: UIImage,
        imageWidth: Int,
        imageHeight: Int,
        cameraMetadata: [String: Double]? = nil
    ) -> (fx: Float, fy: Float, focal35mm: Float, source: String) {
        if let metadataFocal = focalPixelDetailsFromCameraMetadata(
            cameraMetadata,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        ) {
            return metadataFocal
        }

        let focal35mm: Float
        let source: String
        if let exifFocal = focal35mmEquivalent(from: image), exifFocal > 1 {
            focal35mm = exifFocal
            source = "exif"
        } else {
            focal35mm = fallbackFocal35mmEquivalent
            source = "fallback"
        }
        // 35mm-equiv → horizontal FOV on 36mm sensor. Square pixels: same f_px on both axes.
        // Do NOT set fy = fx * H/W — that cancels aspect ratio and forces W == H in measureWall().
        let focalPx = (focal35mm / 36.0) * Float(imageWidth)
        return (focalPx, focalPx, focal35mm, source)
    }

    private static func focalPixelDetailsFromCameraMetadata(
        _ metadata: [String: Double]?,
        imageWidth: Int,
        imageHeight: Int
    ) -> (fx: Float, fy: Float, focal35mm: Float, source: String)? {
        guard let metadata, imageWidth > 1, imageHeight > 1 else { return nil }

        if let trustedFocal = metadataTrustedFocalDetails(from: metadata, imageWidth: imageWidth, imageHeight: imageHeight) {
            return trustedFocal
        }

        let targetWidth = Float(imageWidth)
        if let geoFocal = metadataFloat(metadata, keys: ["geoCalibFocalLengthPx", "geocalibFocalLengthPx"]),
           geoFocal > 1 {
            let geoWidth = metadataFloat(metadata, keys: ["geoCalibImageWidthPx", "geocalibImageWidthPx"]) ?? targetWidth
            let focalPx = geoFocal * targetWidth / max(1, geoWidth)
            if focalPx.isFinite, focalPx > 1 {
                return (focalPx, focalPx, 36.0 * focalPx / targetWidth, "geocalib_square_focal")
            }
        }

        return nil
    }

    private static func metadataTrustedFocalDetails(
        from metadata: [String: Double]?,
        imageWidth: Int,
        imageHeight: Int
    ) -> (fx: Float, fy: Float, focal35mm: Float, source: String)? {
        guard let metadata, imageWidth > 1, imageHeight > 1 else { return nil }

        let targetWidth = Float(imageWidth)
        let targetHeight = Float(imageHeight)
        let sourceWidth = metadataFloat(
            metadata,
            keys: ["imageWidthPx", "exifPixelXDimension", "sourceImageWidthPx"]
        )
        let sourceHeight = metadataFloat(
            metadata,
            keys: ["imageHeightPx", "exifPixelYDimension", "sourceImageHeightPx"]
        )

        if let sourceFx = metadataFloat(
            metadata,
            keys: ["focalLengthPx", "arkitFocalLengthXPx", "predictedFocalLengthPx"]
        ),
           sourceFx > 1 {
            let sourceFy = metadataFloat(
                metadata,
                keys: ["arkitFocalLengthYPx", "focalLengthYPx", "predictedFocalLengthYPx"]
            ) ?? sourceFx

            if let sourceWidth,
               let sourceHeight,
               sourceWidth > 1,
               sourceHeight > 1 {
                let normalError = abs(sourceWidth - targetWidth) + abs(sourceHeight - targetHeight)
                let rotatedError = abs(sourceWidth - targetHeight) + abs(sourceHeight - targetWidth)
                let rotated = rotatedError < normalError

                let fx: Float
                let fy: Float
                let source: String
                if rotated {
                    fx = sourceFy * targetWidth / sourceHeight
                    fy = sourceFx * targetHeight / sourceWidth
                    source = "sidecar_focal_px_rotated_scaled"
                } else {
                    fx = sourceFx * targetWidth / sourceWidth
                    fy = sourceFy * targetHeight / sourceHeight
                    source = "sidecar_focal_px_scaled"
                }

                if fx.isFinite, fy.isFinite, fx > 1, fy > 1 {
                    return (fx, fy, 36.0 * fx / targetWidth, source)
                }
            }
        }

        if let focal35mm = metadataFloat(
            metadata,
            keys: ["focalLength35mmEquivMm", "focalLength35mmEquivalentMm", "focalLength35mmEquivalentMM"]
        ), focal35mm > 1 {
            let focalPx = (focal35mm / 36.0) * targetWidth
            return (focalPx, focalPx, focal35mm, "sidecar_35mm")
        }

        return nil
    }

    private static func mergedCameraMetadata(
        _ first: [String: Double]?,
        _ second: [String: Double]?
    ) -> [String: Double]? {
        var merged = first ?? [:]
        if let second {
            for (key, value) in second {
                merged[key] = value
            }
        }
        return merged.isEmpty ? nil : merged
    }

    private static func metadataFloat(_ metadata: [String: Double], keys: [String]) -> Float? {
        for key in keys {
            guard let value = metadata[key] else { continue }
            let floatValue = Float(value)
            if floatValue.isFinite, floatValue > 0 {
                return floatValue
            }
        }
        return nil
    }

    private static func focal35mmEquivalent(from image: UIImage) -> Float? {
        guard let data = image.jpegData(compressionQuality: 0.95) else { return nil }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any] else {
            return nil
        }
        if let value = exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? NSNumber, value.floatValue > 1 {
            return value.floatValue
        }
        if let value = exif["FocalLenIn35mmFilm" as CFString] as? NSNumber, value.floatValue > 1 {
            return value.floatValue
        }
        return nil
    }

    private static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) * 0.5
        }
        return sorted[middle]
    }
}

private enum DepthAnythingVertex {
    static let byteStride = 8 * MemoryLayout<Float>.size
}

private struct DenseDepthArray {
    let shape: [Int]
    let values: [Float]
}

private struct DenseDepthGrid {
    let width: Int
    let height: Int
    let values: [Float]
}

private struct DepthAnythingRasterImage {
    let width: Int
    let height: Int
    let pixels: [UInt8]

    init(image: UIImage) throws {
        guard let cgImage = image.cgImage else {
            throw DepthAnythingRoomError.invalidImage
        }
        width = cgImage.width
        height = cgImage.height
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw DepthAnythingRoomError.invalidImage
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        pixels = rgba
    }

    func color(x: Int, y: Int) -> DepthAnythingRGB {
        let clampedX = min(max(x, 0), width - 1)
        let clampedY = min(max(y, 0), height - 1)
        let offset = (clampedY * width + clampedX) * 4
        return DepthAnythingRGB(r: pixels[offset], g: pixels[offset + 1], b: pixels[offset + 2])
    }
}

private struct DepthAnythingRGB {
    let r: UInt8
    let g: UInt8
    let b: UInt8

    var floatRGB: SIMD3<Float> {
        SIMD3(Float(r) / 255, Float(g) / 255, Float(b) / 255)
    }
}

enum DepthAnythingRoomError: LocalizedError, Equatable {
    case modelNotFound
    case invalidImage
    case invalidDepthOutput
    case depthImageSizeMismatch
    case emptyMesh
    case exportFailed(String? = nil)

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "DepthAnythingV2MetricIndoorSmall was not found in the app bundle. Add the .mlpackage or .mlmodelc to the iOS target."
        case .invalidImage:
            return "The selected image could not be read."
        case .invalidDepthOutput:
            return "Depth Anything returned an invalid depth output."
        case .depthImageSizeMismatch:
            return "Depth map size does not match the source image."
        case .emptyMesh:
            return "Depth Anything did not produce enough connected room geometry."
        case .exportFailed(let reason):
            if let reason, !reason.isEmpty {
                return "Could not export the reconstructed room as USDZ. \(reason)"
            }
            return "Could not export the reconstructed room as USDZ."
        }
    }
}

private extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

private extension Data {
    mutating func appendFloat32LE(_ value: Float) {
        var bits = value.bitPattern.littleEndian
        Swift.withUnsafeBytes(of: &bits) { append(contentsOf: $0) }
    }

    mutating func appendUInt32LE(_ value: UInt32) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }
}
