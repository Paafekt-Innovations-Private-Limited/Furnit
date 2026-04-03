import Foundation
import simd

public struct RoomGeometryConfig: Sendable, Equatable {
    public var gridRows: Int = 56
    public var gridCols: Int = 56
    public var ransacIterations: Int = 160
    public var ransacInlierThreshold: Float = 0.05
    public var maxWallPlanes: Int = 4
    public var minPlaneConfidence: Float = 0.08
    public var standardCeilingHeightM: Float = 2.4
    public var maxPointDistance: Float = 12.0
    public var minPointCount: Int = 80
    public var minPlaneInlierCount: Int = 18
    public var wallCandidateMinHeightFraction: Float = 0.14
    public var wallCandidateMaxHeightFraction: Float = 0.92

    public static let `default` = RoomGeometryConfig()
}

public enum GeometryExtractionError: LocalizedError {
    case insufficientPoints(count: Int)
    case floorDetectionFailed

    public var errorDescription: String? {
        switch self {
        case .insufficientPoints(let count):
            return "Insufficient splat depth points for room extraction (\(count))."
        case .floorDetectionFailed:
            return "Unable to detect a stable floor plane."
        }
    }
}

public final class RoomGeometryEngine {
    private let depthQuery: SplatDepthQueryable
    private let colorReader: SplatColorReadable?
    private let cameraInfo: SourceCameraInfo
    private let config: RoomGeometryConfig

    public init(
        depthQuery: SplatDepthQueryable,
        colorReader: SplatColorReadable? = nil,
        cameraInfo: SourceCameraInfo,
        config: RoomGeometryConfig = .default
    ) {
        self.depthQuery = depthQuery
        self.colorReader = colorReader
        self.cameraInfo = cameraInfo
        self.config = config
    }

    public func extractRoomModel() throws -> RoomModel {
        let t0 = CFAbsoluteTimeGetCurrent()
        let points = depthQuery.buildPointCloud(
            rows: config.gridRows,
            cols: config.gridCols,
            maxDistance: config.maxPointDistance
        )
        logDebug("📐 [RoomGeometryEngine] pointCloud count=\(points.count) grid=\(config.gridRows)x\(config.gridCols)")
        guard points.count >= config.minPointCount else {
            throw GeometryExtractionError.insufficientPoints(count: points.count)
        }

        let roomBounds = computeAABB(points: points)
        let floor = try extractFloorPlane(from: points)
        let ceiling = extractCeilingPlane(from: points, floor: floor)
        let walls = extractWallPlanes(from: points, floor: floor)
        let corners = findCorners(walls: walls, floor: floor, roomBounds: roomBounds)
        let sceneToMeters = estimateSceneToMeters(floor: floor, ceiling: ceiling, bounds: roomBounds)
        let freeRegions = computeFreespace(points: points, floor: floor, sceneToMeters: sceneToMeters)

        let baseModel = RoomModel(
            floorPlane: floor,
            ceilingPlane: ceiling,
            wallPlanes: walls,
            corners: corners,
            freeFloorRegions: freeRegions,
            roomBounds: roomBounds,
            sceneToMeters: sceneToMeters,
            surfacePalette: SurfacePalette(floor: nil, walls: nil, ceiling: nil),
            cameraInfo: cameraInfo,
            extractedAt: Date()
        )

        let palette: SurfacePalette
        if let colorReader, colorReader.supportsColorReadback {
            palette = TextureSampler(depthQuery: depthQuery, colorReader: colorReader, roomModel: baseModel)
                .samplePalette(gridResolution: 20)
        } else {
            palette = SurfacePalette(floor: nil, walls: nil, ceiling: nil)
        }

        let model = RoomModel(
            floorPlane: floor,
            ceilingPlane: ceiling,
            wallPlanes: walls,
            corners: corners,
            freeFloorRegions: freeRegions,
            roomBounds: roomBounds,
            sceneToMeters: sceneToMeters,
            surfacePalette: palette,
            cameraInfo: cameraInfo,
            extractedAt: baseModel.extractedAt
        )

        logDebug(
            "📐 [RoomGeometryEngine] done walls=\(walls.count) corners=\(corners.count) freeRegions=\(freeRegions.count) " +
            "sceneToMeters=\(String(format: "%.4f", sceneToMeters)) elapsed=\(String(format: "%.3f", CFAbsoluteTimeGetCurrent() - t0))s"
        )
        return model
    }

    public func extractFloorPlane(from points: [SIMD3<Float>]) throws -> DetectedPlane {
        let planeCandidates = candidatePointsForHorizontalPlane(from: points, upper: false)
        let best = ransacFitPlane(points: planeCandidates, expected: .horizontal, preferPositiveY: true)
        guard let best else { throw GeometryExtractionError.floorDetectionFailed }
        return DetectedPlane(
            normal: best.normal.y >= 0 ? best.normal : -best.normal,
            centroid: best.centroid,
            d: best.normal.y >= 0 ? best.d : -best.d,
            type: .floor,
            confidence: best.confidence,
            bounds: best.bounds
        )
    }

    public func extractCeilingPlane(from points: [SIMD3<Float>], floor: DetectedPlane) -> DetectedPlane? {
        let planeCandidates = candidatePointsForHorizontalPlane(from: points, upper: true)
        guard var ceiling = ransacFitPlane(points: planeCandidates, expected: .horizontal, preferPositiveY: false) else {
            return nil
        }
        if ceiling.normal.y > 0 {
            ceiling = PlaneFit(
                normal: -ceiling.normal,
                centroid: ceiling.centroid,
                d: -ceiling.d,
                confidence: ceiling.confidence,
                bounds: ceiling.bounds,
                inlierCount: ceiling.inlierCount,
                rmsError: ceiling.rmsError
            )
        }
        guard ceiling.centroid.y > floor.centroid.y + config.ransacInlierThreshold else { return nil }
        return DetectedPlane(
            normal: ceiling.normal,
            centroid: ceiling.centroid,
            d: ceiling.d,
            type: .ceiling,
            confidence: ceiling.confidence,
            bounds: ceiling.bounds
        )
    }

    public func extractWallPlanes(from points: [SIMD3<Float>], floor: DetectedPlane) -> [DetectedPlane] {
        let floorHeightBand = wallCandidateHeightBand(points: points, floor: floor)
        let floorFiltered = points.filter {
            let floorDistance = dot($0 - floor.centroid, floor.normal)
            guard floorDistance > floorHeightBand.minHeight,
                  floorDistance < floorHeightBand.maxHeight else { return false }
            return abs(floor.distance(to: $0)) > config.ransacInlierThreshold * 1.5
        }
        var remaining = floorFiltered
        var walls: [DetectedPlane] = []

        while walls.count < config.maxWallPlanes, remaining.count >= config.minPointCount / 2 {
            guard let fit = ransacFitPlane(points: remaining, expected: .vertical, preferPositiveY: nil),
                  fit.confidence >= config.minPlaneConfidence else { break }

            let plane = DetectedPlane(
                normal: fit.normal,
                centroid: fit.centroid,
                d: fit.d,
                type: .wall,
                confidence: fit.confidence,
                bounds: fit.bounds
            )

            if walls.contains(where: { abs(dot($0.normal, plane.normal)) > 0.94 && abs($0.d - plane.d) < config.ransacInlierThreshold * 2 }) {
                remaining.removeAll { abs(plane.distance(to: $0)) < config.ransacInlierThreshold }
                continue
            }

            walls.append(plane)
            remaining.removeAll { abs(plane.distance(to: $0)) < config.ransacInlierThreshold }
        }

        logDebug("📐 [RoomGeometryEngine] wallPlanes detected=\(walls.count)")
        return walls.sorted { $0.confidence > $1.confidence }
    }

    public func findCorners(
        walls: [DetectedPlane],
        floor: DetectedPlane,
        roomBounds: AABB3
    ) -> [RoomCorner] {
        var corners: [RoomCorner] = []

        for i in 0..<walls.count {
            for j in (i + 1)..<walls.count {
                let first = walls[i]
                let second = walls[j]
                let orthogonality = abs(dot(first.normal, second.normal))
                guard orthogonality < 0.75 else { continue }
                guard let basePoint = solvePlaneIntersectionPoint(first, second) else { continue }

                let direction = cross(first.normal, second.normal)
                let denom = dot(floor.normal, direction)
                guard abs(denom) > 1e-6 else { continue }

                let t = -(dot(floor.normal, basePoint) + floor.d) / denom
                let point = basePoint + direction * t
                guard roomBounds.expanded(by: config.ransacInlierThreshold * 4).contains(point) else { continue }

                let bisector = normalizeOrZero(first.normal + second.normal)
                corners.append(
                    RoomCorner(
                        position: point,
                        wallIndexA: i,
                        wallIndexB: j,
                        bisector: bisector
                    )
                )
            }
        }

        logDebug("📐 [RoomGeometryEngine] corners detected=\(corners.count)")
        return deduplicatedCorners(corners)
    }

    public func computeFreespace(
        points: [SIMD3<Float>],
        floor: DetectedPlane,
        sceneToMeters: Float
    ) -> [FreeFloorRegion] {
        let basis = makePlaneBasis(normal: floor.normal)
        let floorPoints = points
            .filter { abs(floor.distance(to: $0)) < config.ransacInlierThreshold * 1.5 }
            .map { projectToPlaneUV($0, origin: floor.centroid, u: basis.u, v: basis.v) }

        guard !floorPoints.isEmpty else { return [] }

        var minUV = floorPoints[0]
        var maxUV = floorPoints[0]
        for point in floorPoints.dropFirst() {
            minUV = simd_min(minUV, point)
            maxUV = simd_max(maxUV, point)
        }

        let polygon = [
            SIMD2<Float>(minUV.x, minUV.y),
            SIMD2<Float>(maxUV.x, minUV.y),
            SIMD2<Float>(maxUV.x, maxUV.y),
            SIMD2<Float>(minUV.x, maxUV.y)
        ]
        let sceneArea = max(0, (maxUV.x - minUV.x) * (maxUV.y - minUV.y))
        let areaSqM = sceneArea * sceneToMeters * sceneToMeters

        let region = FreeFloorRegion(
            polygon: polygon,
            areaSqM: areaSqM,
            uvBounds: FloorUVBounds(min: minUV, max: maxUV)
        )
        logDebug("📐 [RoomGeometryEngine] freeRegion sceneArea=\(String(format: "%.3f", sceneArea)) areaSqM=\(String(format: "%.3f", areaSqM))")
        return [region]
    }

    public func estimateSceneToMeters(
        floor: DetectedPlane?,
        ceiling: DetectedPlane?,
        bounds: AABB3
    ) -> Float {
        if let floor, let ceiling {
            let sceneHeight = abs(ceiling.distance(to: floor.centroid))
            if sceneHeight > 0.001 {
                let scale = config.standardCeilingHeightM / sceneHeight
                logDebug("📐 [RoomGeometryEngine] sceneToMeters via floor/ceiling=\(String(format: "%.4f", scale))")
                return scale
            }
        }

        if let subjectDistance = cameraInfo.subjectDistanceMeters,
           subjectDistance > 0.1,
           bounds.size.z > 0.05 {
            let scale = subjectDistance / max(bounds.size.z, 0.001)
            logDebug("📐 [RoomGeometryEngine] sceneToMeters via subjectDistance=\(String(format: "%.4f", scale))")
            return scale
        }

        let fallback = config.standardCeilingHeightM / max(bounds.size.y, 0.001)
        logDebug("📐 [RoomGeometryEngine] sceneToMeters fallback=\(String(format: "%.4f", fallback))")
        return fallback
    }

    private enum PlaneExpectation {
        case horizontal
        case vertical
    }

    private struct PlaneFit {
        let normal: SIMD3<Float>
        let centroid: SIMD3<Float>
        let d: Float
        let confidence: Float
        let bounds: AABB3
        let inlierCount: Int
        let rmsError: Float
    }

    private func candidatePointsForHorizontalPlane(from points: [SIMD3<Float>], upper: Bool) -> [SIMD3<Float>] {
        guard !points.isEmpty else { return [] }
        let sortedY = points.map(\.y).sorted()
        let pivotIndex = upper ? (sortedY.count * 2 / 3) : (sortedY.count / 3)
        let pivot = sortedY[min(max(pivotIndex, 0), sortedY.count - 1)]
        return upper ? points.filter { $0.y >= pivot } : points.filter { $0.y <= pivot }
    }

    private func ransacFitPlane(
        points: [SIMD3<Float>],
        expected: PlaneExpectation,
        preferPositiveY: Bool?
    ) -> PlaneFit? {
        guard points.count >= 3 else { return nil }

        var bestFit: PlaneFit?
        var bestQuality: Float = -.greatestFiniteMagnitude
        for _ in 0..<config.ransacIterations {
            let sample = randomPlaneSample(from: points)
            let rawNormal = cross(sample.1 - sample.0, sample.2 - sample.0)
            guard length_squared(rawNormal) > 1e-8 else { continue }

            var normal = normalize(rawNormal)
            switch expected {
            case .horizontal:
                guard abs(normal.y) > 0.72 else { continue }
            case .vertical:
                guard abs(normal.y) < 0.32 else { continue }
            }

            if let preferPositiveY {
                if preferPositiveY, normal.y < 0 { normal = -normal }
                if !preferPositiveY, normal.y > 0 { normal = -normal }
            }

            let d = -dot(normal, sample.0)
            let inliers = points.filter { abs(dot(normal, $0) + d) < config.ransacInlierThreshold }
            guard inliers.count >= config.minPlaneInlierCount else { continue }
            guard let fit = refinePlaneFit(
                inliers: inliers,
                originalPoints: points,
                expected: expected,
                preferPositiveY: preferPositiveY
            ) else { continue }

            let alignmentScore: Float
            switch expected {
            case .horizontal:
                alignmentScore = abs(fit.normal.y)
            case .vertical:
                alignmentScore = 1 - min(abs(fit.normal.y), 1)
            }
            let quality = fit.confidence + alignmentScore * 0.05 - min(fit.rmsError / max(config.ransacInlierThreshold, 0.001), 1) * 0.08
            if bestFit == nil || quality > bestQuality {
                bestFit = fit
                bestQuality = quality
            }
        }

        return bestFit
    }

    private func refinePlaneFit(
        inliers: [SIMD3<Float>],
        originalPoints: [SIMD3<Float>],
        expected: PlaneExpectation,
        preferPositiveY: Bool?
    ) -> PlaneFit? {
        guard inliers.count >= config.minPlaneInlierCount else { return nil }
        let centroid = inliers.reduce(.zero, +) / Float(inliers.count)
        let covariance = covarianceMatrix(points: inliers, centroid: centroid)
        guard var normal = smallestEigenvector(ofSymmetric3x3: covariance) else { return nil }
        normal = normalizeOrZero(normal)
        guard length_squared(normal) > 1e-8 else { return nil }

        switch expected {
        case .horizontal:
            guard abs(normal.y) > 0.72 else { return nil }
        case .vertical:
            guard abs(normal.y) < 0.35 else { return nil }
        }

        if let preferPositiveY {
            if preferPositiveY, normal.y < 0 { normal = -normal }
            if !preferPositiveY, normal.y > 0 { normal = -normal }
        }

        let d = -dot(normal, centroid)
        let refinedInliers = originalPoints.filter { abs(dot(normal, $0) + d) < config.ransacInlierThreshold }
        guard refinedInliers.count >= config.minPlaneInlierCount else { return nil }

        let refinedCentroid = refinedInliers.reduce(.zero, +) / Float(refinedInliers.count)
        let refinedCovariance = covarianceMatrix(points: refinedInliers, centroid: refinedCentroid)
        guard var refinedNormal = smallestEigenvector(ofSymmetric3x3: refinedCovariance) else { return nil }
        refinedNormal = normalizeOrZero(refinedNormal)
        guard length_squared(refinedNormal) > 1e-8 else { return nil }

        switch expected {
        case .horizontal:
            guard abs(refinedNormal.y) > 0.72 else { return nil }
        case .vertical:
            guard abs(refinedNormal.y) < 0.35 else { return nil }
        }

        if let preferPositiveY {
            if preferPositiveY, refinedNormal.y < 0 { refinedNormal = -refinedNormal }
            if !preferPositiveY, refinedNormal.y > 0 { refinedNormal = -refinedNormal }
        }

        let refinedD = -dot(refinedNormal, refinedCentroid)
        let rms = rmsPlaneError(points: refinedInliers, normal: refinedNormal, d: refinedD)
        return PlaneFit(
            normal: refinedNormal,
            centroid: refinedCentroid,
            d: refinedD,
            confidence: Float(refinedInliers.count) / Float(max(originalPoints.count, 1)),
            bounds: computeAABB(points: refinedInliers),
            inlierCount: refinedInliers.count,
            rmsError: rms
        )
    }

    private func randomPlaneSample(from points: [SIMD3<Float>]) -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) {
        let i0 = Int.random(in: 0..<points.count)
        var i1 = Int.random(in: 0..<points.count)
        var i2 = Int.random(in: 0..<points.count)
        while i1 == i0 { i1 = Int.random(in: 0..<points.count) }
        while i2 == i0 || i2 == i1 { i2 = Int.random(in: 0..<points.count) }
        return (points[i0], points[i1], points[i2])
    }

    private func deduplicatedCorners(_ corners: [RoomCorner]) -> [RoomCorner] {
        var result: [RoomCorner] = []
        for corner in corners {
            if result.contains(where: { simd_distance($0.position, corner.position) < config.ransacInlierThreshold * 2 }) {
                continue
            }
            result.append(corner)
        }
        return result
    }

    private func wallCandidateHeightBand(
        points: [SIMD3<Float>],
        floor: DetectedPlane
    ) -> (minHeight: Float, maxHeight: Float) {
        let positiveHeights = points
            .map { dot($0 - floor.centroid, floor.normal) }
            .filter { $0.isFinite && $0 > config.ransacInlierThreshold }
            .sorted()

        guard let maxObserved = positiveHeights.last else {
            return (config.ransacInlierThreshold * 2, .greatestFiniteMagnitude)
        }

        let minHeight = max(
            config.ransacInlierThreshold * 2,
            maxObserved * config.wallCandidateMinHeightFraction
        )
        let maxHeight = max(
            minHeight + config.ransacInlierThreshold,
            maxObserved * config.wallCandidateMaxHeightFraction
        )
        return (minHeight, maxHeight)
    }

    private func solvePlaneIntersectionPoint(_ p1: DetectedPlane, _ p2: DetectedPlane) -> SIMD3<Float>? {
        let a = SIMD2<Float>(p1.normal.x, p1.normal.z)
        let b = SIMD2<Float>(p2.normal.x, p2.normal.z)
        let determinant = a.x * b.y - a.y * b.x
        guard abs(determinant) > 1e-6 else { return nil }
        let rhs = SIMD2<Float>(-p1.d, -p2.d)
        let x = (rhs.x * b.y - rhs.y * a.y) / determinant
        let z = (a.x * rhs.y - b.x * rhs.x) / determinant
        return SIMD3<Float>(x, 0, z)
    }

    private func computeAABB(points: [SIMD3<Float>]) -> AABB3 {
        guard let first = points.first else {
            return AABB3(min: .zero, max: .zero)
        }
        var minPoint = first
        var maxPoint = first
        for point in points.dropFirst() {
            minPoint = simd_min(minPoint, point)
            maxPoint = simd_max(maxPoint, point)
        }
        return AABB3(min: minPoint, max: maxPoint)
    }

    private func makePlaneBasis(normal: SIMD3<Float>) -> (u: SIMD3<Float>, v: SIMD3<Float>) {
        let helper = abs(normal.y) > 0.9 ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(0, 1, 0)
        let u = normalizeOrZero(cross(helper, normal))
        let v = normalizeOrZero(cross(normal, u))
        return (u, v)
    }

    private func projectToPlaneUV(
        _ point: SIMD3<Float>,
        origin: SIMD3<Float>,
        u: SIMD3<Float>,
        v: SIMD3<Float>
    ) -> SIMD2<Float> {
        let delta = point - origin
        return SIMD2<Float>(dot(delta, u), dot(delta, v))
    }

    private func normalizeOrZero(_ vector: SIMD3<Float>) -> SIMD3<Float> {
        let lengthSquared = simd_length_squared(vector)
        guard lengthSquared > 1e-8 else { return .zero }
        return vector / sqrt(lengthSquared)
    }

    private func covarianceMatrix(points: [SIMD3<Float>], centroid: SIMD3<Float>) -> simd_float3x3 {
        var covariance = simd_float3x3()
        for point in points {
            let delta = point - centroid
            matrixSet(&covariance, row: 0, col: 0, value: matrixGet(covariance, row: 0, col: 0) + delta.x * delta.x)
            matrixSet(&covariance, row: 0, col: 1, value: matrixGet(covariance, row: 0, col: 1) + delta.x * delta.y)
            matrixSet(&covariance, row: 0, col: 2, value: matrixGet(covariance, row: 0, col: 2) + delta.x * delta.z)
            matrixSet(&covariance, row: 1, col: 0, value: matrixGet(covariance, row: 1, col: 0) + delta.y * delta.x)
            matrixSet(&covariance, row: 1, col: 1, value: matrixGet(covariance, row: 1, col: 1) + delta.y * delta.y)
            matrixSet(&covariance, row: 1, col: 2, value: matrixGet(covariance, row: 1, col: 2) + delta.y * delta.z)
            matrixSet(&covariance, row: 2, col: 0, value: matrixGet(covariance, row: 2, col: 0) + delta.z * delta.x)
            matrixSet(&covariance, row: 2, col: 1, value: matrixGet(covariance, row: 2, col: 1) + delta.z * delta.y)
            matrixSet(&covariance, row: 2, col: 2, value: matrixGet(covariance, row: 2, col: 2) + delta.z * delta.z)
        }
        let scale = 1 / Float(max(points.count, 1))
        return covariance * scale
    }

    private func rmsPlaneError(points: [SIMD3<Float>], normal: SIMD3<Float>, d: Float) -> Float {
        guard !points.isEmpty else { return .greatestFiniteMagnitude }
        let sum = points.reduce(Float.zero) { partial, point in
            let error = dot(normal, point) + d
            return partial + error * error
        }
        return sqrt(sum / Float(points.count))
    }

    private func smallestEigenvector(ofSymmetric3x3 matrix: simd_float3x3) -> SIMD3<Float>? {
        var a = matrix
        var v = matrix_identity_float3x3

        for _ in 0..<10 {
            let offDiag = [
                (0, 1, abs(matrixGet(a, row: 0, col: 1))),
                (0, 2, abs(matrixGet(a, row: 0, col: 2))),
                (1, 2, abs(matrixGet(a, row: 1, col: 2)))
            ]
            guard let pivot = offDiag.max(by: { $0.2 < $1.2 }), pivot.2 > 1e-7 else { break }
            jacobiRotate(matrix: &a, eigenvectors: &v, p: pivot.0, q: pivot.1)
        }

        let eigenvalues = [
            matrixGet(a, row: 0, col: 0),
            matrixGet(a, row: 1, col: 1),
            matrixGet(a, row: 2, col: 2)
        ]
        guard let minIndex = eigenvalues.enumerated().min(by: { $0.element < $1.element })?.offset else {
            return nil
        }
        let vector = SIMD3<Float>(
            matrixGet(v, row: 0, col: minIndex),
            matrixGet(v, row: 1, col: minIndex),
            matrixGet(v, row: 2, col: minIndex)
        )
        return length_squared(vector) > 1e-8 ? vector : nil
    }

    private func jacobiRotate(
        matrix a: inout simd_float3x3,
        eigenvectors v: inout simd_float3x3,
        p: Int,
        q: Int
    ) {
        guard p != q else { return }
        let app = matrixGet(a, row: p, col: p)
        let aqq = matrixGet(a, row: q, col: q)
        let apq = matrixGet(a, row: p, col: q)
        guard abs(apq) > 1e-8 else { return }

        let tau = (aqq - app) / (2 * apq)
        let t: Float = tau >= 0
            ? 1 / (tau + sqrt(1 + tau * tau))
            : -1 / (-tau + sqrt(1 + tau * tau))
        let c = 1 / sqrt(1 + t * t)
        let s = t * c

        for r in 0..<3 where r != p && r != q {
            let arp = matrixGet(a, row: r, col: p)
            let arq = matrixGet(a, row: r, col: q)
            let newRp = c * arp - s * arq
            let newRq = c * arq + s * arp
            matrixSet(&a, row: r, col: p, value: newRp)
            matrixSet(&a, row: p, col: r, value: newRp)
            matrixSet(&a, row: r, col: q, value: newRq)
            matrixSet(&a, row: q, col: r, value: newRq)
        }

        let newApp = c * c * app - 2 * s * c * apq + s * s * aqq
        let newAqq = s * s * app + 2 * s * c * apq + c * c * aqq
        matrixSet(&a, row: p, col: p, value: newApp)
        matrixSet(&a, row: q, col: q, value: newAqq)
        matrixSet(&a, row: p, col: q, value: 0)
        matrixSet(&a, row: q, col: p, value: 0)

        for r in 0..<3 {
            let vrp = matrixGet(v, row: r, col: p)
            let vrq = matrixGet(v, row: r, col: q)
            matrixSet(&v, row: r, col: p, value: c * vrp - s * vrq)
            matrixSet(&v, row: r, col: q, value: s * vrp + c * vrq)
        }
    }

    private func matrixGet(_ matrix: simd_float3x3, row: Int, col: Int) -> Float {
        matrix[col][row]
    }

    private func matrixSet(_ matrix: inout simd_float3x3, row: Int, col: Int, value: Float) {
        matrix[col][row] = value
    }
}

private extension AABB3 {
    func expanded(by margin: Float) -> AABB3 {
        let offset = SIMD3<Float>(repeating: margin)
        return AABB3(min: min - offset, max: max + offset)
    }
}
