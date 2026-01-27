// SpatialReasoningService.swift
// Deterministic spatial reasoning layer for furniture placement
// This is the "source of truth" for geometry - VLM comments on results, doesn't compute them

import Foundation
import simd
import CoreGraphics

// MARK: - Spatial Reasoning Service

/// Service for computing furniture placement candidates
/// Uses SHARP geometry data to generate and score placement options
public class SpatialReasoningService {

    // MARK: - Configuration

    public struct Config {
        /// Minimum clearance from walls (meters)
        public var minWallClearance: Float = 0.05

        /// Grid sampling resolution for candidate generation (meters)
        public var samplingResolution: Float = 0.2

        /// Rotation angles to sample (radians)
        public var rotationSamples: [Float] = [0, .pi / 2, .pi, 3 * .pi / 2]

        /// Maximum candidates to return
        public var maxCandidates: Int = 50

        /// Minimum score to include candidate
        public var minScoreThreshold: Float = 0.3

        public init() {}
    }

    public var config: Config

    // MARK: - Initialization

    public init(config: Config = Config()) {
        self.config = config
    }

    // MARK: - Main API

    /// Generate and score placement candidates for furniture in a room
    /// - Parameters:
    ///   - furniture: The furniture item to place
    ///   - room: Room context with geometry
    /// - Returns: Sorted list of placement candidates (best first)
    public func generateCandidates(
        furniture: FurnitureItem,
        room: RoomContext
    ) -> [PlacementCandidate] {
        var candidates: [PlacementCandidate] = []

        // Strategy 1: Sample along walls (for wall-aligned furniture)
        if furniture.placementHints.preferAgainstWall {
            candidates.append(contentsOf: sampleAlongWalls(furniture: furniture, room: room))
        }

        // Strategy 2: Sample corners (for corner furniture)
        if furniture.placementHints.preferCorner {
            candidates.append(contentsOf: sampleCorners(furniture: furniture, room: room))
        }

        // Strategy 3: Sample center area (for centered furniture)
        if furniture.placementHints.preferCentered {
            candidates.append(contentsOf: sampleCenterArea(furniture: furniture, room: room))
        }

        // Strategy 4: Grid sampling (general fallback)
        candidates.append(contentsOf: sampleGrid(furniture: furniture, room: room))

        // Score all candidates
        for i in 0..<candidates.count {
            candidates[i].scores = scoreCandidate(candidates[i], furniture: furniture, room: room)
            candidates[i].violations = checkConstraints(candidates[i], furniture: furniture, room: room)
        }

        // Filter by minimum score and sort by composite score
        let filtered = candidates
            .filter { $0.compositeScore >= config.minScoreThreshold }
            .sorted { $0.compositeScore > $1.compositeScore }

        // Deduplicate nearby candidates
        let deduplicated = deduplicateCandidates(filtered, minDistance: config.samplingResolution * 2)

        return Array(deduplicated.prefix(config.maxCandidates))
    }

    /// Quick check if furniture fits anywhere in the room
    /// - Returns: (fits, best position) or (false, nil) if no valid placement
    public func quickFitCheck(
        furniture: FurnitureItem,
        room: RoomContext
    ) -> (fits: Bool, bestCandidate: PlacementCandidate?) {
        // Check if furniture is larger than room
        if furniture.width > room.dimensions.width || furniture.depth > room.dimensions.depth {
            return (false, nil)
        }

        // Check if furniture footprint exceeds room floor area
        if furniture.footprintArea > room.dimensions.floorArea * 0.5 {
            // Furniture takes more than 50% of floor - probably doesn't fit well
            return (false, nil)
        }

        // Generate candidates and find best valid one
        let candidates = generateCandidates(furniture: furniture, room: room)
        let validCandidates = candidates.filter { $0.isValid }

        if let best = validCandidates.first {
            return (true, best)
        }

        return (false, nil)
    }

    // MARK: - Sampling Strategies

    private func sampleAlongWalls(furniture: FurnitureItem, room: RoomContext) -> [PlacementCandidate] {
        var candidates: [PlacementCandidate] = []

        for wall in room.walls {
            let wallLength = distance(wall.startPoint, wall.endPoint)
            let wallVector = CGPoint(
                x: wall.endPoint.x - wall.startPoint.x,
                y: wall.endPoint.y - wall.startPoint.y
            )
            let wallAngle = atan2(Float(wallVector.y), Float(wallVector.x))

            // Furniture faces away from wall (back against wall)
            let furnitureYaw = wallAngle + .pi / 2

            // Sample positions along wall
            let numSamples = Int(wallLength / CGFloat(config.samplingResolution))
            for i in 0..<max(1, numSamples) {
                let t = CGFloat(i) / CGFloat(max(1, numSamples - 1))
                let baseX = wall.startPoint.x + t * wallVector.x
                let baseZ = wall.startPoint.y + t * wallVector.y

                // Offset from wall by furniture depth/2 + clearance
                let offset = furniture.depth / 2 + furniture.clearance.back + config.minWallClearance
                let offsetX = baseX + CGFloat(cos(furnitureYaw + .pi) * offset)
                let offsetZ = baseZ + CGFloat(sin(furnitureYaw + .pi) * offset)

                let candidate = PlacementCandidate(
                    id: "wall_\(wall.id)_\(i)",
                    x: Float(offsetX),
                    y: 0,
                    z: Float(offsetZ),
                    yaw: furnitureYaw
                )
                candidates.append(candidate)
            }
        }

        return candidates
    }

    private func sampleCorners(furniture: FurnitureItem, room: RoomContext) -> [PlacementCandidate] {
        var candidates: [PlacementCandidate] = []

        // Find corner positions from floor polygon or walls
        let corners = findCorners(room: room)

        for (i, corner) in corners.enumerated() {
            // Try different rotations at each corner
            for rotation in config.rotationSamples {
                // Offset from corner to fit furniture
                let diagonalOffset = sqrt(pow(furniture.width / 2, 2) + pow(furniture.depth / 2, 2))
                let offsetX = corner.x + CGFloat(diagonalOffset * cos(rotation + .pi / 4))
                let offsetZ = corner.y + CGFloat(diagonalOffset * sin(rotation + .pi / 4))

                let candidate = PlacementCandidate(
                    id: "corner_\(i)_\(Int(rotation * 180 / .pi))",
                    x: Float(offsetX),
                    y: 0,
                    z: Float(offsetZ),
                    yaw: rotation
                )
                candidates.append(candidate)
            }
        }

        return candidates
    }

    private func sampleCenterArea(furniture: FurnitureItem, room: RoomContext) -> [PlacementCandidate] {
        var candidates: [PlacementCandidate] = []

        let centerX = room.dimensions.width / 2
        let centerZ = room.dimensions.depth / 2

        // Sample a grid around center
        let radius = min(room.dimensions.width, room.dimensions.depth) * 0.2

        for rotation in config.rotationSamples {
            // Center position
            candidates.append(PlacementCandidate(
                id: "center_\(Int(rotation * 180 / .pi))",
                x: centerX,
                y: 0,
                z: centerZ,
                yaw: rotation
            ))

            // Slight offsets from center
            let offsets: [(Float, Float)] = [(radius, 0), (-radius, 0), (0, radius), (0, -radius)]
            for (dx, dz) in offsets {
                candidates.append(PlacementCandidate(
                    id: "center_offset_\(Int(rotation * 180 / .pi))_\(dx)_\(dz)",
                    x: centerX + dx,
                    y: 0,
                    z: centerZ + dz,
                    yaw: rotation
                ))
            }
        }

        return candidates
    }

    private func sampleGrid(furniture: FurnitureItem, room: RoomContext) -> [PlacementCandidate] {
        var candidates: [PlacementCandidate] = []

        let margin = max(furniture.width, furniture.depth) / 2 + config.minWallClearance
        let xSteps = Int((room.dimensions.width - 2 * margin) / config.samplingResolution)
        let zSteps = Int((room.dimensions.depth - 2 * margin) / config.samplingResolution)

        for xi in 0..<max(1, xSteps) {
            for zi in 0..<max(1, zSteps) {
                let x = margin + Float(xi) * config.samplingResolution
                let z = margin + Float(zi) * config.samplingResolution

                for rotation in config.rotationSamples {
                    candidates.append(PlacementCandidate(
                        id: "grid_\(xi)_\(zi)_\(Int(rotation * 180 / .pi))",
                        x: x,
                        y: 0,
                        z: z,
                        yaw: rotation
                    ))
                }
            }
        }

        return candidates
    }

    // MARK: - Scoring

    private func scoreCandidate(
        _ candidate: PlacementCandidate,
        furniture: FurnitureItem,
        room: RoomContext
    ) -> PlacementScores {
        var scores = PlacementScores()

        // Fit score: Does it fit without major collisions?
        scores.fit = computeFitScore(candidate, furniture: furniture, room: room)

        // Clearance score: How much space around furniture?
        scores.clearance = computeClearanceScore(candidate, furniture: furniture, room: room)

        // Walkway score: Are walkways clear?
        scores.walkway = computeWalkwayScore(candidate, furniture: furniture, room: room)

        // Wall alignment score: Is it nicely aligned with walls?
        scores.wallAlignment = computeWallAlignmentScore(candidate, furniture: furniture, room: room)

        // Camera visibility score: Is it visible from current view?
        scores.cameraVisibility = computeVisibilityScore(candidate, furniture: furniture, room: room)

        // Style match (placeholder - could use simple heuristics or VLM later)
        scores.styleMatch = 0.5

        return scores
    }

    private func computeFitScore(
        _ candidate: PlacementCandidate,
        furniture: FurnitureItem,
        room: RoomContext
    ) -> Float {
        let footprint = computeFootprint(candidate, furniture: furniture)

        // Check if footprint is within room bounds
        if !isPolygonInsideRoom(footprint, room: room) {
            return 0
        }

        // Check collisions with obstacles
        var collisionPenalty: Float = 0
        for obstacle in room.obstacles {
            let overlap = computePolygonOverlap(footprint, with: obstacle.footprint)
            if overlap > 0 {
                collisionPenalty += min(1.0, overlap * 10)  // Penalize overlap
            }
        }

        return max(0, 1.0 - collisionPenalty)
    }

    private func computeClearanceScore(
        _ candidate: PlacementCandidate,
        furniture: FurnitureItem,
        room: RoomContext
    ) -> Float {
        // Compute clearance envelope
        let envelope = computeClearanceEnvelope(candidate, furniture: furniture)

        // Check how much of envelope is free
        var freeFraction: Float = 1.0

        for obstacle in room.obstacles {
            let overlap = computePolygonOverlap(envelope, with: obstacle.footprint)
            if overlap > 0 {
                let envelopeArea = polygonArea(envelope)
                freeFraction -= (overlap / max(0.001, envelopeArea))
            }
        }

        return max(0, freeFraction)
    }

    private func computeWalkwayScore(
        _ candidate: PlacementCandidate,
        furniture: FurnitureItem,
        room: RoomContext
    ) -> Float {
        if room.walkways.isEmpty {
            return 1.0  // No defined walkways
        }

        let footprint = computeFootprint(candidate, furniture: furniture)
        var score: Float = 1.0

        for walkway in room.walkways {
            let walkwayPoly = walkwayToPolygon(walkway)
            let overlap = computePolygonOverlap(footprint, with: walkwayPoly)
            if overlap > 0 {
                score -= Float(walkway.priority) * 0.3
            }
        }

        return max(0, score)
    }

    private func computeWallAlignmentScore(
        _ candidate: PlacementCandidate,
        furniture: FurnitureItem,
        room: RoomContext
    ) -> Float {
        // Score based on how parallel furniture is to nearest wall
        guard !room.walls.isEmpty else { return 0.5 }

        var bestAlignment: Float = 0

        for wall in room.walls {
            let wallVector = CGPoint(
                x: wall.endPoint.x - wall.startPoint.x,
                y: wall.endPoint.y - wall.startPoint.y
            )
            let wallAngle = atan2(Float(wallVector.y), Float(wallVector.x))

            // Check if furniture is parallel or perpendicular to wall
            let angleDiff = abs(normalizeAngle(candidate.yaw - wallAngle))
            let parallelScore = 1.0 - min(angleDiff, Float.pi - angleDiff) / (.pi / 4)

            bestAlignment = max(bestAlignment, max(0, parallelScore))
        }

        return bestAlignment
    }

    private func computeVisibilityScore(
        _ candidate: PlacementCandidate,
        furniture: FurnitureItem,
        room: RoomContext
    ) -> Float {
        guard let camera = room.cameraPose else { return 0.5 }

        let furniturePos = SIMD3<Float>(candidate.x, candidate.y + furniture.height / 2, candidate.z)
        let toFurniture = furniturePos - camera.position
        let distance = simd_length(toFurniture)

        if distance < 0.5 { return 0.2 }  // Too close

        let viewDir = simd_normalize(toFurniture)
        let alignment = simd_dot(viewDir, camera.forward)

        return max(0, alignment)
    }

    // MARK: - Constraint Checking

    private func checkConstraints(
        _ candidate: PlacementCandidate,
        furniture: FurnitureItem,
        room: RoomContext
    ) -> [ConstraintViolation] {
        var violations: [ConstraintViolation] = []

        let footprint = computeFootprint(candidate, furniture: furniture)

        // Check room bounds
        if !isPolygonInsideRoom(footprint, room: room) {
            violations.append(ConstraintViolation(
                type: .outOfBounds,
                severity: .blocking,
                message: "Furniture extends outside room boundaries"
            ))
        }

        // Check obstacle collisions
        for obstacle in room.obstacles {
            let overlap = computePolygonOverlap(footprint, with: obstacle.footprint)
            if overlap > 0.01 {
                violations.append(ConstraintViolation(
                    type: .obstacleCollision,
                    severity: .blocking,
                    message: "Collides with \(obstacle.className)",
                    relatedObjectId: obstacle.id,
                    overlapAmount: overlap
                ))
            }
        }

        // Check door clearance
        for opening in room.openings where opening.type == .door {
            let doorClearance = computeDoorClearanceZone(opening)
            let overlap = computePolygonOverlap(footprint, with: doorClearance)
            if overlap > 0.01 {
                violations.append(ConstraintViolation(
                    type: .doorSwingBlocked,
                    severity: furniture.placementHints.avoidNearDoor ? .blocking : .warning,
                    message: "Blocks door swing clearance",
                    relatedObjectId: opening.id,
                    overlapAmount: overlap
                ))
            }
        }

        // Check walkways
        for walkway in room.walkways {
            let walkwayPoly = walkwayToPolygon(walkway)
            let overlap = computePolygonOverlap(footprint, with: walkwayPoly)
            if overlap > 0.01 {
                let severity: ConstraintViolation.Severity = walkway.priority > 2 ? .blocking : .warning
                violations.append(ConstraintViolation(
                    type: .walkwayBlocked,
                    severity: severity,
                    message: "Blocks walkway",
                    relatedObjectId: walkway.id,
                    overlapAmount: overlap
                ))
            }
        }

        // Check wall clearance
        let distToWall = minDistanceToWalls(footprint, room: room)
        if distToWall < config.minWallClearance {
            violations.append(ConstraintViolation(
                type: .insufficientClearance,
                severity: .warning,
                message: "Very close to wall (\(String(format: "%.0f", distToWall * 100))cm)",
                overlapAmount: config.minWallClearance - distToWall
            ))
        }

        return violations
    }

    // MARK: - Geometry Helpers

    private func computeFootprint(_ candidate: PlacementCandidate, furniture: FurnitureItem) -> [CGPoint] {
        let halfW = furniture.width / 2
        let halfD = furniture.depth / 2

        // Local corners
        let corners: [(Float, Float)] = [
            (-halfW, -halfD),
            (halfW, -halfD),
            (halfW, halfD),
            (-halfW, halfD)
        ]

        // Rotate and translate
        let cosYaw = cos(candidate.yaw)
        let sinYaw = sin(candidate.yaw)

        return corners.map { (lx, lz) in
            let rx = lx * cosYaw - lz * sinYaw
            let rz = lx * sinYaw + lz * cosYaw
            return CGPoint(x: CGFloat(candidate.x + rx), y: CGFloat(candidate.z + rz))
        }
    }

    private func computeClearanceEnvelope(_ candidate: PlacementCandidate, furniture: FurnitureItem) -> [CGPoint] {
        let c = furniture.clearance
        let halfW = (furniture.width + c.left + c.right) / 2
        let halfD = (furniture.depth + c.front + c.back) / 2

        let corners: [(Float, Float)] = [
            (-halfW, -halfD),
            (halfW, -halfD),
            (halfW, halfD),
            (-halfW, halfD)
        ]

        let cosYaw = cos(candidate.yaw)
        let sinYaw = sin(candidate.yaw)

        return corners.map { (lx, lz) in
            let rx = lx * cosYaw - lz * sinYaw
            let rz = lx * sinYaw + lz * cosYaw
            return CGPoint(x: CGFloat(candidate.x + rx), y: CGFloat(candidate.z + rz))
        }
    }

    private func isPolygonInsideRoom(_ polygon: [CGPoint], room: RoomContext) -> Bool {
        let roomBounds = CGRect(
            x: 0,
            y: 0,
            width: CGFloat(room.dimensions.width),
            height: CGFloat(room.dimensions.depth)
        )

        for point in polygon {
            if !roomBounds.contains(point) {
                return false
            }
        }

        return true
    }

    private func computePolygonOverlap(_ poly1: [CGPoint], with poly2: [CGPoint]) -> Float {
        // Simplified: use bounding box overlap as approximation
        let bb1 = boundingBox(of: poly1)
        let bb2 = boundingBox(of: poly2)

        let intersection = bb1.intersection(bb2)
        if intersection.isNull || intersection.isEmpty {
            return 0
        }

        return Float(intersection.width * intersection.height)
    }

    private func boundingBox(of polygon: [CGPoint]) -> CGRect {
        guard !polygon.isEmpty else { return .zero }

        var minX = polygon[0].x
        var maxX = polygon[0].x
        var minY = polygon[0].y
        var maxY = polygon[0].y

        for point in polygon {
            minX = min(minX, point.x)
            maxX = max(maxX, point.x)
            minY = min(minY, point.y)
            maxY = max(maxY, point.y)
        }

        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    private func polygonArea(_ polygon: [CGPoint]) -> Float {
        guard polygon.count >= 3 else { return 0 }

        var area: CGFloat = 0
        let n = polygon.count

        for i in 0..<n {
            let j = (i + 1) % n
            area += polygon[i].x * polygon[j].y
            area -= polygon[j].x * polygon[i].y
        }

        return Float(abs(area) / 2)
    }

    private func findCorners(room: RoomContext) -> [CGPoint] {
        if !room.floorPolygon.isEmpty {
            return room.floorPolygon
        }

        // Fallback: rectangular room corners
        let w = CGFloat(room.dimensions.width)
        let d = CGFloat(room.dimensions.depth)
        return [
            CGPoint(x: 0, y: 0),
            CGPoint(x: w, y: 0),
            CGPoint(x: w, y: d),
            CGPoint(x: 0, y: d)
        ]
    }

    private func walkwayToPolygon(_ walkway: WalkwayInfo) -> [CGPoint] {
        let dx = walkway.endPoint.x - walkway.startPoint.x
        let dy = walkway.endPoint.y - walkway.startPoint.y
        let len = sqrt(dx * dx + dy * dy)
        guard len > 0 else { return [] }

        let perpX = -dy / len * CGFloat(walkway.minWidth / 2)
        let perpY = dx / len * CGFloat(walkway.minWidth / 2)

        return [
            CGPoint(x: walkway.startPoint.x + perpX, y: walkway.startPoint.y + perpY),
            CGPoint(x: walkway.endPoint.x + perpX, y: walkway.endPoint.y + perpY),
            CGPoint(x: walkway.endPoint.x - perpX, y: walkway.endPoint.y - perpY),
            CGPoint(x: walkway.startPoint.x - perpX, y: walkway.startPoint.y - perpY)
        ]
    }

    private func computeDoorClearanceZone(_ opening: OpeningInfo) -> [CGPoint] {
        let radius = CGFloat(opening.swingClearance)
        let pos = opening.position

        // Simplified: rectangular clearance zone
        return [
            CGPoint(x: pos.x - radius, y: pos.y - radius),
            CGPoint(x: pos.x + radius, y: pos.y - radius),
            CGPoint(x: pos.x + radius, y: pos.y + radius),
            CGPoint(x: pos.x - radius, y: pos.y + radius)
        ]
    }

    private func minDistanceToWalls(_ polygon: [CGPoint], room: RoomContext) -> Float {
        // Simplified: distance to room boundary
        var minDist: Float = .greatestFiniteMagnitude

        for point in polygon {
            let distToLeft = Float(point.x)
            let distToRight = room.dimensions.width - Float(point.x)
            let distToBottom = Float(point.y)
            let distToTop = room.dimensions.depth - Float(point.y)

            minDist = min(minDist, distToLeft, distToRight, distToBottom, distToTop)
        }

        return max(0, minDist)
    }

    private func normalizeAngle(_ angle: Float) -> Float {
        var a = angle
        while a < 0 { a += 2 * .pi }
        while a >= 2 * .pi { a -= 2 * .pi }
        return a
    }

    private func distance(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
        return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2))
    }

    private func deduplicateCandidates(_ candidates: [PlacementCandidate], minDistance: Float) -> [PlacementCandidate] {
        var result: [PlacementCandidate] = []

        for candidate in candidates {
            let isDuplicate = result.contains { existing in
                let dx = candidate.x - existing.x
                let dz = candidate.z - existing.z
                let dist = sqrt(dx * dx + dz * dz)
                let angleDiff = abs(normalizeAngle(candidate.yaw - existing.yaw))
                return dist < minDistance && angleDiff < 0.1
            }

            if !isDuplicate {
                result.append(candidate)
            }
        }

        return result
    }
}
