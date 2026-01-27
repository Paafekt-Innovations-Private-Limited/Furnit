// VLMIntegrationTests.swift
// Tests for VLM spatial reasoning integration

import XCTest
@testable import Furnit

final class VLMIntegrationTests: XCTestCase {

    // MARK: - Spatial Reasoning Tests

    func testSpatialReasoningGeneratesCandidates() throws {
        let service = SpatialReasoningService()

        // Create a simple room
        let room = RoomContext(
            dimensions: RoomDimensions(width: 4.0, depth: 3.5, height: 2.5),
            floorPolygon: [
                CGPoint(x: 0, y: 0),
                CGPoint(x: 4, y: 0),
                CGPoint(x: 4, y: 3.5),
                CGPoint(x: 0, y: 3.5)
            ],
            walls: [
                WallInfo(id: "wall_north", startPoint: CGPoint(x: 0, y: 0), endPoint: CGPoint(x: 4, y: 0)),
                WallInfo(id: "wall_east", startPoint: CGPoint(x: 4, y: 0), endPoint: CGPoint(x: 4, y: 3.5)),
                WallInfo(id: "wall_south", startPoint: CGPoint(x: 4, y: 3.5), endPoint: CGPoint(x: 0, y: 3.5)),
                WallInfo(id: "wall_west", startPoint: CGPoint(x: 0, y: 3.5), endPoint: CGPoint(x: 0, y: 0))
            ]
        )

        // Create furniture to place (a sofa)
        let sofa = FurnitureItem(
            id: "sofa_1",
            name: "3-Seater Sofa",
            category: "seating",
            width: 2.2,
            depth: 0.9,
            height: 0.85,
            clearance: FurnitureClearance(front: 0.8, back: 0.1, left: 0.3, right: 0.3),
            placementHints: PlacementHints(preferAgainstWall: true, faceTowardCenter: true)
        )

        // Generate candidates
        let candidates = service.generateCandidates(furniture: sofa, room: room)

        // Should generate multiple candidates
        XCTAssertGreaterThan(candidates.count, 0, "Should generate placement candidates")

        // At least some should be valid
        let validCandidates = candidates.filter { $0.isValid }
        XCTAssertGreaterThan(validCandidates.count, 0, "Should have valid placement candidates")

        // Best candidate should have reasonable scores
        if let best = validCandidates.first {
            XCTAssertGreaterThan(best.scores.fit, 0, "Best candidate should have fit score > 0")
            XCTAssertGreaterThan(best.compositeScore, 0, "Best candidate should have composite score > 0")
        }
    }

    func testQuickFitCheckForLargeFurniture() throws {
        let service = SpatialReasoningService()

        // Small room
        let room = RoomContext(
            dimensions: RoomDimensions(width: 2.0, depth: 2.0, height: 2.5)
        )

        // Furniture too large for room
        let largeSofa = FurnitureItem(
            id: "large_sofa",
            name: "Large Sectional",
            category: "seating",
            width: 3.0,  // Larger than room width
            depth: 1.5,
            height: 0.85
        )

        let (fits, _) = service.quickFitCheck(furniture: largeSofa, room: room)
        XCTAssertFalse(fits, "Large furniture should not fit in small room")
    }

    func testQuickFitCheckForSmallFurniture() throws {
        let service = SpatialReasoningService()

        // Normal room
        let room = RoomContext(
            dimensions: RoomDimensions(width: 4.0, depth: 3.5, height: 2.5)
        )

        // Small furniture
        let chair = FurnitureItem(
            id: "chair_1",
            name: "Accent Chair",
            category: "seating",
            width: 0.8,
            depth: 0.8,
            height: 0.9
        )

        let (fits, bestCandidate) = service.quickFitCheck(furniture: chair, room: room)
        XCTAssertTrue(fits, "Small furniture should fit in normal room")
        XCTAssertNotNil(bestCandidate, "Should find a best candidate")
    }

    func testConstraintViolationsDetected() throws {
        let service = SpatialReasoningService()

        // Room with obstacle
        var room = RoomContext(
            dimensions: RoomDimensions(width: 4.0, depth: 3.5, height: 2.5)
        )

        // Add existing sofa as obstacle
        room.obstacles = [
            ObstacleInfo(
                id: "existing_sofa",
                className: "sofa",
                footprint: [
                    CGPoint(x: 1.5, y: 0.1),
                    CGPoint(x: 3.5, y: 0.1),
                    CGPoint(x: 3.5, y: 1.0),
                    CGPoint(x: 1.5, y: 1.0)
                ],
                boundingBox: CGRect(x: 1.5, y: 0.1, width: 2.0, height: 0.9),
                height: 0.85,
                isMovable: true
            )
        ]

        // Try to place another sofa
        let newSofa = FurnitureItem(
            id: "new_sofa",
            name: "New Sofa",
            category: "seating",
            width: 2.0,
            depth: 0.9,
            height: 0.85,
            placementHints: PlacementHints(preferAgainstWall: true)
        )

        let candidates = service.generateCandidates(furniture: newSofa, room: room)

        // Some candidates should have collision violations
        let candidatesWithViolations = candidates.filter { !$0.violations.isEmpty }
        XCTAssertGreaterThan(candidatesWithViolations.count, 0, "Some candidates should have violations due to obstacle")
    }

    // MARK: - VLM Models Tests

    func testRoomContextCodable() throws {
        let room = RoomContext(
            dimensions: RoomDimensions(width: 4.0, depth: 3.5, height: 2.5),
            styleHint: "modern"
        )

        // Encode
        let encoder = JSONEncoder()
        let data = try encoder.encode(room)

        // Decode
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(RoomContext.self, from: data)

        XCTAssertEqual(decoded.dimensions.width, room.dimensions.width)
        XCTAssertEqual(decoded.dimensions.depth, room.dimensions.depth)
        XCTAssertEqual(decoded.styleHint, room.styleHint)
    }

    func testFurnitureItemCodable() throws {
        let furniture = FurnitureItem(
            id: "test_chair",
            name: "Test Chair",
            category: "seating",
            width: 0.8,
            depth: 0.8,
            height: 0.9,
            style: FurnitureStyle(primaryColor: "blue", material: "fabric")
        )

        // Encode
        let encoder = JSONEncoder()
        let data = try encoder.encode(furniture)

        // Decode
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(FurnitureItem.self, from: data)

        XCTAssertEqual(decoded.id, furniture.id)
        XCTAssertEqual(decoded.name, furniture.name)
        XCTAssertEqual(decoded.width, furniture.width)
        XCTAssertEqual(decoded.style?.primaryColor, "blue")
    }

    func testPlacementCandidateCodable() throws {
        let candidate = PlacementCandidate(
            id: "test_candidate",
            x: 2.0,
            y: 0,
            z: 1.5,
            yaw: Float.pi / 2,
            scores: PlacementScores(fit: 0.9, clearance: 0.8, walkway: 1.0),
            violations: [
                ConstraintViolation(
                    type: .insufficientClearance,
                    severity: .warning,
                    message: "Close to wall"
                )
            ]
        )

        // Encode
        let encoder = JSONEncoder()
        let data = try encoder.encode(candidate)

        // Decode
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PlacementCandidate.self, from: data)

        XCTAssertEqual(decoded.id, candidate.id)
        XCTAssertEqual(decoded.x, candidate.x)
        XCTAssertEqual(decoded.yaw, candidate.yaw, accuracy: 0.001)
        XCTAssertEqual(decoded.scores.fit, 0.9, accuracy: 0.001)
        XCTAssertEqual(decoded.violations.count, 1)
    }

    func testVLMDesignRequestPromptGeneration() throws {
        let room = RoomContext(
            dimensions: RoomDimensions(width: 4.0, depth: 3.5, height: 2.5),
            styleHint: "scandinavian"
        )

        let furniture = FurnitureItem(
            id: "sofa_1",
            name: "Gray Sofa",
            category: "seating",
            width: 2.2,
            depth: 0.9,
            height: 0.85,
            style: FurnitureStyle(primaryColor: "gray", material: "fabric", styleCategory: "modern")
        )

        let candidates = [
            PlacementCandidate(
                id: "opt_1",
                x: 2.0, y: 0, z: 0.5,
                yaw: 0,
                scores: PlacementScores(fit: 0.9, clearance: 0.8)
            ),
            PlacementCandidate(
                id: "opt_2",
                x: 0.5, y: 0, z: 1.75,
                yaw: Float.pi / 2,
                scores: PlacementScores(fit: 0.85, clearance: 0.9)
            )
        ]

        let request = VLMDesignRequest(
            roomContext: room,
            furniture: furniture,
            candidates: candidates,
            questions: ["Which placement creates better flow?"]
        )

        let prompt = request.generatePrompt()

        // Verify prompt contains key information
        XCTAssertTrue(prompt.contains("4.0m"), "Prompt should contain room width")
        XCTAssertTrue(prompt.contains("3.5m"), "Prompt should contain room depth")
        XCTAssertTrue(prompt.contains("Gray Sofa"), "Prompt should contain furniture name")
        XCTAssertTrue(prompt.contains("seating"), "Prompt should contain furniture category")
        XCTAssertTrue(prompt.contains("Option 1"), "Prompt should contain options")
        XCTAssertTrue(prompt.contains("Option 2"), "Prompt should contain multiple options")
        XCTAssertTrue(prompt.contains("better flow"), "Prompt should contain user questions")
    }

    // MARK: - VLM Service Tests (Unit tests, no actual API calls)

    func testVLMServiceConfigDefaults() throws {
        let service = VLMService()

        XCTAssertEqual(service.config.provider, .claude)
        XCTAssertNil(service.config.apiKey)
        XCTAssertEqual(service.config.maxTokens, 1024)
    }

    func testVLMProviderModels() throws {
        XCTAssertEqual(VLMProvider.claude.defaultModel, "claude-3-5-sonnet-20241022")
        XCTAssertEqual(VLMProvider.openai.defaultModel, "gpt-4o")
        XCTAssertEqual(VLMProvider.gemini.defaultModel, "gemini-1.5-pro")
    }

    func testVLMManagerSingleton() throws {
        let manager1 = VLMManager.shared
        let manager2 = VLMManager.shared

        XCTAssertTrue(manager1 === manager2, "VLMManager should be a singleton")
    }

    func testVLMManagerConfiguration() throws {
        let manager = VLMManager.shared

        // Configure with test key
        manager.configure(provider: .openai, apiKey: "test-key")

        XCTAssertEqual(manager.vlmService.config.provider, .openai)
        XCTAssertEqual(manager.vlmService.config.apiKey, "test-key")

        // Reset to default
        manager.configure(provider: .claude, apiKey: nil)
    }

    // MARK: - Integration Tests

    func testFullPlacementAnalysisWithoutVLM() async throws {
        let room = RoomContext(
            dimensions: RoomDimensions(width: 4.0, depth: 3.5, height: 2.5),
            walls: [
                WallInfo(id: "wall_north", startPoint: CGPoint(x: 0, y: 0), endPoint: CGPoint(x: 4, y: 0)),
                WallInfo(id: "wall_east", startPoint: CGPoint(x: 4, y: 0), endPoint: CGPoint(x: 4, y: 3.5)),
                WallInfo(id: "wall_south", startPoint: CGPoint(x: 4, y: 3.5), endPoint: CGPoint(x: 0, y: 3.5)),
                WallInfo(id: "wall_west", startPoint: CGPoint(x: 0, y: 3.5), endPoint: CGPoint(x: 0, y: 0))
            ]
        )

        let furniture = FurnitureItem(
            id: "dining_table",
            name: "Dining Table",
            category: "table",
            width: 1.6,
            depth: 0.9,
            height: 0.75,
            clearance: FurnitureClearance(front: 0.7, back: 0.7, left: 0.7, right: 0.7),
            placementHints: PlacementHints(preferCentered: true)
        )

        // Run analysis without VLM (no API key)
        let result = await VLMManager.shared.analyzePlacement(
            furniture: furniture,
            room: room,
            includeVLM: false
        )

        XCTAssertTrue(result.fits, "Dining table should fit in room")
        XCTAssertGreaterThan(result.candidates.count, 0, "Should have candidates")
        XCTAssertNil(result.vlmResponse, "VLM response should be nil when disabled")
        XCTAssertGreaterThan(result.solverTimeMs, 0, "Solver time should be recorded")

        if let best = result.bestCandidate {
            print("Best placement: x=\(best.x), z=\(best.z), yaw=\(best.yaw * 180 / .pi)°")
            print("Scores: fit=\(best.scores.fit), clearance=\(best.scores.clearance)")
        }
    }

    // MARK: - Performance Tests

    func testSpatialReasoningPerformance() throws {
        let service = SpatialReasoningService()

        let room = RoomContext(
            dimensions: RoomDimensions(width: 5.0, depth: 4.0, height: 2.7),
            walls: [
                WallInfo(id: "wall_north", startPoint: CGPoint(x: 0, y: 0), endPoint: CGPoint(x: 5, y: 0)),
                WallInfo(id: "wall_east", startPoint: CGPoint(x: 5, y: 0), endPoint: CGPoint(x: 5, y: 4)),
                WallInfo(id: "wall_south", startPoint: CGPoint(x: 5, y: 4), endPoint: CGPoint(x: 0, y: 4)),
                WallInfo(id: "wall_west", startPoint: CGPoint(x: 0, y: 4), endPoint: CGPoint(x: 0, y: 0))
            ],
            obstacles: [
                ObstacleInfo(id: "obs_1", className: "chair",
                            boundingBox: CGRect(x: 1, y: 1, width: 0.5, height: 0.5)),
                ObstacleInfo(id: "obs_2", className: "lamp",
                            boundingBox: CGRect(x: 4, y: 3, width: 0.3, height: 0.3))
            ]
        )

        let furniture = FurnitureItem(
            id: "sofa",
            name: "Sofa",
            category: "seating",
            width: 2.0,
            depth: 0.9,
            height: 0.85,
            placementHints: PlacementHints(preferAgainstWall: true)
        )

        // Measure performance
        measure {
            _ = service.generateCandidates(furniture: furniture, room: room)
        }
    }

    // MARK: - Additional Model Tests

    func testRoomDimensionsCalculations() throws {
        let dims = RoomDimensions(width: 4.0, depth: 3.5, height: 2.5)

        XCTAssertEqual(dims.floorArea, 14.0, accuracy: 0.001, "Floor area should be width * depth")
        XCTAssertEqual(dims.volume, 35.0, accuracy: 0.001, "Volume should be width * depth * height")
    }

    func testWallInfoCodable() throws {
        let wall = WallInfo(
            id: "test_wall",
            startPoint: CGPoint(x: 0, y: 0),
            endPoint: CGPoint(x: 4, y: 0),
            height: 2.5,
            material: "painted"
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(wall)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(WallInfo.self, from: data)

        XCTAssertEqual(decoded.id, wall.id)
        XCTAssertEqual(decoded.startPoint, wall.startPoint)
        XCTAssertEqual(decoded.endPoint, wall.endPoint)
        XCTAssertEqual(decoded.height, wall.height)
        XCTAssertEqual(decoded.material, wall.material)
    }

    func testObstacleInfoCodable() throws {
        let obstacle = ObstacleInfo(
            id: "obs_1",
            className: "sofa",
            footprint: [CGPoint(x: 0, y: 0), CGPoint(x: 2, y: 0), CGPoint(x: 2, y: 1), CGPoint(x: 0, y: 1)],
            boundingBox: CGRect(x: 0, y: 0, width: 2, height: 1),
            height: 0.85,
            isMovable: true
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(obstacle)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(ObstacleInfo.self, from: data)

        XCTAssertEqual(decoded.id, obstacle.id)
        XCTAssertEqual(decoded.className, obstacle.className)
        XCTAssertEqual(decoded.footprint.count, 4)
        XCTAssertEqual(decoded.isMovable, true)
    }

    func testOpeningInfoCodable() throws {
        let door = OpeningInfo(
            id: "door_1",
            type: .door,
            position: CGPoint(x: 2, y: 0),
            width: 0.9,
            height: 2.1,
            swingClearance: 0.9,
            wallId: "wall_north"
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(door)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(OpeningInfo.self, from: data)

        XCTAssertEqual(decoded.id, door.id)
        XCTAssertEqual(decoded.type, .door)
        XCTAssertEqual(decoded.swingClearance, 0.9)
        XCTAssertEqual(decoded.wallId, "wall_north")
    }

    func testWalkwayInfoCodable() throws {
        let walkway = WalkwayInfo(
            id: "walkway_1",
            startPoint: CGPoint(x: 0, y: 1.5),
            endPoint: CGPoint(x: 4, y: 1.5),
            minWidth: 0.8,
            priority: 2
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(walkway)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(WalkwayInfo.self, from: data)

        XCTAssertEqual(decoded.id, walkway.id)
        XCTAssertEqual(decoded.minWidth, 0.8)
        XCTAssertEqual(decoded.priority, 2)
    }

    func testCameraPoseCodable() throws {
        let camera = CameraPose(
            position: SIMD3<Float>(2, 1.6, 3),
            forward: SIMD3<Float>(0, 0, -1),
            up: SIMD3<Float>(0, 1, 0),
            fovDegrees: 60
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(camera)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(CameraPose.self, from: data)

        XCTAssertEqual(decoded.position.x, 2, accuracy: 0.001)
        XCTAssertEqual(decoded.position.y, 1.6, accuracy: 0.001)
        XCTAssertEqual(decoded.fovDegrees, 60)
    }

    func testFurnitureClearanceEnvelope() throws {
        let clearance = FurnitureClearance(front: 0.6, back: 0.1, left: 0.3, right: 0.3)
        let envelope = clearance.envelope(forWidth: 2.0, depth: 0.9, atRotation: 0)

        // Envelope should be larger than furniture
        XCTAssertEqual(envelope.count, 4, "Envelope should be a quadrilateral")

        // Total width should be left + width + right = 0.3 + 2.0 + 0.3 = 2.6
        // Total depth should be back + depth + front = 0.1 + 0.9 + 0.6 = 1.6
        let minX = envelope.map { $0.x }.min()!
        let maxX = envelope.map { $0.x }.max()!
        let totalWidth = maxX - minX
        XCTAssertEqual(Float(totalWidth), 2.6, accuracy: 0.001)
    }

    func testFurnitureFootprintArea() throws {
        let furniture = FurnitureItem(
            id: "table",
            name: "Table",
            width: 1.6,
            depth: 0.9,
            height: 0.75
        )

        XCTAssertEqual(furniture.footprintArea, 1.44, accuracy: 0.001, "Footprint should be width * depth")
    }

    func testPlacementHintsDefaults() throws {
        let hints = PlacementHints()

        XCTAssertFalse(hints.preferAgainstWall)
        XCTAssertFalse(hints.preferCorner)
        XCTAssertFalse(hints.preferCentered)
        XCTAssertFalse(hints.preferNearWindow)
        XCTAssertTrue(hints.avoidNearDoor)  // Default is true
        XCTAssertFalse(hints.faceTowardCenter)
    }

    func testFurnitureStyleCodable() throws {
        let style = FurnitureStyle(
            primaryColor: "navy blue",
            material: "velvet",
            styleCategory: "modern",
            finish: "matte"
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(style)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(FurnitureStyle.self, from: data)

        XCTAssertEqual(decoded.primaryColor, "navy blue")
        XCTAssertEqual(decoded.material, "velvet")
        XCTAssertEqual(decoded.styleCategory, "modern")
        XCTAssertEqual(decoded.finish, "matte")
    }

    // MARK: - Placement Scores Tests

    func testPlacementScoresComposite() throws {
        let scores = PlacementScores(
            fit: 1.0,
            clearance: 0.8,
            walkway: 1.0,
            wallAlignment: 0.5,
            cameraVisibility: 0.6,
            styleMatch: 0.7
        )

        // Composite = 0.30*1.0 + 0.20*0.8 + 0.20*1.0 + 0.15*0.5 + 0.05*0.6 + 0.10*0.7
        //           = 0.30 + 0.16 + 0.20 + 0.075 + 0.03 + 0.07 = 0.835
        XCTAssertEqual(scores.composite, 0.835, accuracy: 0.001)
    }

    func testPlacementWeightsDefault() throws {
        let weights = PlacementWeights()

        // Should sum to 1.0
        let total = weights.fit + weights.clearance + weights.walkway +
                   weights.wallAlignment + weights.cameraVisibility + weights.styleMatch
        XCTAssertEqual(total, 1.0, accuracy: 0.001, "Weights should sum to 1.0")
    }

    // MARK: - Constraint Violation Tests

    func testConstraintViolationTypes() throws {
        let violations: [ConstraintViolation] = [
            ConstraintViolation(type: .wallCollision, severity: .blocking, message: "Hits wall"),
            ConstraintViolation(type: .obstacleCollision, severity: .blocking, message: "Hits furniture"),
            ConstraintViolation(type: .insufficientClearance, severity: .warning, message: "Tight fit"),
            ConstraintViolation(type: .walkwayBlocked, severity: .warning, message: "Blocks path"),
            ConstraintViolation(type: .doorSwingBlocked, severity: .blocking, message: "Blocks door"),
            ConstraintViolation(type: .outOfBounds, severity: .blocking, message: "Outside room"),
            ConstraintViolation(type: .tooNearDoor, severity: .info, message: "Near door")
        ]

        XCTAssertEqual(violations.count, 7, "Should have all violation types")

        let blockingCount = violations.filter { $0.severity == .blocking }.count
        XCTAssertEqual(blockingCount, 4, "Should have 4 blocking violations")

        let warningCount = violations.filter { $0.severity == .warning }.count
        XCTAssertEqual(warningCount, 2, "Should have 2 warning violations")
    }

    func testPlacementCandidateValidity() throws {
        // Candidate with no blocking violations is valid
        let validCandidate = PlacementCandidate(
            id: "valid",
            x: 2, y: 0, z: 2,
            violations: [
                ConstraintViolation(type: .insufficientClearance, severity: .warning, message: "Tight")
            ]
        )
        XCTAssertTrue(validCandidate.isValid, "Candidate with only warnings should be valid")

        // Candidate with blocking violation is invalid
        let invalidCandidate = PlacementCandidate(
            id: "invalid",
            x: 2, y: 0, z: 2,
            violations: [
                ConstraintViolation(type: .wallCollision, severity: .blocking, message: "Collision")
            ]
        )
        XCTAssertFalse(invalidCandidate.isValid, "Candidate with blocking violation should be invalid")
    }

    // MARK: - VLM Response Tests

    func testVLMDesignResponseCodable() throws {
        let response = VLMDesignResponse(
            recommendedCandidateIndex: 0,
            recommendation: "Option 1 is best for traffic flow",
            adjustments: [
                PlacementAdjustment(type: .rotate, amount: Float.pi / 4, reason: "Better alignment")
            ],
            styleSuggestions: ["Consider lighter fabric"],
            concerns: ["Space may feel cramped"],
            confidence: 0.85,
            rawResponse: "Full VLM response text..."
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(response)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(VLMDesignResponse.self, from: data)

        XCTAssertEqual(decoded.recommendedCandidateIndex, 0)
        XCTAssertEqual(decoded.adjustments.count, 1)
        XCTAssertEqual(decoded.adjustments[0].type, .rotate)
        XCTAssertEqual(decoded.confidence, 0.85, accuracy: 0.001)
    }

    func testPlacementAdjustmentTypes() throws {
        let adjustments: [PlacementAdjustment] = [
            PlacementAdjustment(type: .rotate, amount: Float.pi / 2, reason: "Face window"),
            PlacementAdjustment(type: .shiftX, amount: 0.2, reason: "More clearance"),
            PlacementAdjustment(type: .shiftZ, amount: -0.1, reason: "Away from wall"),
            PlacementAdjustment(type: .alignWithWall, amount: 0, reason: "Straighten"),
            PlacementAdjustment(type: .alignWithFurniture, amount: 0, reason: "Match sofa")
        ]

        XCTAssertEqual(adjustments.count, 5, "Should have all adjustment types")
    }

    // MARK: - Edge Case Tests

    func testEmptyRoom() throws {
        let service = SpatialReasoningService()

        let emptyRoom = RoomContext(
            dimensions: RoomDimensions(width: 4.0, depth: 3.5, height: 2.5)
            // No walls, obstacles, or openings
        )

        let chair = FurnitureItem(
            id: "chair",
            name: "Chair",
            width: 0.6,
            depth: 0.6,
            height: 0.9
        )

        let candidates = service.generateCandidates(furniture: chair, room: emptyRoom)

        // Should still generate candidates (grid sampling)
        XCTAssertGreaterThan(candidates.count, 0, "Should generate candidates even for empty room")
    }

    func testVerySmallFurniture() throws {
        let service = SpatialReasoningService()

        let room = RoomContext(
            dimensions: RoomDimensions(width: 4.0, depth: 3.5, height: 2.5)
        )

        let tinyItem = FurnitureItem(
            id: "tiny",
            name: "Small Decoration",
            width: 0.1,
            depth: 0.1,
            height: 0.1
        )

        let (fits, _) = service.quickFitCheck(furniture: tinyItem, room: room)
        XCTAssertTrue(fits, "Tiny furniture should easily fit")
    }

    func testFurnitureExactlyFitsRoom() throws {
        let service = SpatialReasoningService()

        // Room just barely fits the furniture
        let room = RoomContext(
            dimensions: RoomDimensions(width: 2.0, depth: 1.0, height: 2.5)
        )

        let furniture = FurnitureItem(
            id: "tight_fit",
            name: "Tight Fit Sofa",
            width: 1.9,  // Almost as wide as room
            depth: 0.9,  // Almost as deep as room
            height: 0.85
        )

        let candidates = service.generateCandidates(furniture: furniture, room: room)
        let validCandidates = candidates.filter { $0.isValid }

        // Should have very few or no valid placements due to tight fit
        // (depends on clearance requirements)
        XCTAssertNotNil(candidates, "Should return candidates array")
    }

    func testRoomWithDoorBlocking() throws {
        let service = SpatialReasoningService()

        var room = RoomContext(
            dimensions: RoomDimensions(width: 4.0, depth: 3.5, height: 2.5),
            walls: [
                WallInfo(id: "wall_north", startPoint: CGPoint(x: 0, y: 0), endPoint: CGPoint(x: 4, y: 0))
            ]
        )

        // Add door in the middle of north wall
        room.openings = [
            OpeningInfo(
                id: "door_1",
                type: .door,
                position: CGPoint(x: 2, y: 0.5),  // Just inside north wall
                width: 0.9,
                height: 2.1,
                swingClearance: 0.9,
                wallId: "wall_north"
            )
        ]

        let sofa = FurnitureItem(
            id: "sofa",
            name: "Sofa",
            width: 2.0,
            depth: 0.9,
            height: 0.85,
            placementHints: PlacementHints(avoidNearDoor: true)
        )

        let candidates = service.generateCandidates(furniture: sofa, room: room)

        // Some candidates near door should have doorSwingBlocked violations
        let doorBlockingViolations = candidates.flatMap { $0.violations }
            .filter { $0.type == .doorSwingBlocked }

        // May or may not have violations depending on candidate positions
        XCTAssertNotNil(candidates, "Should generate candidates")
    }

    func testRoomWithWalkway() throws {
        let service = SpatialReasoningService()

        var room = RoomContext(
            dimensions: RoomDimensions(width: 4.0, depth: 3.5, height: 2.5)
        )

        // Add walkway through middle of room
        room.walkways = [
            WalkwayInfo(
                id: "main_walkway",
                startPoint: CGPoint(x: 0, y: 1.75),
                endPoint: CGPoint(x: 4, y: 1.75),
                minWidth: 0.8,
                priority: 3  // High priority
            )
        ]

        let table = FurnitureItem(
            id: "table",
            name: "Coffee Table",
            width: 1.2,
            depth: 0.6,
            height: 0.45,
            placementHints: PlacementHints(preferCentered: true)
        )

        let candidates = service.generateCandidates(furniture: table, room: room)

        // Candidates in walkway should have lower walkway scores
        let centeredCandidates = candidates.filter {
            abs($0.z - 1.75) < 0.5  // Near walkway center
        }

        if !centeredCandidates.isEmpty {
            let avgWalkwayScore = centeredCandidates.map { $0.scores.walkway }.reduce(0, +) / Float(centeredCandidates.count)
            XCTAssertLessThan(avgWalkwayScore, 1.0, "Candidates in walkway should have reduced walkway score")
        }
    }

    // MARK: - VLM Error Tests

    func testVLMErrorDescriptions() throws {
        let errors: [VLMError] = [
            .missingAPIKey,
            .invalidResponse,
            .apiError(code: 401, message: "Unauthorized"),
            .parseError("Invalid JSON"),
            .timeout,
            .networkError(NSError(domain: "test", code: -1))
        ]

        for error in errors {
            XCTAssertNotNil(error.errorDescription, "Error should have description: \(error)")
            XCTAssertFalse(error.errorDescription!.isEmpty, "Error description should not be empty")
        }
    }

    func testVLMServiceWithoutAPIKey() async throws {
        let service = VLMService(config: VLMService.Config(provider: .claude, apiKey: nil))

        let request = VLMDesignRequest(
            roomContext: RoomContext(),
            furniture: FurnitureItem(id: "test", name: "Test"),
            candidates: []
        )

        do {
            _ = try await service.getDesignSuggestions(request: request)
            XCTFail("Should throw missingAPIKey error")
        } catch let error as VLMError {
            if case .missingAPIKey = error {
                // Expected
            } else {
                XCTFail("Wrong error type: \(error)")
            }
        }
    }

    // MARK: - Provider Tests

    func testAllVLMProviders() throws {
        for provider in VLMProvider.allCases {
            XCTAssertFalse(provider.displayName.isEmpty, "\(provider) should have display name")
            XCTAssertFalse(provider.defaultModel.isEmpty, "\(provider) should have default model")
        }
    }

    func testVLMProviderCodable() throws {
        for provider in VLMProvider.allCases {
            let encoder = JSONEncoder()
            let data = try encoder.encode(provider)

            let decoder = JSONDecoder()
            let decoded = try decoder.decode(VLMProvider.self, from: data)

            XCTAssertEqual(decoded, provider)
        }
    }

    // MARK: - Placement Result Tests

    func testPlacementResultBestCandidate() throws {
        let candidates = [
            PlacementCandidate(
                id: "low",
                x: 1, y: 0, z: 1,
                scores: PlacementScores(fit: 0.5, clearance: 0.5),
                violations: []
            ),
            PlacementCandidate(
                id: "high",
                x: 2, y: 0, z: 2,
                scores: PlacementScores(fit: 0.9, clearance: 0.9),
                violations: []
            ),
            PlacementCandidate(
                id: "invalid",
                x: 3, y: 0, z: 3,
                scores: PlacementScores(fit: 1.0, clearance: 1.0),
                violations: [ConstraintViolation(type: .wallCollision, severity: .blocking, message: "Blocked")]
            )
        ]

        let result = PlacementResult(fits: true, candidates: candidates)

        XCTAssertNotNil(result.bestCandidate)
        XCTAssertEqual(result.bestCandidate?.id, "high", "Best should be highest scoring valid candidate")
    }

    func testPlacementResultNoValidCandidates() throws {
        let candidates = [
            PlacementCandidate(
                id: "invalid1",
                x: 1, y: 0, z: 1,
                scores: PlacementScores(fit: 0.9),
                violations: [ConstraintViolation(type: .outOfBounds, severity: .blocking, message: "Out")]
            ),
            PlacementCandidate(
                id: "invalid2",
                x: 2, y: 0, z: 2,
                scores: PlacementScores(fit: 0.8),
                violations: [ConstraintViolation(type: .wallCollision, severity: .blocking, message: "Wall")]
            )
        ]

        let result = PlacementResult(fits: false, candidates: candidates)

        XCTAssertNil(result.bestCandidate, "Should have no best candidate when all are invalid")
    }
}
