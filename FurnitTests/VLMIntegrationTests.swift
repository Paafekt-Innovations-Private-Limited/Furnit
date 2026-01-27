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
}
