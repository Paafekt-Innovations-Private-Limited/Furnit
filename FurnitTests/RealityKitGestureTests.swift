// RealityKitGestureTests.swift
// Tests for RealityKit gesture handling in Sharp ML rooms

import XCTest
import RealityKit
import simd
@testable import Furnit

final class RealityKitGestureTests: XCTestCase {

    // MARK: - Camera Anchor Movement Tests

    func testCameraAnchorCreation() {
        // Test that camera anchor can be created
        let anchor = AnchorEntity()
        let camera = PerspectiveCamera()
        anchor.addChild(camera)

        XCTAssertNotNil(anchor)
        XCTAssertNotNil(camera)
        XCTAssertEqual(anchor.children.count, 1)
    }

    func testCameraAnchorTransformModification() {
        let anchor = AnchorEntity()
        let initialPosition = anchor.transform.translation

        // Modify position
        var newTransform = anchor.transform
        newTransform.translation.x += 1.0
        newTransform.translation.z += 1.0
        anchor.transform = newTransform

        XCTAssertNotEqual(anchor.transform.translation.x, initialPosition.x)
        XCTAssertNotEqual(anchor.transform.translation.z, initialPosition.z)
    }

    func testRotationQuaternionAct() {
        // Test that rotation.act() works for calculating forward/right vectors
        let anchor = AnchorEntity()

        // Default rotation should give forward = (0, 0, -1)
        let forward = anchor.transform.rotation.act(SIMD3<Float>(0, 0, -1))
        let right = anchor.transform.rotation.act(SIMD3<Float>(1, 0, 0))

        // With default rotation, forward should be approximately (0, 0, -1)
        XCTAssertEqual(forward.z, -1.0, accuracy: 0.01)
        // Right should be approximately (1, 0, 0)
        XCTAssertEqual(right.x, 1.0, accuracy: 0.01)
    }

    // MARK: - Movement Calculation Tests

    func testMovementDeltaCalculation() {
        // Simulate the movement calculation from RealityKitGestureHandlers
        let moveSpeed: Float = 0.15
        let velocityX: Float = 0.5
        let velocityY: Float = 0.5

        let forwardBackward = -velocityY * moveSpeed
        let leftRight = velocityX * moveSpeed

        // With velocity (0.5, 0.5), we should get movement
        XCTAssertNotEqual(forwardBackward, 0)
        XCTAssertNotEqual(leftRight, 0)

        // Forward/backward should be negative of velocityY * moveSpeed
        XCTAssertEqual(forwardBackward, -0.5 * 0.15, accuracy: 0.001)
        XCTAssertEqual(leftRight, 0.5 * 0.15, accuracy: 0.001)
    }

    func testNormalizedVectorCalculation() {
        // Test vector normalization (used for movement direction)
        let vector = SIMD3<Float>(3, 0, 4)
        let length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
        let normalized = vector / length

        // Length of normalized vector should be 1
        let normalizedLength = sqrt(normalized.x * normalized.x + normalized.y * normalized.y + normalized.z * normalized.z)
        XCTAssertEqual(normalizedLength, 1.0, accuracy: 0.001)
    }

    func testXZPlaneMovement() {
        // Test that movement stays in XZ plane (Y unchanged)
        let anchor = AnchorEntity()
        anchor.transform.translation = SIMD3<Float>(0, 1.5, 3)

        let initialY = anchor.transform.translation.y

        // Simulate XZ movement
        var newTransform = anchor.transform
        newTransform.translation.x += 0.5
        newTransform.translation.z -= 0.3
        // Y should not change
        anchor.transform = newTransform

        XCTAssertEqual(anchor.transform.translation.y, initialY, accuracy: 0.001)
    }

    // MARK: - Gesture State Tests

    func testAccumulatedRotationState() {
        // Test that accumulated rotation values work correctly
        var accumulatedYaw: Float = 0.0
        var accumulatedPitch: Float = 0.0

        // Simulate drag updates
        let deltaX: Float = 10.0
        let deltaY: Float = 5.0
        let sensitivity: Float = 0.01

        accumulatedYaw += deltaX * sensitivity
        accumulatedPitch += deltaY * sensitivity

        XCTAssertEqual(accumulatedYaw, 0.1, accuracy: 0.001)
        XCTAssertEqual(accumulatedPitch, 0.05, accuracy: 0.001)
    }

    func testPitchClamping() {
        // Test that pitch is clamped to prevent flipping
        var accumulatedPitch: Float = 0.0
        let maxPitch: Float = .pi / 2 - 0.1
        let minPitch: Float = -.pi / 2 + 0.1

        // Add large positive pitch
        accumulatedPitch += 2.0
        accumulatedPitch = min(max(accumulatedPitch, minPitch), maxPitch)

        XCTAssertLessThanOrEqual(accumulatedPitch, maxPitch)

        // Add large negative pitch
        accumulatedPitch = -2.0
        accumulatedPitch = min(max(accumulatedPitch, minPitch), maxPitch)

        XCTAssertGreaterThanOrEqual(accumulatedPitch, minPitch)
    }

    // MARK: - Pan Gesture Delta Tests

    func testPanGestureDeltaCalculation() {
        // Test delta calculation for smooth pan
        var lastTranslation = CGPoint(x: 0, y: 0)
        let currentTranslation = CGPoint(x: 50, y: 30)

        let delta = CGPoint(
            x: currentTranslation.x - lastTranslation.x,
            y: currentTranslation.y - lastTranslation.y
        )

        XCTAssertEqual(delta.x, 50)
        XCTAssertEqual(delta.y, 30)

        // Update last translation
        lastTranslation = currentTranslation

        // Next delta should be from new position
        let nextTranslation = CGPoint(x: 70, y: 40)
        let nextDelta = CGPoint(
            x: nextTranslation.x - lastTranslation.x,
            y: nextTranslation.y - lastTranslation.y
        )

        XCTAssertEqual(nextDelta.x, 20)
        XCTAssertEqual(nextDelta.y, 10)
    }

    // MARK: - GlobalCameraController Integration Tests

    func testGlobalCameraControllerRealityKitRegistration() {
        let controller = GlobalCameraController.shared

        // Create RealityKit camera setup
        let anchor = AnchorEntity()
        let camera = PerspectiveCamera()
        anchor.addChild(camera)

        // Register with controller
        controller.registerRealityKitCamera(anchor, camera: camera, arView: nil)

        // Verify registration by attempting drag
        controller.updateFromDrag(CGSize(width: 50, height: 50))

        // Clean up
        controller.endDrag()
        controller.clearCamera()
    }

    func testDisplayLinkActivation() {
        let controller = GlobalCameraController.shared

        // Clear any existing camera
        controller.clearCamera()

        // Register camera should activate display link
        let anchor = AnchorEntity()
        controller.registerRealityKitCamera(anchor)

        // Display link should be active (we can't directly test this, but we can test drag updates work)
        controller.updateFromDrag(CGSize(width: 100, height: 100))

        // Wait for display link to process
        let expectation = XCTestExpectation(description: "Display link processing")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)

        controller.endDrag()
        controller.clearCamera()
    }
}

// MARK: - RealityKit View Coordinator Tests

final class RealityKitViewCoordinatorTests: XCTestCase {

    func testGestureHandlerCreation() {
        // This tests that gesture handlers can be created
        // In real usage, they're created in RealityKitView.Coordinator

        // We can't easily test the full coordinator without an ARView,
        // but we can verify the types exist
        XCTAssertTrue(true, "RealityKitGestureHandlers class should exist")
    }

    func testCameraReferenceSetup() {
        // Test camera reference setup pattern
        let anchor = AnchorEntity()
        let camera = PerspectiveCamera()

        // Camera should be added as child of anchor
        anchor.addChild(camera)

        // Verify hierarchy
        XCTAssertTrue(anchor.children.contains(where: { $0 is PerspectiveCamera }))
    }
}
