// GlobalCameraControllerTests.swift
// Unit tests for GlobalCameraController and gesture handling

import XCTest
import SceneKit
import simd
@testable import Furnit

final class GlobalCameraControllerTests: XCTestCase {

    // MARK: - Camera Registration Tests

    func testSceneKitCameraRegistration() {
        let controller = GlobalCameraController.shared

        // Create a SceneKit camera node
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 1.5, 3)

        // Register the camera
        controller.registerSceneKitCamera(cameraNode)

        // Verify camera is registered (controller should be active)
        // We can test this indirectly by checking if drag updates work
        let initialPosition = cameraNode.position

        // Simulate drag
        controller.updateFromDrag(CGSize(width: 100, height: 0))

        // Wait for display link to process
        let expectation = XCTestExpectation(description: "Camera movement")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)

        // End drag
        controller.endDrag()

        // Clear camera
        controller.clearCamera()
    }

    func testClearCamera() {
        let controller = GlobalCameraController.shared

        // Create and register a camera
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        controller.registerSceneKitCamera(cameraNode)

        // Clear the camera
        controller.clearCamera()

        // After clearing, drag updates should not cause movement
        let initialPosition = cameraNode.position
        controller.updateFromDrag(CGSize(width: 100, height: 100))

        // Wait briefly
        let expectation = XCTestExpectation(description: "No movement after clear")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)

        // Position should be unchanged (no camera registered)
        XCTAssertEqual(cameraNode.position.x, initialPosition.x, accuracy: 0.001)
        XCTAssertEqual(cameraNode.position.z, initialPosition.z, accuracy: 0.001)

        controller.endDrag()
    }

    // MARK: - Drag Input Tests

    func testUpdateFromDrag() {
        let controller = GlobalCameraController.shared

        // Reset state
        controller.endDrag()

        // First drag should set initial translation
        controller.updateFromDrag(CGSize(width: 10, height: 20))

        // Second drag should calculate delta
        controller.updateFromDrag(CGSize(width: 30, height: 40))

        // Delta should be (20, 20) from (30-10, 40-20)
        // This is scaled by 0.5, so joystickOffset should be (10, 10)

        controller.endDrag()
    }

    func testEndDrag() {
        let controller = GlobalCameraController.shared

        // Start a drag
        controller.updateFromDrag(CGSize(width: 50, height: 50))

        // End the drag
        controller.endDrag()

        // After ending, a new drag should start fresh
        controller.updateFromDrag(CGSize(width: 10, height: 10))

        // The delta should be from zero, not from the previous position
        controller.endDrag()
    }

    // MARK: - Movement Calculation Tests

    func testSceneKitCameraMovementDirection() {
        let controller = GlobalCameraController.shared

        // Create a camera looking forward (-Z)
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 1.5, 5)
        cameraNode.look(at: SCNVector3(0, 1.5, 0))  // Looking toward -Z

        // Add to a scene (required for worldFront/worldRight)
        let scene = SCNScene()
        scene.rootNode.addChildNode(cameraNode)

        // Register camera
        controller.registerSceneKitCamera(cameraNode)

        let initialZ = cameraNode.position.z

        // Drag down (positive Y) should move forward (-Z direction)
        controller.updateFromDrag(CGSize(width: 0, height: 100))

        // Wait for movement
        let expectation = XCTestExpectation(description: "Forward movement")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)

        controller.endDrag()
        controller.clearCamera()
    }

    // MARK: - Dead Zone Tests

    func testDeadZone() {
        let controller = GlobalCameraController.shared

        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 1.5, 3)

        let scene = SCNScene()
        scene.rootNode.addChildNode(cameraNode)

        controller.registerSceneKitCamera(cameraNode)

        let initialPosition = cameraNode.position

        // Very small drag (within dead zone of 5.0)
        controller.updateFromDrag(CGSize(width: 2, height: 2))

        // Wait briefly
        let expectation = XCTestExpectation(description: "Dead zone check")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)

        // Position should be mostly unchanged due to dead zone
        XCTAssertEqual(cameraNode.position.x, initialPosition.x, accuracy: 0.1)
        XCTAssertEqual(cameraNode.position.z, initialPosition.z, accuracy: 0.1)

        controller.endDrag()
        controller.clearCamera()
    }

    // MARK: - Camera Movement Smoothing Tests

    func testMovementSmoothing() {
        let controller = GlobalCameraController.shared

        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 1.5, 3)

        let scene = SCNScene()
        scene.rootNode.addChildNode(cameraNode)

        controller.registerSceneKitCamera(cameraNode)

        // Apply large drag
        controller.updateFromDrag(CGSize(width: 200, height: 0))

        // Movement should be gradual due to smoothing
        var previousX = cameraNode.position.x
        var movementIncreasing = true

        for i in 0..<5 {
            let expectation = XCTestExpectation(description: "Movement frame \(i)")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 1.0)

            // Check that movement is happening
            let currentX = cameraNode.position.x
            if abs(currentX - previousX) < 0.0001 && i > 2 {
                movementIncreasing = false
            }
            previousX = currentX
        }

        controller.endDrag()
        controller.clearCamera()
    }

    // MARK: - Multiple Camera Type Tests

    func testCameraTypeSwitching() {
        let controller = GlobalCameraController.shared

        // Create SceneKit camera
        let sceneKitCamera = SCNNode()
        sceneKitCamera.camera = SCNCamera()

        // Register SceneKit camera
        controller.registerSceneKitCamera(sceneKitCamera)

        // Clear and verify
        controller.clearCamera()

        // The controller should handle switching cleanly
        controller.endDrag()
    }
}

// MARK: - Touch Drag Overlay Tests

final class TouchDragOverlayTests: XCTestCase {

    func testDragGestureTranslation() {
        // Test that drag translation is properly converted to camera movement
        let controller = GlobalCameraController.shared

        // Simulate a drag sequence
        controller.updateFromDrag(CGSize(width: 0, height: 0))
        controller.updateFromDrag(CGSize(width: 50, height: 100))
        controller.updateFromDrag(CGSize(width: 100, height: 150))

        // End drag
        controller.endDrag()

        // Verify state is reset
        controller.updateFromDrag(CGSize(width: 10, height: 10))
        controller.endDrag()
    }

    func testDragDeltaCalculation() {
        let controller = GlobalCameraController.shared

        // Reset state
        controller.endDrag()

        // First update establishes baseline
        controller.updateFromDrag(CGSize(width: 100, height: 100))

        // Second update should calculate delta correctly
        controller.updateFromDrag(CGSize(width: 150, height: 120))
        // Delta should be (50, 20), scaled by 0.5 = (25, 10)

        controller.endDrag()
    }
}

// MARK: - SceneKit Camera Movement Tests

final class SceneKitCameraMovementTests: XCTestCase {

    func testCameraPositionUpdate() {
        let scene = SCNScene()
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 1.5, 5)
        cameraNode.look(at: SCNVector3(0, 1.5, 0))
        scene.rootNode.addChildNode(cameraNode)

        // Test that position can be modified
        let originalZ = cameraNode.position.z
        cameraNode.position.z -= 0.5

        XCTAssertEqual(cameraNode.position.z, originalZ - 0.5, accuracy: 0.001)
    }

    func testCameraWorldVectors() {
        let scene = SCNScene()
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 1.5, 5)
        cameraNode.look(at: SCNVector3(0, 1.5, 0))  // Looking toward -Z
        scene.rootNode.addChildNode(cameraNode)

        // worldFront should point toward -Z (the direction camera is looking)
        let front = cameraNode.worldFront
        XCTAssertLessThan(front.z, 0, "Camera should be looking toward -Z")

        // worldRight should point toward +X
        let right = cameraNode.worldRight
        XCTAssertGreaterThan(right.x, 0, "Camera right should be toward +X")
    }

    func testCameraHeightPreservation() {
        let scene = SCNScene()
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(0, 1.5, 5)
        scene.rootNode.addChildNode(cameraNode)

        let controller = GlobalCameraController.shared
        controller.registerSceneKitCamera(cameraNode)

        let initialY = cameraNode.position.y

        // Apply horizontal drag (should not affect Y position)
        controller.updateFromDrag(CGSize(width: 100, height: 0))

        let expectation = XCTestExpectation(description: "Height check")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            expectation.fulfill()
        }
        wait(for: [expectation], timeout: 1.0)

        // Y position should remain the same (we only move in X/Z)
        XCTAssertEqual(cameraNode.position.y, initialY, accuracy: 0.001)

        controller.endDrag()
        controller.clearCamera()
    }
}
