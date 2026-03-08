// SharpRoomZoomTests.swift
// Tests for pinch zoom handling: zoom in (scale > 1) = closer, zoom out (scale < 1) = further.
// 1) Unit tests using the same math as the JS in SharpRoomView.
// 2) Integration test that runs the real zoom formula in a WKWebView.

import XCTest
import WebKit
@testable import Furnit

// MARK: - Zoom math (mirrors JS in SharpRoomView: amplifiedScale, offset scale, distance clamp)

private struct ZoomCameraMath {

    static let amplification: Double = 2.5
    static let minDistance: Double = 0.01
    static let maxDistance: Double = 50

    struct Vec3 {
        var x: Double
        var y: Double
        var z: Double
        var length: Double { (x * x + y * y + z * z).squareRoot() }
        mutating func multiplyScalar(_ s: Double) {
            x *= s; y *= s; z *= s
        }
        mutating func setLength(_ L: Double) {
            let len = length
            guard len > 1e-12 else { return }
            let f = L / len
            x *= f; y *= f; z *= f
        }
    }

    /// Returns new camera position after zoom. Same formula as viewer's zoomCamera(scale).
    static func zoom(
        cameraPosition: Vec3,
        target: Vec3,
        scale: Double,
        roomBounds: (minX: Double, maxX: Double, minY: Double, maxY: Double, minZ: Double, maxZ: Double)?
    ) -> Vec3? {
        guard scale > 0, scale.isFinite else { return nil }
        let amplifiedScale = 1 + (scale - 1) * amplification
        var offset = Vec3(
            x: cameraPosition.x - target.x,
            y: cameraPosition.y - target.y,
            z: cameraPosition.z - target.z
        )
        offset.multiplyScalar(1 / amplifiedScale)
        var dist = offset.length
        if dist < minDistance { offset.setLength(minDistance) }
        else if dist > maxDistance { offset.setLength(maxDistance) }
        dist = offset.length

        var newCam = Vec3(
            x: target.x + offset.x,
            y: target.y + offset.y,
            z: target.z + offset.z
        )

        if let b = roomBounds {
            let marginSide = 0.05
            let marginBack = 0.02
            let minZ = b.minZ + marginSide
            let maxZInside = b.maxZ - marginBack
            newCam.x = min(b.maxX - marginSide, max(b.minX + marginSide, newCam.x))
            newCam.y = min(b.maxY - marginSide, max(b.minY + marginSide, newCam.y))
            // Mirror production: if camera is in front of front wall (z > maxZ), don't clamp z from above
            if newCam.z <= b.maxZ {
                newCam.z = min(maxZInside, max(minZ, newCam.z))
            } else {
                newCam.z = max(minZ, newCam.z)
            }
        }
        return newCam
    }

    static func distance(_ a: Vec3, _ b: Vec3) -> Double {
        let dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z
        return (dx * dx + dy * dy + dz * dz).squareRoot()
    }
}

// MARK: - Unit tests (zoom math)

final class SharpRoomZoomTests: XCTestCase {

    func testZoomInDecreasesDistance() {
        let target = ZoomCameraMath.Vec3(x: 0, y: 0, z: 0)
        let camera = ZoomCameraMath.Vec3(x: 0, y: 0, z: 5)
        let initialDist = ZoomCameraMath.distance(camera, target)

        let afterZoomIn = ZoomCameraMath.zoom(cameraPosition: camera, target: target, scale: 1.2, roomBounds: nil)
        XCTAssertNotNil(afterZoomIn)
        let distAfterIn = ZoomCameraMath.distance(afterZoomIn!, target)
        XCTAssertLessThan(distAfterIn, initialDist, "Pinch out (scale 1.2) should move camera closer to target")
    }

    func testZoomOutIncreasesDistance() {
        let target = ZoomCameraMath.Vec3(x: 0, y: 0, z: 0)
        let camera = ZoomCameraMath.Vec3(x: 0, y: 0, z: 5)
        let initialDist = ZoomCameraMath.distance(camera, target)

        let afterZoomOut = ZoomCameraMath.zoom(cameraPosition: camera, target: target, scale: 0.8, roomBounds: nil)
        XCTAssertNotNil(afterZoomOut)
        let distAfterOut = ZoomCameraMath.distance(afterZoomOut!, target)
        XCTAssertGreaterThan(distAfterOut, initialDist, "Pinch in (scale 0.8) should move camera further from target")
    }

    func testZoomInThenZoomOutRestoresDirection() {
        let target = ZoomCameraMath.Vec3(x: 1, y: 2, z: 0)
        var camera = ZoomCameraMath.Vec3(x: 1, y: 2, z: 6)

        let afterIn = ZoomCameraMath.zoom(cameraPosition: camera, target: target, scale: 1.5, roomBounds: nil)
        XCTAssertNotNil(afterIn)
        camera = afterIn!

        let afterOut = ZoomCameraMath.zoom(cameraPosition: camera, target: target, scale: 1/1.5, roomBounds: nil)
        XCTAssertNotNil(afterOut)
        let distOriginal: Double = 6
        let distBack = ZoomCameraMath.distance(afterOut!, target)
        XCTAssertEqual(distBack, distOriginal, accuracy: 0.01, "Zoom in then zoom out by inverse scale should restore distance")
    }

    func testInvalidScaleNoChange() {
        let target = ZoomCameraMath.Vec3(x: 0, y: 0, z: 0)
        let camera = ZoomCameraMath.Vec3(x: 0, y: 0, z: 5)

        XCTAssertNil(ZoomCameraMath.zoom(cameraPosition: camera, target: target, scale: 0, roomBounds: nil))
        XCTAssertNil(ZoomCameraMath.zoom(cameraPosition: camera, target: target, scale: -1, roomBounds: nil))
        XCTAssertNil(ZoomCameraMath.zoom(cameraPosition: camera, target: target, scale: .nan, roomBounds: nil))
    }

    func testDistanceClampMin() {
        let target = ZoomCameraMath.Vec3(x: 0, y: 0, z: 0)
        let camera = ZoomCameraMath.Vec3(x: 0, y: 0, z: 0.005)
        let after = ZoomCameraMath.zoom(cameraPosition: camera, target: target, scale: 1.5, roomBounds: nil)
        XCTAssertNotNil(after)
        let dist = ZoomCameraMath.distance(after!, target)
        XCTAssertGreaterThanOrEqual(dist, 0.01, accuracy: 0.001, "Distance should be clamped to min 0.01")
    }

    func testDistanceClampMax() {
        let target = ZoomCameraMath.Vec3(x: 0, y: 0, z: 0)
        let camera = ZoomCameraMath.Vec3(x: 0, y: 0, z: 100)
        let after = ZoomCameraMath.zoom(cameraPosition: camera, target: target, scale: 0.5, roomBounds: nil)
        XCTAssertNotNil(after)
        let dist = ZoomCameraMath.distance(after!, target)
        XCTAssertLessThanOrEqual(dist, 50, accuracy: 0.1, "Distance should be clamped to max 50")
    }

    func testRoomBoundsClamp() {
        let target = ZoomCameraMath.Vec3(x: 0, y: 0, z: 0)
        var camera = ZoomCameraMath.Vec3(x: 0, y: 0, z: 5)
        let bounds = (minX: -2.0, maxX: 2.0, minY: -1.5, maxY: 1.5, minZ: -4.0, maxZ: 0.0)

        let after = ZoomCameraMath.zoom(cameraPosition: camera, target: target, scale: 0.3, roomBounds: bounds)
        XCTAssertNotNil(after)
        XCTAssertLessThanOrEqual(after!.x, 2.0 - 0.05)
        XCTAssertGreaterThanOrEqual(after!.x, -2.0 + 0.05)
        XCTAssertLessThanOrEqual(after!.z, 0.0 - 0.02)
        XCTAssertGreaterThanOrEqual(after!.z, -4.0 + 0.05)
    }

    // MARK: - Zoom out from in front of wall (must not snap through to back of wall)

    func testZoomOutFromInFrontOfWallKeepsCameraInFront() {
        // Front wall at maxZ = 0. Camera starts just in front (z = 0.025). Target on wall (z = 0).
        let bounds = (minX: -2.0, maxX: 2.0, minY: -1.5, maxY: 1.5, minZ: -4.0, maxZ: 0.0)
        let target = ZoomCameraMath.Vec3(x: 0, y: 0, z: bounds.maxZ)
        let cameraInFront = ZoomCameraMath.Vec3(x: 0, y: 0, z: bounds.maxZ + 0.025)

        let afterZoomOut = ZoomCameraMath.zoom(cameraPosition: cameraInFront, target: target, scale: 0.8, roomBounds: bounds)
        XCTAssertNotNil(afterZoomOut, "Zoom out from in front should produce a position")
        XCTAssertGreaterThan(
            afterZoomOut!.z,
            bounds.maxZ,
            "Zoom out from in front of front wall must not snap camera through the wall (z should stay > maxZ); got z=\(afterZoomOut!.z), maxZ=\(bounds.maxZ)"
        )
    }

    func testZoomOutFromInsideRoomStaysInside() {
        // Camera inside room (z < maxZ). Zoom out should still clamp to room (z <= maxZ - marginBack).
        let bounds = (minX: -2.0, maxX: 2.0, minY: -1.5, maxY: 1.5, minZ: -4.0, maxZ: 0.0)
        let target = ZoomCameraMath.Vec3(x: 0, y: 0, z: -2.0)
        let cameraInside = ZoomCameraMath.Vec3(x: 0, y: 0, z: -1.0)

        let afterZoomOut = ZoomCameraMath.zoom(cameraPosition: cameraInside, target: target, scale: 0.5, roomBounds: bounds)
        XCTAssertNotNil(afterZoomOut)
        XCTAssertLessThanOrEqual(afterZoomOut!.z, bounds.maxZ - 0.02, "Camera inside room should stay inside after zoom out")
        XCTAssertGreaterThanOrEqual(afterZoomOut!.z, bounds.minZ + 0.05)
    }
}

// MARK: - Integration test (run real zoom formula in WKWebView)

extension SharpRoomZoomTests {

    /// HTML that implements the exact same zoom formula as SharpRoomView (plain JS, no THREE).
    private static let zoomTestHTML = """
    <!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
    <script>
    (function() {
        const AMP = 2.5, MIN_D = 0.01, MAX_D = 50;
        const camera = { position: { x: 0, y: 0, z: 5 } };
        const target = { x: 0, y: 0, z: 0 };
        const roomBounds = null;

        function length(v) {
            return Math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        }
        function zoomCamera(scale) {
            if (typeof scale !== 'number' || scale <= 0 || !isFinite(scale)) return;
            const amplifiedScale = 1 + (scale - 1) * AMP;
            const ox = camera.position.x - target.x;
            const oy = camera.position.y - target.y;
            const oz = camera.position.z - target.z;
            let len = Math.sqrt(ox*ox + oy*oy + oz*oz);
            const s = 1 / amplifiedScale;
            let nx = ox * s, ny = oy * s, nz = oz * s;
            len = len * s;
            if (len < MIN_D) { const f = MIN_D/len; nx*=f; ny*=f; nz*=f; len=MIN_D; }
            else if (len > MAX_D) { const f = MAX_D/len; nx*=f; ny*=f; nz*=f; len=MAX_D; }
            camera.position.x = target.x + nx;
            camera.position.y = target.y + ny;
            camera.position.z = target.z + nz;
        }
        function dist() {
            const dx = camera.position.x - target.x;
            const dy = camera.position.y - target.y;
            const dz = camera.position.z - target.z;
            return Math.sqrt(dx*dx + dy*dy + dz*dz);
        }
        window.runZoomTest = function() {
            const d0 = dist();
            zoomCamera(1.2);
            const dAfterIn = dist();
            zoomCamera(0.8);
            const dAfterOut = dist();
            return JSON.stringify({
                initialDistance: d0,
                distanceAfterZoomIn: dAfterIn,
                distanceAfterZoomOut: dAfterOut,
                zoomInMovedCloser: dAfterIn < d0,
                zoomOutMovedFurther: dAfterOut > dAfterIn
            });
        };
    })();
    </script>
    </body></html>
    """

    func testZoomBehaviorInWebView() throws {
        let expectation = expectation(description: "WebView load and zoom test")
        let config = WKWebViewConfiguration()
        let webView = WKWebView(frame: .zero, configuration: config)
        var result: Result<String, Error>?

        webView.loadHTMLString(Self.zoomTestHTML, baseURL: nil)

        let observer = webView.observe(\.url, options: [.new]) { _, _ in }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            webView.evaluateJavaScript("window.runZoomTest ? window.runZoomTest() : 'notReady';") { value, error in
                if let error = error {
                    result = .failure(error)
                } else if let json = value as? String, json != "notReady" {
                    result = .success(json)
                } else {
                    result = .failure(NSError(domain: "ZoomTest", code: -1, userInfo: [NSLocalizedDescriptionKey: "runZoomTest not ready"]))
                }
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 5.0)
        observer.invalidate()

        guard case .success(let jsonString)? = result else {
            if case .failure(let error)? = result { XCTFail("Zoom WebView test failed: \(error)") }
            else { XCTFail("Zoom WebView test: no result") }
            return
        }
        let data = Data(jsonString.utf8)
        let dict = try XCTUnwrap(try? JSONSerialization.jsonObject(with: data) as? [String: Any], "Zoom WebView test: invalid JSON")

        let initialDistance = dict["initialDistance"] as! Double
        let distanceAfterZoomIn = dict["distanceAfterZoomIn"] as! Double
        let distanceAfterZoomOut = dict["distanceAfterZoomOut"] as! Double
        let zoomInMovedCloser = dict["zoomInMovedCloser"] as! Bool
        let zoomOutMovedFurther = dict["zoomOutMovedFurther"] as! Bool

        XCTAssertEqual(initialDistance, 5.0, accuracy: 0.001)
        XCTAssertTrue(zoomInMovedCloser, "Pinch out (scale 1.2) should move camera closer: \(distanceAfterZoomIn) < \(initialDistance)")
        XCTAssertTrue(zoomOutMovedFurther, "Pinch in (scale 0.8) should move camera further: \(distanceAfterZoomOut) > \(distanceAfterZoomIn)")
    }
}
