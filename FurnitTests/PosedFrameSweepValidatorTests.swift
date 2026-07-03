//
//  PosedFrameSweepValidatorTests.swift
//  FurnitTests
//

import XCTest
@testable import Furnit

final class PosedFrameSweepValidatorTests: XCTestCase {
    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("PFSV-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let scratch {
            try? FileManager.default.removeItem(at: scratch)
        }
    }

    func testFrameDiagnosticsSummaryFormatting() {
        let diag = PosedFrameSweepFrameDiagnostics(
            frameID: "f0",
            trackingState: "normal",
            rgbResolution: PixelResolution(width: 1920, height: 1440),
            depthResolution: PixelResolution(width: 256, height: 192),
            fxDepth: 211.5,
            fyDepth: 211.5,
            cxDepth: 128.25,
            cyDepth: 95.75
        )

        XCTAssertEqual(
            diag.summary,
            "id=f0 tracking=normal rgb=1920x1440 depth=256x192 " +
                "Kd=(fx:211.50, fy:211.50, cx:128.25, cy:95.75)"
        )
    }

    func testResultSummaryAllNilOptionalsRenderNA() {
        let diag = PosedFrameSweepFrameDiagnostics(
            frameID: "f0",
            trackingState: "normal",
            rgbResolution: PixelResolution(width: 1920, height: 1440),
            depthResolution: PixelResolution(width: 256, height: 192),
            fxDepth: 1,
            fyDepth: 1,
            cxDepth: 1,
            cyDepth: 1
        )
        let result = PosedFrameSweepValidationResult(
            singleFramePLYURL: URL(fileURLWithPath: "/tmp/a.ply"),
            overlayPLYURL: nil,
            firstFrameIndex: 0,
            secondFrameIndex: nil,
            firstFrameDiagnostics: diag,
            secondFrameDiagnostics: nil,
            firstFramePointCount: 0,
            secondFramePointCount: 0,
            matchedPointCount: 0,
            medianAlignmentOffsetMeters: nil,
            p90AlignmentOffsetMeters: nil,
            medianPlaneResidualMeters: nil,
            p90PlaneResidualMeters: nil,
            planeNormalAngleDegrees: nil,
            planeOffsetMeters: nil,
            firstFrameDepthMedianMeters: nil
        )

        XCTAssertTrue(result.summary.contains("frames=(0,n/a)"))
        XCTAssertTrue(result.summary.contains("nnMedian=n/a nnP90=n/a"))
        XCTAssertTrue(result.summary.contains("planeMedian=n/a planeP90=n/a"))
        XCTAssertTrue(result.summary.contains("planeAngle=n/a planeOffset=n/a"))
        XCTAssertTrue(result.summary.contains("firstDepthMedian=n/a"))
        XCTAssertTrue(result.summary.contains("second={n/a}"))
    }

    func testResultSummaryMetersToCentimetersAndDepth() {
        let diag = PosedFrameSweepFrameDiagnostics(
            frameID: "f0",
            trackingState: "normal",
            rgbResolution: PixelResolution(width: 1920, height: 1440),
            depthResolution: PixelResolution(width: 256, height: 192),
            fxDepth: 1,
            fyDepth: 1,
            cxDepth: 1,
            cyDepth: 1
        )
        let result = PosedFrameSweepValidationResult(
            singleFramePLYURL: URL(fileURLWithPath: "/tmp/a.ply"),
            overlayPLYURL: URL(fileURLWithPath: "/tmp/b.ply"),
            firstFrameIndex: 0,
            secondFrameIndex: 1,
            firstFrameDiagnostics: diag,
            secondFrameDiagnostics: diag,
            firstFramePointCount: 1_000,
            secondFramePointCount: 980,
            matchedPointCount: 850,
            medianAlignmentOffsetMeters: 0.012,
            p90AlignmentOffsetMeters: 0.031,
            medianPlaneResidualMeters: 0.008,
            p90PlaneResidualMeters: 0.019,
            planeNormalAngleDegrees: 2.5,
            planeOffsetMeters: 0.03,
            firstFrameDepthMedianMeters: 1.42
        )
        let summary = result.summary

        XCTAssertTrue(summary.contains("frames=(0,1)"))
        XCTAssertTrue(summary.contains("points=(1000,980) matched=850"))
        XCTAssertTrue(summary.contains("nnMedian=1.2cm nnP90=3.1cm"))
        XCTAssertTrue(summary.contains("planeMedian=0.8cm planeP90=1.9cm"))
        XCTAssertTrue(summary.contains("planeAngle=2.5deg planeOffset=3.0cm"))
        XCTAssertTrue(summary.contains("firstDepthMedian=1.42m"))
    }

    func testErrorDescriptions() {
        XCTAssertEqual(
            PosedFrameSweepValidationError.noFrames.errorDescription,
            "Sweep has no posed frames."
        )
        XCTAssertEqual(
            PosedFrameSweepValidationError.missingDepth.errorDescription,
            "Posed frame is missing LiDAR depth."
        )
        XCTAssertEqual(
            PosedFrameSweepValidationError.nonNormalTracking("limited").errorDescription,
            "Posed frame was captured with non-normal tracking: limited."
        )
        XCTAssertEqual(
            PosedFrameSweepValidationError.invalidGridSize("depth_00.bin").errorDescription,
            "Grid file has an invalid byte size: depth_00.bin."
        )
        XCTAssertEqual(
            PosedFrameSweepValidationError.invalidMatrix.errorDescription,
            "Pose or intrinsics matrix has invalid shape."
        )
        XCTAssertEqual(
            PosedFrameSweepValidationError.invalidFrameIndex(index: 7, frameCount: 3).errorDescription,
            "Frame index 7 is outside the sweep frame range 0..<3."
        )
        XCTAssertEqual(
            PosedFrameSweepValidationError.duplicateFrameIndex(2).errorDescription,
            "Validator needs two different frames; frame 2 was selected twice."
        )
    }

    func testValidateThrowsWhenPosesJSONMissing() {
        XCTAssertThrowsError(try PosedFrameSweepValidator.validate(sessionURL: scratch))
    }

    func testValidateThrowsNoFramesOnEmptyManifest() throws {
        let session = try writeManifest(frames: [])

        XCTAssertThrowsError(try PosedFrameSweepValidator.validate(sessionURL: session)) { error in
            XCTAssertEqual(error as? PosedFrameSweepValidationError, .noFrames)
        }
    }

    func testValidateThrowsInvalidFrameIndexBeforeReadingDepth() throws {
        let session = try writeManifest(frames: [
            makeFrame(id: "frame_0000", depthRelativePath: "missing0.bin"),
            makeFrame(id: "frame_0001", depthRelativePath: "missing1.bin")
        ])

        XCTAssertThrowsError(
            try PosedFrameSweepValidator.validate(sessionURL: session, firstFrameIndex: 5)
        ) { error in
            XCTAssertEqual(
                error as? PosedFrameSweepValidationError,
                .invalidFrameIndex(index: 5, frameCount: 2)
            )
        }
    }

    func testValidateThrowsDuplicateFrameIndexBeforeReadingDepth() throws {
        let session = try writeManifest(frames: [
            makeFrame(id: "frame_0000", depthRelativePath: "missing0.bin"),
            makeFrame(id: "frame_0001", depthRelativePath: "missing1.bin")
        ])

        XCTAssertThrowsError(
            try PosedFrameSweepValidator.validate(
                sessionURL: session,
                firstFrameIndex: 0,
                secondFrameIndex: 0
            )
        ) { error in
            XCTAssertEqual(error as? PosedFrameSweepValidationError, .duplicateFrameIndex(0))
        }
    }

    func testValidateReportsScaledIntrinsicsAndWritesDebugPLYs() throws {
        try writeDepthGrid(relativePath: "depth0.bin", width: 16, height: 12, depthMeters: 1)
        try writeDepthGrid(relativePath: "depth1.bin", width: 16, height: 12, depthMeters: 1)
        let session = try writeManifest(frames: [
            makeFrame(id: "frame_0000", depthRelativePath: "depth0.bin"),
            makeFrame(id: "frame_0001", depthRelativePath: "depth1.bin")
        ])

        let result = try PosedFrameSweepValidator.validate(
            sessionURL: session,
            stride: 1,
            maxAlignmentDistanceMeters: 0.25
        )

        XCTAssertEqual(result.firstFramePointCount, 192)
        XCTAssertEqual(result.secondFramePointCount, 192)
        XCTAssertEqual(result.firstFrameDiagnostics.rgbResolution.width, 160)
        XCTAssertEqual(result.firstFrameDiagnostics.rgbResolution.height, 120)
        XCTAssertEqual(result.firstFrameDiagnostics.depthResolution.width, 16)
        XCTAssertEqual(result.firstFrameDiagnostics.depthResolution.height, 12)
        XCTAssertEqual(result.firstFrameDiagnostics.fxDepth, 10, accuracy: 0.001)
        XCTAssertEqual(result.firstFrameDiagnostics.fyDepth, 10, accuracy: 0.001)
        XCTAssertEqual(result.firstFrameDiagnostics.cxDepth, 8, accuracy: 0.001)
        XCTAssertEqual(result.firstFrameDiagnostics.cyDepth, 6, accuracy: 0.001)
        XCTAssertEqual(result.firstFrameDepthMedianMeters ?? -1, 1, accuracy: 0.001)
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.singleFramePLYURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.overlayPLYURL?.path ?? ""))
    }

    func testValidatePlaneResidualCatchesNormalDoubleWallOffset() throws {
        try writeDepthGrid(relativePath: "depth0.bin", width: 32, height: 24, depthMeters: 1)
        try writeDepthGrid(relativePath: "depth1.bin", width: 32, height: 24, depthMeters: 1)
        let session = try writeManifest(frames: [
            makeFrame(id: "frame_0000", depthRelativePath: "depth0.bin", depthWidth: 32, depthHeight: 24, transformZ: 0),
            makeFrame(id: "frame_0001", depthRelativePath: "depth1.bin", depthWidth: 32, depthHeight: 24, transformZ: 0.08)
        ])

        let result = try PosedFrameSweepValidator.validate(
            sessionURL: session,
            stride: 1,
            maxAlignmentDistanceMeters: 0.25
        )

        XCTAssertEqual(result.planeOffsetMeters ?? -1, 0.08, accuracy: 0.015)
        XCTAssertEqual(result.medianPlaneResidualMeters ?? -1, 0.08, accuracy: 0.015)
        XCTAssertLessThan(result.planeNormalAngleDegrees ?? 999, 1.0)
    }

    private func writeManifest(frames: [PosedFrameRecord]) throws -> URL {
        let manifest = PosedFrameSweepManifest(
            version: 1,
            createdAt: Date(timeIntervalSince1970: 0),
            sessionID: "test-session",
            coordinateConvention: "test",
            frames: frames
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: scratch.appendingPathComponent("poses.json"), options: [.atomic])
        return scratch
    }

    private func makeFrame(
        id: String,
        depthRelativePath: String?,
        depthWidth: Int = 16,
        depthHeight: Int = 12,
        transformZ: Float = 0
    ) -> PosedFrameRecord {
        PosedFrameRecord(
            id: id,
            imageRelativePath: "\(id).jpg",
            depthRelativePath: depthRelativePath,
            confidenceRelativePath: nil,
            cameraTransformWorldFromCamera: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, transformZ, 1
            ],
            intrinsicsRGB: [
                100, 0, 0,
                0, 100, 0,
                80, 60, 1
            ],
            rgbResolution: PixelResolution(width: 160, height: 120),
            depthResolution: PixelResolution(width: depthWidth, height: depthHeight),
            trackingState: "normal",
            timestamp: 0
        )
    }

    private func writeDepthGrid(
        relativePath: String,
        width: Int,
        height: Int,
        depthMeters: Float
    ) throws {
        var data = Data(capacity: width * height * MemoryLayout<Float>.size)
        for _ in 0..<(width * height) {
            var value = depthMeters
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        try data.write(to: scratch.appendingPathComponent(relativePath), options: [.atomic])
    }
}
