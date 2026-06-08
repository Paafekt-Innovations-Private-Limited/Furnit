import AVFoundation
import CoreImage
import CoreML
import XCTest
@testable import Furnit

final class RTMDetVideoIntegrationTests: XCTestCase {

    func testRTMDetSegmentsRealChairPhotoWithFurnitureFilter() throws {
        let imagePath = "/Users/al/Downloads/WhatsApp Image 2026-06-08 at 16.08.53.jpeg"
        guard FileManager.default.fileExists(atPath: imagePath),
              let image = UIImage(contentsOfFile: imagePath) else {
            throw XCTSkip("Real chair photo fixture is not available at \(imagePath)")
        }

        let model = try loadRTMDetModel()
        let result = try RTMDetImageInference.runInstanceSegmentation(
            image: image,
            model: model,
            confidenceThreshold: 0.25,
            classBlacklist: [],
            allowedClassIndices: [56, 57, 59, 60],
            maxMaskCount: 3,
            maxDetectionCount: 8,
            buildInstanceMasks: true
        )

        let chairDetections = result.detections.filter { $0.classIdx == 56 }
        XCTAssertGreaterThanOrEqual(chairDetections.count, 2, "Expected at least the visible chairs to be detected. Detections: \(result.detections)")

        guard let overlay = result.overlayMaskImage else {
            XCTFail("Expected RTMDet to produce a chair mask overlay")
            return
        }
        let stats = alphaStats(in: overlay)
        XCTAssertGreaterThan(stats.pixelCount, 5000, "Chair overlay is too sparse: \(stats)")
        XCTAssertGreaterThan(stats.bounds.width, 100, "Chair overlay is too narrow: \(stats.bounds)")
        XCTAssertGreaterThan(stats.bounds.height, 100, "Chair overlay is too short: \(stats.bounds)")
        XCTAssertLessThan(stats.fillRatio, 0.82, "Chair overlay is still too rectangular: \(stats)")

        let chairMaskStats = result.instanceMaskImages
            .compactMap { $0 }
            .map(alphaStats(in:))
            .filter { $0.pixelCount > 5000 && $0.bounds.width > 100 && $0.bounds.height > 100 }
        XCTAssertGreaterThanOrEqual(
            chairMaskStats.count,
            2,
            "Expected per-instance masks for at least two chairs. Mask stats: \(chairMaskStats)"
        )
    }

    func testRTMDetLiveCadenceRunsRealInferenceOnVideoFrames() throws {
        let model = try loadRTMDetModel()
        assertRTMDetInterface(model)

        let sourceImage = try loadFixtureImage(named: "bus", extension: "jpg")
        let videoURL = try makeVideoFixture(from: sourceImage, frameCount: 9, fps: 3)
        let frames = try readBGRAVideoFrames(from: videoURL)

        XCTAssertEqual(frames.count, 9, "Video fixture should decode every frame")

        var processedTimestamps: [TimeInterval] = []
        var outputSummaries: Set<String> = []
        var bestDetectionCount = 0
        var bestMaskAlphaPixels = 0
        var bestMaskBounds = CGRect.zero
        var bestMaskFillRatio = 1.0
        var lastProcessTime = Date.distantPast
        let minimumRTMDetInterval: TimeInterval = 1.0
        let baseProcessInterval: TimeInterval = 0.07
        let effectiveInterval = max(baseProcessInterval, minimumRTMDetInterval)
        let startDate = Date(timeIntervalSince1970: 10_000)

        for frame in frames {
            let now = startDate.addingTimeInterval(frame.timestamp)
            guard now.timeIntervalSince(lastProcessTime) >= effectiveInterval else {
                continue
            }
            lastProcessTime = now
            processedTimestamps.append(frame.timestamp)

            let result = try RTMDetImageInference.runInstanceSegmentation(
                pixelBuffer: frame.pixelBuffer,
                model: model,
                confidenceThreshold: 0.20,
                classBlacklist: [],
                maxMaskCount: 3,
                maxDetectionCount: 8
            )

            bestDetectionCount = max(bestDetectionCount, result.detections.count)
            if let overlayMaskImage = result.overlayMaskImage {
                let stats = alphaStats(in: overlayMaskImage)
                if stats.pixelCount > bestMaskAlphaPixels {
                    bestMaskAlphaPixels = stats.pixelCount
                    bestMaskBounds = stats.bounds
                    bestMaskFillRatio = stats.fillRatio
                }
            }
            result.outputSummary.forEach { outputSummaries.insert($0) }
        }

        XCTAssertEqual(processedTimestamps.count, 3)
        zip(processedTimestamps, [0.0, 1.0, 2.0]).forEach { actual, expected in
            XCTAssertEqual(actual, expected, accuracy: 0.05)
        }
        XCTAssertGreaterThan(bestDetectionCount, 0, "Real RTMDet inference should detect at least one object in the video fixture")
        XCTAssertGreaterThan(
            bestMaskAlphaPixels,
            0,
            "RTMDet-Ins must produce a non-empty mask overlay from video frames. Outputs: \(outputSummaries.sorted())"
        )
        XCTAssertGreaterThan(bestMaskBounds.width, 100, "RTMDet mask overlay is too narrow: \(bestMaskBounds)")
        XCTAssertGreaterThan(bestMaskBounds.height, 100, "RTMDet mask overlay is too short: \(bestMaskBounds)")
        XCTAssertLessThan(bestMaskFillRatio, 0.88, "RTMDet mask overlay is too rectangular: bounds=\(bestMaskBounds) fillRatio=\(bestMaskFillRatio)")
        XCTAssertTrue(outputSummaries.contains { $0 == "dets: 1x100x5 float32" }, "Missing dets output: \(outputSummaries)")
        XCTAssertTrue(outputSummaries.contains { $0 == "labels: 1x100 int32" }, "Missing labels output: \(outputSummaries)")
        XCTAssertTrue(outputSummaries.contains { $0 == "masks: 1x100x640x640 float32" }, "Missing masks output: \(outputSummaries)")
    }

    private struct VideoFrame {
        let timestamp: TimeInterval
        let pixelBuffer: CVPixelBuffer
    }

    private func loadRTMDetModel() throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly

        let candidates: [(Bundle, String?)] = [
            (Bundle.main, "Models/RTMDet"),
            (Bundle.main, nil),
            (Bundle(for: type(of: self)), "Models/RTMDet"),
            (Bundle(for: type(of: self)), nil),
        ]

        for (bundle, subdirectory) in candidates {
            if let url = bundle.url(forResource: "rtmdet-ins-m", withExtension: "mlmodelc", subdirectory: subdirectory) ??
                bundle.url(forResource: "rtmdet-ins-m", withExtension: "mlpackage", subdirectory: subdirectory) {
                return try MLModel(contentsOf: url, configuration: config)
            }
        }

        let sourcePackageURL = URL(fileURLWithPath: "/Users/al/Documents/tries01/Furnit/Furnit/Models/RTMDet/rtmdet-ins-m.mlpackage")
        if FileManager.default.fileExists(atPath: sourcePackageURL.path) {
            return try MLModel(contentsOf: sourcePackageURL, configuration: config)
        }

        throw XCTSkip("rtmdet-ins-m Core ML model is not available in the app/test bundle")
    }

    private func assertRTMDetInterface(_ model: MLModel) {
        let inputs = model.modelDescription.inputDescriptionsByName
        XCTAssertEqual(inputs.count, 1)
        XCTAssertNotNil(inputs["input"])
        XCTAssertEqual(inputs["input"]?.type, .multiArray)
        XCTAssertEqual(inputs["input"]?.multiArrayConstraint?.shape.map(\.intValue), [1, 3, 640, 640])

        let outputs = model.modelDescription.outputDescriptionsByName
        XCTAssertNotNil(outputs["dets"])
        XCTAssertNotNil(outputs["labels"])
        XCTAssertNotNil(outputs["masks"])
    }

    private func alphaStats(in image: UIImage) -> (pixelCount: Int, bounds: CGRect, fillRatio: Double) {
        guard let cgImage = image.cgImage else { return (0, .zero, 0) }
        let width = cgImage.width
        let height = cgImage.height
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
            return (0, .zero, 0)
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        var minX = width
        var minY = height
        var maxX = -1
        var maxY = -1
        var count = 0
        for y in 0..<height {
            for x in 0..<width {
                let alphaIndex = (y * width + x) * 4 + 3
                guard rgba[alphaIndex] > 0 else { continue }
                count += 1
                minX = min(minX, x)
                minY = min(minY, y)
                maxX = max(maxX, x)
                maxY = max(maxY, y)
            }
        }
        guard count > 0 else {
            return (0, .zero, 0)
        }
        let bounds = CGRect(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1)
        let boundsArea = max(1, Int(bounds.width * bounds.height))
        return (
            count,
            bounds,
            Double(count) / Double(boundsArea)
        )
    }

    private func loadFixtureImage(named name: String, extension ext: String) throws -> UIImage {
        let bundle = Bundle(for: type(of: self))
        if let url = bundle.url(forResource: name, withExtension: ext),
           let image = UIImage(contentsOfFile: url.path) {
            return image
        }

        let sourceURL = URL(fileURLWithPath: "/Users/al/Documents/tries01/Furnit/FurnitTests/\(name).\(ext)")
        if let image = UIImage(contentsOfFile: sourceURL.path) {
            return image
        }

        XCTFail("Missing fixture \(name).\(ext)")
        throw NSError(domain: "RTMDetVideoIntegrationTests", code: 1)
    }

    private func makeVideoFixture(from image: UIImage, frameCount: Int, fps: Int32) throws -> URL {
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("rtmdet-video-fixture-\(UUID().uuidString).mov")
        let size = CGSize(width: 720, height: 1280)
        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: Int(size.width),
            AVVideoHeightKey: Int(size.height),
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: Int(size.width),
                kCVPixelBufferHeightKey as String: Int(size.height),
                kCVPixelBufferCGImageCompatibilityKey as String: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
            ]
        )

        XCTAssertTrue(writer.canAdd(input))
        writer.add(input)
        XCTAssertTrue(writer.startWriting())
        writer.startSession(atSourceTime: .zero)

        for index in 0..<frameCount {
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.01)
            }
            let presentationTime = CMTime(value: CMTimeValue(index), timescale: fps)
            let frame = try makePixelBufferFrame(image: image, size: size, frameIndex: index)
            XCTAssertTrue(adaptor.append(frame, withPresentationTime: presentationTime))
        }

        input.markAsFinished()
        let finished = expectation(description: "video writer finished")
        writer.finishWriting {
            finished.fulfill()
        }
        wait(for: [finished], timeout: 10)
        XCTAssertEqual(writer.status, .completed, "Video writer failed: \(String(describing: writer.error))")
        return outputURL
    }

    private func makePixelBufferFrame(image: UIImage, size: CGSize, frameIndex: Int) throws -> CVPixelBuffer {
        guard let cgImage = image.cgImage else {
            throw NSError(domain: "RTMDetVideoIntegrationTests", code: 2)
        }

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32BGRA,
            [
                kCVPixelBufferCGImageCompatibilityKey: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            ] as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw NSError(domain: "RTMDetVideoIntegrationTests", code: 3)
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else {
            throw NSError(domain: "RTMDetVideoIntegrationTests", code: 4)
        }

        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(origin: .zero, size: size))

        let imageSize = CGSize(width: cgImage.width, height: cgImage.height)
        let scale = max(size.width / imageSize.width, size.height / imageSize.height)
        let drawnSize = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        let pan = CGFloat(frameIndex - 4) * 8
        let drawRect = CGRect(
            x: (size.width - drawnSize.width) * 0.5 + pan,
            y: (size.height - drawnSize.height) * 0.5,
            width: drawnSize.width,
            height: drawnSize.height
        )
        context.draw(cgImage, in: drawRect)
        return buffer
    }

    private func readBGRAVideoFrames(from url: URL) throws -> [VideoFrame] {
        let asset = AVAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            XCTFail("Video fixture has no video track")
            return []
        }

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            ]
        )
        output.alwaysCopiesSampleData = false
        XCTAssertTrue(reader.canAdd(output))
        reader.add(output)
        XCTAssertTrue(reader.startReading())

        var frames: [VideoFrame] = []
        while let sample = output.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            let timestamp = CMSampleBufferGetPresentationTimeStamp(sample).seconds
            frames.append(VideoFrame(timestamp: timestamp, pixelBuffer: pixelBuffer))
        }

        XCTAssertEqual(reader.status, .completed, "Video reader failed: \(String(describing: reader.error))")
        return frames
    }
}
