import CoreGraphics
import CoreML
import Foundation
import UIKit

struct GeoCalibCalibrationResult: Sendable {
    let focalLengthXPixels: Float
    let focalLengthYPixels: Float
    let rollRadians: Float
    let pitchRadians: Float
    let finalCost: Float
    let iterations: Int
    let sourceWidth: Int
    let sourceHeight: Int

    var metadata: [String: Double] {
        [
            "geoCalibFocalLengthPx": Double(focalLengthXPixels),
            "geoCalibFocalLengthYPx": Double(focalLengthYPixels),
            "geoCalibImageWidthPx": Double(sourceWidth),
            "geoCalibImageHeightPx": Double(sourceHeight),
            "geoCalibRollRadians": Double(rollRadians),
            "geoCalibPitchRadians": Double(pitchRadians),
            "geoCalibFinalCost": Double(finalCost),
            "geoCalibIterations": Double(iterations),
        ]
    }
}

final class GeoCalibCalibrationService: @unchecked Sendable {
    static let shared = GeoCalibCalibrationService()

    private let lock = NSLock()
    private var cachedModel: MLModel?
    private var modelLoadAttempted = false

    private init() {}

    func estimateCalibration(image: UIImage) async -> GeoCalibCalibrationResult? {
        await Task.detached(priority: .userInitiated) {
            self.estimateCalibrationSync(image: image)
        }.value
    }

    private func estimateCalibrationSync(image: UIImage) -> GeoCalibCalibrationResult? {
        guard let model = loadModel() else {
            logDebug("[GeoCalib][CNN] unavailable reason=model_not_found")
            return nil
        }
        guard let cgImage = image.fixedOrientation().cgImage else {
            logDebug("[GeoCalib][CNN] unavailable reason=invalid_image")
            return nil
        }

        let sourceWidth = cgImage.width
        let sourceHeight = cgImage.height
        guard let input = Self.makeInputArray(cgImage: cgImage) else {
            logDebug("[GeoCalib][CNN] unavailable reason=input_preprocess_failed")
            return nil
        }

        let provider: MLDictionaryFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(multiArray: input)
            ])
        } catch {
            logDebug("[GeoCalib][CNN] unavailable reason=input_provider_failed error=\(error.localizedDescription)")
            return nil
        }

        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: provider)
        } catch {
            logDebug("[GeoCalib][CNN] unavailable reason=inference_failed error=\(error.localizedDescription)")
            return nil
        }

        guard let fields = Self.decodeFields(from: output) else {
            logDebug("[GeoCalib][CNN] unavailable reason=missing_outputs names=\(Array(output.featureNames).sorted())")
            return nil
        }

        logDebug(
            "[GeoCalib][CNN] outputs field=\(fields.width)x\(fields.height) " +
            "source=\(sourceWidth)x\(sourceHeight)"
        )

        guard let optimized = GeoCalibLMSolver.solve(fields: fields) else {
            logDebug("[GeoCalib][LM] unavailable reason=optimizer_failed")
            return nil
        }

        let cropSide = min(sourceWidth, sourceHeight)
        let scaleToSource = Float(cropSide) / Float(fields.width)
        let focalPx = optimized.focalPixels * scaleToSource

        guard focalPx.isFinite, focalPx > 1 else {
            logDebug("[GeoCalib][LM] unavailable reason=invalid_focal focal=\(focalPx)")
            return nil
        }

        logDebug(
            "[GeoCalib][LM] focal_model_px=\(String(format: "%.2f", optimized.focalPixels)) " +
            "focal_source_px=\(String(format: "%.2f", focalPx)) crop_side=\(cropSide) " +
            "roll=\(String(format: "%.4f", optimized.rollRadians)) " +
            "pitch=\(String(format: "%.4f", optimized.pitchRadians)) " +
            "cost=\(String(format: "%.6f", optimized.finalCost)) " +
            "iterations=\(optimized.iterations)"
        )

        return GeoCalibCalibrationResult(
            focalLengthXPixels: focalPx,
            focalLengthYPixels: focalPx,
            rollRadians: optimized.rollRadians,
            pitchRadians: optimized.pitchRadians,
            finalCost: optimized.finalCost,
            iterations: optimized.iterations,
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight
        )
    }

    private func loadModel() -> MLModel? {
        lock.lock()
        if let cachedModel {
            lock.unlock()
            return cachedModel
        }
        if modelLoadAttempted {
            lock.unlock()
            return nil
        }
        modelLoadAttempted = true
        lock.unlock()

        let candidates = [
            "GeoCalibPinholeCNN",
            "geocalib-pinhole-cnn",
            "geocalib_pinhole_cnn",
        ]
        let extensions = ["mlmodelc", "mlpackage", "mlmodel"]
        let urls = Self.candidateModelURLs(baseNames: candidates, extensions: extensions)

        let config = MLModelConfiguration()
        config.computeUnits = .all

        for sourceURL in urls {
            do {
                let loadURL = sourceURL.pathExtension == "mlpackage" || sourceURL.pathExtension == "mlmodel"
                    ? try MLModel.compileModel(at: sourceURL)
                    : sourceURL
                let model = try MLModel(contentsOf: loadURL, configuration: config)
                lock.lock()
                cachedModel = model
                lock.unlock()
                logDebug("[GeoCalib][CNN] loaded model=\(sourceURL.lastPathComponent)")
                return model
            } catch {
                logDebug("[GeoCalib][CNN] load_failed file=\(sourceURL.lastPathComponent) error=\(error.localizedDescription)")
            }
        }

        return nil
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

    private static func makeInputArray(cgImage: CGImage) -> MLMultiArray? {
        let side = 320
        let sourceWidth = cgImage.width
        let sourceHeight = cgImage.height
        let cropSide = min(sourceWidth, sourceHeight)
        let cropOriginX = (sourceWidth - cropSide) / 2
        let cropOriginY = (sourceHeight - cropSide) / 2

        guard let array = try? MLMultiArray(
            shape: [1, 3, NSNumber(value: side), NSNumber(value: side)],
            dataType: .float32
        ) else {
            return nil
        }

        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixels,
            width: side,
            height: side,
            bitsPerComponent: 8,
            bytesPerRow: side * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGImageByteOrderInfo.order32Big.rawValue
        ) else {
            return nil
        }
        context.interpolationQuality = .high
        let cropRect = CGRect(
            x: CGFloat(cropOriginX),
            y: CGFloat(sourceHeight - cropOriginY - cropSide),
            width: CGFloat(cropSide),
            height: CGFloat(cropSide)
        )
        guard let croppedImage = cgImage.cropping(to: cropRect) else {
            return nil
        }
        context.translateBy(x: 0, y: CGFloat(side))
        context.scaleBy(x: 1, y: -1)
        context.draw(croppedImage, in: CGRect(x: 0, y: 0, width: side, height: side))

        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let planeSize = side * side
        for y in 0..<side {
            for x in 0..<side {
                let pixelIndex = (y * side + x) * 4
                let outIndex = y * side + x
                ptr[outIndex] = Float(pixels[pixelIndex]) / 255.0
                ptr[planeSize + outIndex] = Float(pixels[pixelIndex + 1]) / 255.0
                ptr[2 * planeSize + outIndex] = Float(pixels[pixelIndex + 2]) / 255.0
            }
        }
        return array
    }

    private static func decodeFields(from provider: MLFeatureProvider) -> GeoCalibLMSolver.Fields? {
        guard let up = provider.featureValue(for: "up_field")?.multiArrayValue,
              let latitude = provider.featureValue(for: "latitude_field")?.multiArrayValue else {
            return nil
        }

        let upShape = up.shape.map(\.intValue)
        let height = upShape.suffix(2).first ?? 0
        let width = upShape.suffix(1).first ?? 0
        guard width > 0, height > 0 else { return nil }

        var upX = [Float](repeating: 0, count: width * height)
        var upY = [Float](repeating: 0, count: width * height)
        var lat = [Float](repeating: 0, count: width * height)
        var upConf = [Float](repeating: 1, count: width * height)
        var latConf = [Float](repeating: 1, count: width * height)

        let upStrides = normalizedStrides(up)
        let latStrides = normalizedStrides(latitude)
        let upConfArray = provider.featureValue(for: "up_confidence")?.multiArrayValue
        let latConfArray = provider.featureValue(for: "latitude_confidence")?.multiArrayValue
        let upConfStrides = upConfArray.map { normalizedStrides($0) }
        let latConfStrides = latConfArray.map { normalizedStrides($0) }

        for y in 0..<height {
            for x in 0..<width {
                let i = y * width + x
                upX[i] = value(up, offset: offset(shape: upShape, strides: upStrides, b: 0, c: 0, y: y, x: x))
                upY[i] = value(up, offset: offset(shape: upShape, strides: upStrides, b: 0, c: 1, y: y, x: x))
                lat[i] = value(latitude, offset: offset(shape: latitude.shape.map(\.intValue), strides: latStrides, b: 0, c: 0, y: y, x: x))
                if let upConfArray, let upConfStrides {
                    upConf[i] = value(upConfArray, offset: offset(shape: upConfArray.shape.map(\.intValue), strides: upConfStrides, b: 0, c: nil, y: y, x: x))
                }
                if let latConfArray, let latConfStrides {
                    latConf[i] = value(latConfArray, offset: offset(shape: latConfArray.shape.map(\.intValue), strides: latConfStrides, b: 0, c: nil, y: y, x: x))
                }
            }
        }

        return GeoCalibLMSolver.Fields(
            width: width,
            height: height,
            upX: upX,
            upY: upY,
            latitudeRadians: lat,
            upConfidence: upConf,
            latitudeConfidence: latConf
        )
    }

    private static func normalizedStrides(_ array: MLMultiArray) -> [Int] {
        array.strides.map(\.intValue)
    }

    private static func offset(shape: [Int], strides: [Int], b: Int, c: Int?, y: Int, x: Int) -> Int {
        if shape.count == 4 {
            return b * strides[0] + (c ?? 0) * strides[1] + y * strides[2] + x * strides[3]
        }
        if shape.count == 3 {
            return b * strides[0] + y * strides[1] + x * strides[2]
        }
        if shape.count == 2 {
            return y * strides[0] + x * strides[1]
        }
        return y * (shape.last ?? 0) + x
    }

    private static func value(_ array: MLMultiArray, offset: Int) -> Float {
        guard offset >= 0, offset < array.count else { return 0 }
        switch array.dataType {
        case .float32:
            return array.dataPointer.assumingMemoryBound(to: Float.self)[offset]
        case .double:
            return Float(array.dataPointer.assumingMemoryBound(to: Double.self)[offset])
        case .float16:
            let bits = array.dataPointer.assumingMemoryBound(to: UInt16.self)[offset]
            return Float(Float16(bitPattern: bits))
        case .int32:
            return Float(array.dataPointer.assumingMemoryBound(to: Int32.self)[offset])
        default:
            return array[offset].floatValue
        }
    }
}

enum GeoCalibLMSolver {
    struct Fields {
        let width: Int
        let height: Int
        let upX: [Float]
        let upY: [Float]
        let latitudeRadians: [Float]
        let upConfidence: [Float]
        let latitudeConfidence: [Float]
    }

    struct Result {
        let focalPixels: Float
        let rollRadians: Float
        let pitchRadians: Float
        let finalCost: Float
        let iterations: Int
    }

    private struct Sample {
        let x: Float
        let y: Float
        let upX: Float
        let upY: Float
        let latitudeSin: Float
        let upWeight: Float
        let latitudeWeight: Float
    }

    private struct State {
        var roll: Float
        var pitch: Float
        var logFocal: Float
    }

    static func solve(fields: Fields) -> Result? {
        let samples = makeSamples(fields: fields, stride: 8)
        guard samples.count >= 64 else { return nil }

        let initialFocal = max(12, 0.7 * Float(max(fields.width, fields.height)))
        var state = State(roll: 0, pitch: 0, logFocal: log(initialFocal))
        var lambda: Float = 0.1
        var previous = evaluate(state: state, samples: samples, width: fields.width, height: fields.height)
        var completedIterations = 0

        for iteration in 0..<20 {
            let system = linearizedSystem(
                state: state,
                residuals: previous.residuals,
                samples: samples,
                width: fields.width,
                height: fields.height
            )
            guard let delta = solveDamped(system: system, lambda: lambda) else {
                break
            }

            let candidate = State(
                roll: (state.roll + delta.0).clamped(to: -1.3...1.3),
                pitch: (state.pitch + delta.1).clamped(to: -1.3...1.3),
                logFocal: (state.logFocal + delta.2).clamped(
                    to: log(minFocal(height: fields.height))...log(maxFocal(height: fields.height))
                )
            )
            let next = evaluate(state: candidate, samples: samples, width: fields.width, height: fields.height)
            completedIterations = iteration + 1

            if next.cost < previous.cost {
                let improvement = previous.cost - next.cost
                state = candidate
                previous = next
                lambda = max(1e-6, lambda * 0.1)
                if improvement < max(1e-8, previous.cost * 1e-5) {
                    break
                }
            } else {
                lambda = min(100, lambda * 10)
            }
        }

        let focal = exp(state.logFocal)
        guard focal.isFinite, focal > 1 else { return nil }
        return Result(
            focalPixels: focal,
            rollRadians: state.roll,
            pitchRadians: state.pitch,
            finalCost: previous.cost,
            iterations: completedIterations
        )
    }

    private static func makeSamples(fields: Fields, stride: Int) -> [Sample] {
        var samples: [Sample] = []
        samples.reserveCapacity((fields.width / stride) * (fields.height / stride))
        var y = stride / 2
        while y < fields.height {
            var x = stride / 2
            while x < fields.width {
                let i = y * fields.width + x
                let ux = fields.upX[i]
                let uy = fields.upY[i]
                let lat = fields.latitudeRadians[i]
                if ux.isFinite, uy.isFinite, lat.isFinite {
                    let upNorm = max(1e-6, sqrt(ux * ux + uy * uy))
                    let upConfidence = fields.upConfidence[i].clamped(to: 0...1)
                    let latConfidence = fields.latitudeConfidence[i].clamped(to: 0...1)
                    if upConfidence > 1e-4 || latConfidence > 1e-4 {
                        samples.append(
                            Sample(
                                x: Float(x),
                                y: Float(y),
                                upX: ux / upNorm,
                                upY: uy / upNorm,
                                latitudeSin: sin(lat),
                                upWeight: max(0.05, upConfidence),
                                latitudeWeight: max(0.05, latConfidence)
                            )
                        )
                    }
                }
                x += stride
            }
            y += stride
        }
        return samples
    }

    private static func evaluate(
        state: State,
        samples: [Sample],
        width: Int,
        height: Int
    ) -> (cost: Float, residuals: [[Float]]) {
        var totalCost: Float = 0
        var residuals: [[Float]] = []
        residuals.reserveCapacity(samples.count)
        for sample in samples {
            let prediction = predictedFields(state: state, sample: sample, width: width, height: height)
            let upRX = sample.upX - prediction.upX
            let upRY = sample.upY - prediction.upY
            let latR = sample.latitudeSin - prediction.latitudeSin
            let upSquared = upRX * upRX + upRY * upRY
            let latSquared = latR * latR
            totalCost += sample.upWeight * huberCost(upSquared, scale: 1e-2)
            totalCost += sample.latitudeWeight * huberCost(latSquared, scale: 1e-2)
            residuals.append([upRX, upRY, latR])
        }
        let divisor = max(1, samples.count)
        return (totalCost / Float(divisor), residuals)
    }

    private static func linearizedSystem(
        state: State,
        residuals: [[Float]],
        samples: [Sample],
        width: Int,
        height: Int
    ) -> (gradient: [Float], hessian: [[Float]]) {
        var gradient = [Float](repeating: 0, count: 3)
        var hessian = Array(repeating: [Float](repeating: 0, count: 3), count: 3)
        let eps: [Float] = [1e-3, 1e-3, 1e-3]

        for (index, sample) in samples.enumerated() {
            let base = predictedFields(state: state, sample: sample, width: width, height: height)
            let residual = residuals[index]
            let upSquared = residual[0] * residual[0] + residual[1] * residual[1]
            let latSquared = residual[2] * residual[2]
            let upWeight = sample.upWeight * huberWeight(upSquared, scale: 1e-2)
            let latWeight = sample.latitudeWeight * huberWeight(latSquared, scale: 1e-2)

            var jacobian = Array(repeating: [Float](repeating: 0, count: 3), count: 3)
            for parameter in 0..<3 {
                var stepped = state
                if parameter == 0 {
                    stepped.roll += eps[parameter]
                } else if parameter == 1 {
                    stepped.pitch += eps[parameter]
                } else {
                    stepped.logFocal += eps[parameter]
                }
                let next = predictedFields(state: stepped, sample: sample, width: width, height: height)
                jacobian[0][parameter] = (next.upX - base.upX) / eps[parameter]
                jacobian[1][parameter] = (next.upY - base.upY) / eps[parameter]
                jacobian[2][parameter] = (next.latitudeSin - base.latitudeSin) / eps[parameter]
            }

            let weights = [upWeight, upWeight, latWeight]
            for row in 0..<3 {
                for p in 0..<3 {
                    gradient[p] += weights[row] * jacobian[row][p] * residual[row]
                    for q in 0..<3 {
                        hessian[p][q] += weights[row] * jacobian[row][p] * jacobian[row][q]
                    }
                }
            }
        }
        return (gradient, hessian)
    }

    private static func predictedFields(
        state: State,
        sample: Sample,
        width: Int,
        height: Int
    ) -> (upX: Float, upY: Float, latitudeSin: Float) {
        let focal = exp(state.logFocal)
        let cx = Float(width) * 0.5
        let cy = Float(height) * 0.5
        let u = (sample.x - cx) / focal
        let v = (sample.y - cy) / focal

        let sr = sin(state.roll)
        let cr = cos(state.roll)
        let sp = sin(state.pitch)
        let cp = cos(state.pitch)
        let gx = -sr * cp
        let gy = -cr * cp
        let gz = sp

        var upX = gx - gz * u
        var upY = gy - gz * v
        let upNorm = max(1e-6, sqrt(upX * upX + upY * upY))
        upX /= upNorm
        upY /= upNorm

        let rayNorm = max(1e-6, sqrt(u * u + v * v + 1))
        let latitudeSin = (u * gx + v * gy + gz) / rayNorm
        return (upX, upY, latitudeSin.clamped(to: -1...1))
    }

    private static func huberCost(_ squared: Float, scale: Float) -> Float {
        let normalized = squared / (scale * scale)
        if normalized <= 1 {
            return squared
        }
        return (2 * sqrt(max(normalized, 0)) - 1) * scale * scale
    }

    private static func huberWeight(_ squared: Float, scale: Float) -> Float {
        let normalized = squared / (scale * scale)
        if normalized <= 1 {
            return 1
        }
        return 1 / max(sqrt(normalized), Float.ulpOfOne)
    }

    private static func solveDamped(
        system: (gradient: [Float], hessian: [[Float]]),
        lambda: Float
    ) -> (Float, Float, Float)? {
        var a = system.hessian
        let g = system.gradient
        for i in 0..<3 {
            a[i][i] += max(1e-6, abs(a[i][i]) * lambda)
        }
        return solve3x3(a, g)
    }

    private static func solve3x3(_ matrix: [[Float]], _ rhs: [Float]) -> (Float, Float, Float)? {
        var a = matrix
        var b = rhs

        for pivot in 0..<3 {
            var best = pivot
            var bestValue = abs(a[pivot][pivot])
            for row in pivot + 1..<3 {
                let value = abs(a[row][pivot])
                if value > bestValue {
                    best = row
                    bestValue = value
                }
            }
            guard bestValue > 1e-9 else { return nil }
            if best != pivot {
                a.swapAt(best, pivot)
                b.swapAt(best, pivot)
            }
            let pivotValue = a[pivot][pivot]
            for column in pivot..<3 {
                a[pivot][column] /= pivotValue
            }
            b[pivot] /= pivotValue
            for row in 0..<3 where row != pivot {
                let factor = a[row][pivot]
                guard factor != 0 else { continue }
                for column in pivot..<3 {
                    a[row][column] -= factor * a[pivot][column]
                }
                b[row] -= factor * b[pivot]
            }
        }

        return (b[0], b[1], b[2])
    }

    private static func minFocal(height: Int) -> Float {
        fovToFocal(degrees: 150, size: Float(height))
    }

    private static func maxFocal(height: Int) -> Float {
        fovToFocal(degrees: 5, size: Float(height))
    }

    private static func fovToFocal(degrees: Float, size: Float) -> Float {
        let radians = degrees * .pi / 180
        return size / (2 * tan(radians / 2))
    }
}

private extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
