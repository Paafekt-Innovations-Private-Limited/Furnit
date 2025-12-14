// SmartyPantsML.swift
// Stage 1 and Stage 2 ML inference for SmartyPants detection

import Foundation
import CoreML
import CoreVideo

// MARK: - Stage 1 Inference Result

struct Stage1Result {
    let detections: [DetectionSmarty]
    let detectionsArray: MLMultiArray
    let prototypesArray: MLMultiArray
}

// MARK: - Stage 2 Inference Result

struct Stage2Result {
    let detections: [DetectionSmarty]
    let prototypesArray: MLMultiArray
}

// MARK: - Stage 1 Inference

/// Run Stage 1 inference on full frame
/// - Parameters:
///   - model: The CoreML model
///   - pixelBuffer: Source pixel buffer from camera
///   - confThreshold: Confidence threshold for detections
///   - detectAllObjects: Whether to detect all classes or furniture only
///   - furnitureClasses: Dictionary of furniture class indices to names
///   - debugMode: Whether to print debug info
/// - Returns: Stage1Result with detections and arrays, or nil on failure
func runStage1Inference(
    model: MLModel,
    pixelBuffer: CVPixelBuffer,
    confThreshold: Float,
    detectAllObjects: Bool,
    furnitureClasses: [Int: String],
    debugMode: Bool
) -> Stage1Result? {

    if debugMode {
        print("🔬 ========== STAGE 1: FULL FRAME ==========")
    }

    // Preprocess
    let stage1PreStart = Date()
    guard let resized = resizePixelBufferToSquare(pixelBuffer, size: 960, debugMode: debugMode) else {
        if debugMode { print("❌ Stage1: Failed to resize pixel buffer") }
        return nil
    }
    guard let inputArray = pixelBufferToMLMultiArray(resized, debugMode: debugMode) else {
        if debugMode { print("❌ Stage1: Failed to convert to MLMultiArray") }
        return nil
    }
    let stage1PreEnd = Date()
    if debugMode {
        print(String(format: "⏱ Stage1 preprocess (letterbox+toMultiArray): %.2f ms",
                     stage1PreEnd.timeIntervalSince(stage1PreStart) * 1000.0))
    }

    // Inference
    let stage1InfStart = Date()
    guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
        if debugMode { print("❌ Stage1: Failed to create input provider") }
        return nil
    }
    guard let output = try? model.prediction(from: inputProvider) else {
        if debugMode { print("❌ Stage1: Model prediction failed") }
        return nil
    }
    let stage1InfEnd = Date()
    if debugMode {
        print(String(format: "⏱ Stage1 model.prediction: %.2f ms",
                     stage1InfEnd.timeIntervalSince(stage1InfStart) * 1000.0))
    }

    if debugMode {
        let names = output.featureNames.joined(separator: ", ")
        print("📤 Model outputs: \(names)")
    }

    // Find detections array
    var detectionsArray: MLMultiArray?
    if let arr = output.featureValue(for: "var_1432")?.multiArrayValue {
        detectionsArray = arr
    } else if let arr = output.featureValue(for: "var_2497")?.multiArrayValue {
        detectionsArray = arr
    } else {
        for name in output.featureNames {
            if let arr = output.featureValue(for: name)?.multiArrayValue {
                let shape = arr.shape.map { $0.intValue }
                if shape.count == 3 && shape[0] == 1 && shape[1] > 100 {
                    detectionsArray = arr
                    if debugMode { print("   → Using '\(name)' as detections: \(shape)") }
                    break
                }
            }
        }
    }

    guard let detArray = detectionsArray else {
        if debugMode { print("❌ Stage1: No detections array found") }
        return nil
    }

    guard let prototypesArray = output.featureValue(for: "p")?.multiArrayValue else {
        if debugMode { print("❌ Stage1: No prototypes array found") }
        return nil
    }

    // Decode detections
    let decodeStart = Date()
    let detections = extractDetections(
        from: detArray,
        confThreshold: confThreshold,
        detectAllObjects: detectAllObjects,
        furnitureClasses: furnitureClasses,
        debugMode: debugMode
    )
    let decodeEnd = Date()
    if debugMode {
        print("📊 Stage 1: \(detections.count) detections")
        print(String(format: "⏱ Stage1 detection decode: %.2f ms",
                     decodeEnd.timeIntervalSince(decodeStart) * 1000.0))
    }

    return Stage1Result(
        detections: detections,
        detectionsArray: detArray,
        prototypesArray: prototypesArray
    )
}

// MARK: - Stage 2 Inference

/// Run Stage 2 inference on cropped region around primary detection
/// - Parameters:
///   - model: The CoreML model
///   - pixelBuffer: Source pixel buffer from camera
///   - primaryBBox: Primary detection to crop around
///   - confThreshold: Confidence threshold for detections
///   - detectAllObjects: Whether to detect all classes or furniture only
///   - furnitureClasses: Dictionary of furniture class indices to names
///   - debugMode: Whether to print debug info
/// - Returns: Stage2Result with detections and prototypes, or nil on failure
func runStage2Inference(
    model: MLModel,
    pixelBuffer: CVPixelBuffer,
    primaryBBox: DetectionSmarty,
    confThreshold: Float,
    detectAllObjects: Bool,
    furnitureClasses: [Int: String],
    debugMode: Bool
) -> Stage2Result? {

    if debugMode {
        print("\n🔬 ========== STAGE 2: CROPPED ==========")
    }

    let stage2Start = Date()

    // Crop and preprocess
    guard let croppedBuffer = cropPixelBuffer(pixelBuffer, toBBox: primaryBBox, padding: 0.0, debugMode: debugMode) else {
        if debugMode { print("⚠️ Stage 2: Failed to crop pixel buffer") }
        return nil
    }

    guard let resizedCrop = resizePixelBufferToSquare(croppedBuffer, size: 960, debugMode: debugMode) else {
        if debugMode { print("⚠️ Stage 2: Failed to resize cropped buffer") }
        return nil
    }

    guard let cropInputArray = pixelBufferToMLMultiArray(resizedCrop, debugMode: debugMode) else {
        if debugMode { print("⚠️ Stage 2: Failed to convert crop to MLMultiArray") }
        return nil
    }

    guard let cropInputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": cropInputArray]) else {
        if debugMode { print("⚠️ Stage 2: Failed to create input provider") }
        return nil
    }

    // Inference
    let stage2InfStart = Date()
    guard let cropOutput = try? model.prediction(from: cropInputProvider) else {
        if debugMode { print("⚠️ Stage 2: Model prediction failed") }
        return nil
    }
    let stage2InfEnd = Date()
    if debugMode {
        print(String(format: "⏱ Stage2 model.prediction: %.2f ms",
                     stage2InfEnd.timeIntervalSince(stage2InfStart) * 1000.0))
    }

    // Find detections array
    var cropDetArray: MLMultiArray?
    if let arr = cropOutput.featureValue(for: "var_2497")?.multiArrayValue {
        cropDetArray = arr
    } else {
        for name in cropOutput.featureNames {
            if let arr = cropOutput.featureValue(for: name)?.multiArrayValue {
                let shape = arr.shape.map { $0.intValue }
                if shape.count == 3 && shape[0] == 1 && shape[1] > 100 {
                    cropDetArray = arr
                    break
                }
            }
        }
    }

    guard let detArray = cropDetArray else {
        if debugMode { print("⚠️ Stage 2: No detections array found") }
        return nil
    }

    guard let protoArray = cropOutput.featureValue(for: "p")?.multiArrayValue else {
        if debugMode { print("⚠️ Stage 2: No prototypes array found") }
        return nil
    }

    // Decode detections
    let s2DecodeStart = Date()
    let detections = extractDetections(
        from: detArray,
        confThreshold: confThreshold,
        detectAllObjects: detectAllObjects,
        furnitureClasses: furnitureClasses,
        debugMode: debugMode
    )
    let s2DecodeEnd = Date()

    if debugMode {
        print("📊 Stage 2: \(detections.count) detections")
        print(String(format: "⏱ Stage2 detection decode: %.2f ms",
                     s2DecodeEnd.timeIntervalSince(s2DecodeStart) * 1000.0))
    }

    let stage2End = Date()
    if debugMode {
        print(String(format: "⏱ Stage2 total (crop+preprocess+infer+decode): %.2f ms",
                     stage2End.timeIntervalSince(stage2Start) * 1000.0))
    }

    return Stage2Result(
        detections: detections,
        prototypesArray: protoArray
    )
}

// MARK: - Detection Sorting

/// Sort detections by area + confidence score
/// - Parameter detections: Array of detections to sort
/// - Returns: Sorted array with highest scoring first
func sortDetectionsByScore(_ detections: [DetectionSmarty]) -> [DetectionSmarty] {
    return detections.sorted {
        let area0 = $0.width * $0.height
        let area1 = $1.width * $1.height
        let score0 = area0 + $0.confidence
        let score1 = area1 + $1.confidence
        return score0 > score1
    }
}

// MARK: - Raw Detections Extraction

/// Extract raw detections from model output with specified threshold
/// - Parameters:
///   - detectionsArray: MLMultiArray from model output
///   - confThreshold: Confidence threshold
///   - detectAllObjects: Whether to detect all classes
///   - furnitureClasses: Dictionary of furniture class indices
///   - debugMode: Whether to print debug info
/// - Returns: Array of detections
func extractRawDetections(
    from detectionsArray: MLMultiArray,
    confThreshold: Float,
    detectAllObjects: Bool,
    furnitureClasses: [Int: String],
    debugMode: Bool
) -> [DetectionSmarty] {
    return extractDetections(
        from: detectionsArray,
        confThreshold: confThreshold,
        detectAllObjects: detectAllObjects,
        furnitureClasses: furnitureClasses,
        debugMode: debugMode
    )
}
