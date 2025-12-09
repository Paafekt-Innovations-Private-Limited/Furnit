// SmartyPantsTypes.swift
// Shared types and constants for SmartyPants segmentation

import Foundation

// MARK: - Constants
let kModelInputSize = 960
let kModelInputSizeFloat = Float(960)

// MARK: - Detection Struct
struct DetectionSmarty {
    let x: Float
    let y: Float
    let width: Float
    let height: Float
    let confidence: Float
    let classIdx: Int
    let className: String
    let maskCoeffs: [Float]
}
