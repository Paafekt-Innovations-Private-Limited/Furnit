// SmartyPantsImageUtils.swift
// Image and pixel buffer utilities for SmartyPants detection

import Foundation
import CoreML
import Accelerate
import CoreVideo
import UIKit

// MARK: - Pixel Buffer Utilities

/// Resize a pixel buffer to a square of specified size using vImage
/// - Parameters:
///   - src: Source pixel buffer
///   - size: Target square size (default 960)
///   - debugMode: Whether to print timing info
/// - Returns: Resized pixel buffer or nil on failure
func resizePixelBufferToSquare(_ src: CVPixelBuffer, size: Int = 960, debugMode: Bool = false) -> CVPixelBuffer? {
    let t0 = Date()

    CVPixelBufferLockBaseAddress(src, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }

    let srcW = CVPixelBufferGetWidth(src)
    let srcH = CVPixelBufferGetHeight(src)

    var dstOpt: CVPixelBuffer?
    let status = CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, nil, &dstOpt)
    guard status == kCVReturnSuccess, let dst = dstOpt else { return nil }

    CVPixelBufferLockBaseAddress(dst, [])
    defer { CVPixelBufferUnlockBaseAddress(dst, []) }

    guard let srcBase = CVPixelBufferGetBaseAddress(src),
          let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

    var srcBuffer = vImage_Buffer(data: srcBase,
                                  height: vImagePixelCount(srcH),
                                  width: vImagePixelCount(srcW),
                                  rowBytes: CVPixelBufferGetBytesPerRow(src))
    var dstBuffer = vImage_Buffer(data: dstBase,
                                  height: vImagePixelCount(size),
                                  width: vImagePixelCount(size),
                                  rowBytes: CVPixelBufferGetBytesPerRow(dst))

    let err = vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(0))
    guard err == kvImageNoError else { return nil }

    if debugMode {
        let dt = Date().timeIntervalSince(t0) * 1000.0
        print(String(format: "⏱ letterbox %dx%d → %dx%d: %.2f ms",
                     srcW, srcH, size, size, dt))
    }

    return dst
}

/// Convert pixel buffer to MLMultiArray for model input
/// - Parameters:
///   - pixelBuffer: Source pixel buffer (960x960 expected)
///   - debugMode: Whether to print timing info
/// - Returns: MLMultiArray with shape [1, 3, 960, 960] or nil on failure
func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer, debugMode: Bool = false) -> MLMultiArray? {
    let t0 = Date()
    guard let array = try? MLMultiArray(shape: [1, 3, 960, 960], dataType: .float32) else { return nil }
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let width = 960
    let height = 960
    let pixelCount = width * height
    let src = baseAddress.assumingMemoryBound(to: UInt8.self)

    let floatSize = MemoryLayout<Float32>.size
    let planeStrideBytes = pixelCount * floatSize
    let rPtr = array.dataPointer.advanced(by: 0 * planeStrideBytes).assumingMemoryBound(to: Float32.self)
    let gPtr = array.dataPointer.advanced(by: 1 * planeStrideBytes).assumingMemoryBound(to: Float32.self)
    let bPtr = array.dataPointer.advanced(by: 2 * planeStrideBytes).assumingMemoryBound(to: Float32.self)

    var indicesR = [vDSP_Length](repeating: 0, count: width)
    var indicesG = [vDSP_Length](repeating: 0, count: width)
    var indicesB = [vDSP_Length](repeating: 0, count: width)
    for i in 0..<width {
        indicesR[i] = vDSP_Length(2 + i * 4)
        indicesG[i] = vDSP_Length(1 + i * 4)
        indicesB[i] = vDSP_Length(0 + i * 4)
    }

    var rowUInt8 = [UInt8](repeating: 0, count: width * 4)
    var rowFloat = [Float](repeating: 0, count: width * 4)

    var scaleF: Float = 1.0 / 255.0

    for y in 0..<height {
        let rowStart = src.advanced(by: y * bytesPerRow)
        memcpy(&rowUInt8, rowStart, width * 4)

        rowUInt8.withUnsafeBufferPointer { u8Ptr in
            rowFloat.withUnsafeMutableBufferPointer { fPtr in
                vDSP_vfltu8(u8Ptr.baseAddress!, 1, fPtr.baseAddress!, 1, vDSP_Length(width * 4))
                vDSP_vsmul(fPtr.baseAddress!, 1, &scaleF, fPtr.baseAddress!, 1, vDSP_Length(width * 4))
            }
        }

        rowFloat.withUnsafeBufferPointer { rf in
            let baseF = rf.baseAddress!
            vDSP_vgathr(baseF, indicesR, 1, rPtr.advanced(by: y * width), 1, vDSP_Length(width))
            vDSP_vgathr(baseF, indicesG, 1, gPtr.advanced(by: y * width), 1, vDSP_Length(width))
            vDSP_vgathr(baseF, indicesB, 1, bPtr.advanced(by: y * width), 1, vDSP_Length(width))
        }
    }

    if debugMode {
        let dt = Date().timeIntervalSince(t0) * 1000.0
        print(String(format: "⏱ pixelBufferToMLMultiArray: %.2f ms", dt))
    }

    return array
}

/// Crop a pixel buffer to a bounding box region
/// - Parameters:
///   - pixelBuffer: Source pixel buffer
///   - det: Detection with bounding box info
///   - padding: Padding ratio around the box (default 0.1)
///   - debugMode: Whether to print timing info
/// - Returns: Cropped pixel buffer or nil on failure
func cropPixelBuffer(_ pixelBuffer: CVPixelBuffer, toBBox det: DetectionSmarty, padding: Float = 0.1, debugMode: Bool = false) -> CVPixelBuffer? {
    let cropStart = Date()

    let fullWf = Float(CVPixelBufferGetWidth(pixelBuffer))
    let fullHf = Float(CVPixelBufferGetHeight(pixelBuffer))

    let scaleX = fullWf / 960.0
    let scaleY = fullHf / 960.0

    let centerX = det.x * scaleX
    let centerY = det.y * scaleY
    let boxW = det.width * scaleX
    let boxH = det.height * scaleY

    let padW = boxW * padding
    let padH = boxH * padding

    var x1 = centerX - boxW / 2 - padW
    var y1 = centerY - boxH / 2 - padH
    var x2 = centerX + boxW / 2 + padW
    var y2 = centerY + boxH / 2 + padH

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(fullWf, x2)
    y2 = min(fullHf, y2)

    let cropW = Int(x2 - x1)
    let cropH = Int(y2 - y1)

    guard cropW > 10 && cropH > 10 else { return nil }

    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

    guard let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
    let srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

    var out: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, cropW, cropH, kCVPixelFormatType_32BGRA, nil, &out)
    guard status == kCVReturnSuccess, let dst = out else { return nil }

    CVPixelBufferLockBaseAddress(dst, [])
    defer { CVPixelBufferUnlockBaseAddress(dst, []) }
    guard let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }
    let dstBytesPerRow = CVPixelBufferGetBytesPerRow(dst)

    let x1Int = Int(x1)
    let y1Int = Int(y1)
    let srcOffsetPtr = srcBase.advanced(by: y1Int * srcBytesPerRow + x1Int * 4)

    var srcBuf = vImage_Buffer(
        data: srcOffsetPtr,
        height: vImagePixelCount(cropH),
        width: vImagePixelCount(cropW),
        rowBytes: srcBytesPerRow
    )
    var dstBuf = vImage_Buffer(
        data: dstBase,
        height: vImagePixelCount(cropH),
        width: vImagePixelCount(cropW),
        rowBytes: dstBytesPerRow
    )

    let copyErr = vImageCopyBuffer(&srcBuf, &dstBuf, 4, vImage_Flags(kvImageNoFlags))
    if copyErr != kvImageNoError {
        let scaleErr = vImageScale_ARGB8888(&srcBuf, &dstBuf, nil, vImage_Flags(kvImageNoFlags))
        if scaleErr != kvImageNoError {
            let srcPtr = srcBase.assumingMemoryBound(to: UInt8.self)
            let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)
            for row in 0..<cropH {
                let s = (y1Int + row) * srcBytesPerRow + x1Int * 4
                let d = row * dstBytesPerRow
                memcpy(dstPtr + d, srcPtr + s, cropW * 4)
            }
        }
    }

    if debugMode {
        let dt = Date().timeIntervalSince(cropStart) * 1000.0
        print(String(format: "⏱ cropPixelBuffer: %.2f ms (rect %dx%d)", dt, cropW, cropH))
    }

    return dst
}

/// Calculate average brightness (luma) of a pixel buffer
/// - Parameters:
///   - pixelBuffer: Source pixel buffer
///   - sampleStride: Sample every Nth pixel (default 8 for performance)
/// - Returns: Average luma value (0.0 to 1.0)
func averageLuma(of pixelBuffer: CVPixelBuffer, sampleStride: Int = 8) -> Float {
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return 0 }
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

    let ptr = base.assumingMemoryBound(to: UInt8.self)
    var sum: Float = 0
    var count: Int = 0

    // Sample every Nth pixel to reduce cost
    let step = max(1, sampleStride)
    var y = 0
    while y < height {
        let row = ptr.advanced(by: y * bytesPerRow)
        var x = 0
        while x < width {
            let px = row.advanced(by: x * 4)
            let b = Float(px[0]) * (1.0 / 255.0)
            let g = Float(px[1]) * (1.0 / 255.0)
            let r = Float(px[2]) * (1.0 / 255.0)
            // Rec. 709 luma
            let y709 = 0.2126 * r + 0.7152 * g + 0.0722 * b
            sum += Float(y709)
            count += 1
            x += step
        }
        y += step
    }
    if count == 0 { return 0 }
    return sum / Float(count)
}

/// Create a fully transparent image of given dimensions
func makeTransparentImage(width: Int, height: Int) -> CGImage? {
    let bytesPerRow = width * 4
    let bufSize = bytesPerRow * height
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }

    guard let data = malloc(bufSize) else { return nil }
    memset(data, 0, bufSize)

    guard let ctx = CGContext(data: data,
                              width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bytesPerRow: bytesPerRow,
                              space: colorSpace,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
        free(data)
        return nil
    }

    let image = ctx.makeImage()
    free(data)
    return image
}
