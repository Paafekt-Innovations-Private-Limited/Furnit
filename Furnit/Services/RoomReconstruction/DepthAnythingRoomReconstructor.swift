import CoreGraphics
import CoreML
import Foundation
import ImageIO
import ModelIO
import SceneKit
import UIKit
@preconcurrency import Vision

struct DepthAnythingRoomResult: Sendable {
    let usdzURL: URL
    let vertexCount: Int
    let triangleCount: Int
    let imageWidth: Int
    let imageHeight: Int
    let roomWidthMeters: Float
    let roomHeightMeters: Float
    let roomDepthMeters: Float

    var photoOrientation: PhotoOrientation {
        if imageWidth == imageHeight { return .square }
        return imageWidth > imageHeight ? .landscape : .portrait
    }

    var summary: String {
        String(
            format: "vertices=%d triangles=%d image=%dx%d dims=%.2fx%.2fx%.2fm usdz=%@",
            vertexCount,
            triangleCount,
            imageWidth,
            imageHeight,
            roomWidthMeters,
            roomHeightMeters,
            roomDepthMeters,
            usdzURL.lastPathComponent
        )
    }
}

final class DepthAnythingRoomReconstructor {
    let model: VNCoreMLModel

    private let modelName: String
    private let pixelStep: Int
    private let depthDiscontinuityThresholdMeters: Float
    private let maxReconstructionImageDimension: Int
    private let outputDirectory: URL?
    private let wallMargin: Float

    /// Matches `scripts/depthanything_measure_room.py --flat-mesh` (photo on a flat plane).
    private static let usesFlatMesh = true
    private static let minimumRoomWidthMeters: Float = 2.0
    private static let fallbackFocal35mmEquivalent: Float = 28.0

    init(
        pixelStep: Int = 2,
        depthDiscontinuityThresholdMeters: Float = 0.15,
        maxReconstructionImageDimension: Int = 1600,
        outputDirectory: URL? = nil,
        wallMargin: Float = 0.05
    ) throws {
        self.pixelStep = max(1, pixelStep)
        self.depthDiscontinuityThresholdMeters = depthDiscontinuityThresholdMeters
        self.maxReconstructionImageDimension = max(256, maxReconstructionImageDimension)
        self.outputDirectory = outputDirectory
        self.wallMargin = min(max(wallMargin, 0), 0.45)

        let config = MLModelConfiguration()
        config.computeUnits = .all
        let loaded = try Self.loadDepthAnythingModel(configuration: config)
        self.modelName = loaded.name
        self.model = try VNCoreMLModel(for: loaded.model)
    }

    func reconstruct(image: UIImage) async throws -> URL {
        let result = try await reconstructWithResult(image: image)
        return result.usdzURL
    }

    func reconstructWithResult(image: UIImage) async throws -> DepthAnythingRoomResult {
        let fixedImage = image.fixedOrientation()
        let workingImage = try Self.downsampledImage(fixedImage, maxDimension: maxReconstructionImageDimension)
        let depthMap = try await inferDepth(image: workingImage)
        let imageWidth = depthMap.first?.count ?? 0
        let imageHeight = depthMap.count
        let focal = Self.focalPixels(image: workingImage, imageWidth: imageWidth, imageHeight: imageHeight)
        let measured = Self.measureWall(
            depthMap: depthMap,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            fx: focal.fx,
            fy: focal.fy,
            wallMargin: wallMargin
        )
        let meshRoomWidthMeters = max(measured.width, Self.minimumRoomWidthMeters)
        let meshRoomHeightMeters = meshRoomWidthMeters * Float(imageHeight) / Float(max(imageWidth, 1))
        let meshDepthMap = Self.usesFlatMesh ? Self.flattenDepthForMesh(depthMap) : depthMap
        let mesh = try buildMesh(
            image: workingImage,
            depthMap: meshDepthMap,
            roomWidthMeters: meshRoomWidthMeters
        )
        let url = try exportUSDZ(mesh: mesh, textureImage: workingImage)
        return DepthAnythingRoomResult(
            usdzURL: url,
            vertexCount: mesh.vertexCount,
            triangleCount: mesh.submeshes?.compactMap { $0 as? MDLSubmesh }.reduce(0) { $0 + $1.indexCount / 3 } ?? 0,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            roomWidthMeters: meshRoomWidthMeters,
            roomHeightMeters: meshRoomHeightMeters,
            roomDepthMeters: measured.depth
        )
    }

    func inferDepth(image: UIImage) async throws -> [[Float]] {
        let fixedImage = image.fixedOrientation()
        guard let cgImage = fixedImage.cgImage else {
            throw DepthAnythingRoomError.invalidImage
        }

        let targetWidth = cgImage.width
        let targetHeight = cgImage.height
        let observation = try await runVisionDepthRequest(cgImage: cgImage)
        let dense = try Self.depthGrid(from: observation)
        let remapped = Self.resizeBilinear(
            values: dense.values,
            width: dense.width,
            height: dense.height,
            targetWidth: targetWidth,
            targetHeight: targetHeight
        )

        var rows: [[Float]] = []
        rows.reserveCapacity(targetHeight)
        for row in 0..<targetHeight {
            let start = row * targetWidth
            rows.append(Array(remapped[start..<start + targetWidth]))
        }
        return rows
    }

    func buildMesh(image: UIImage, depthMap: [[Float]], roomWidthMeters: Float) throws -> MDLMesh {
        let fixedImage = image.fixedOrientation()
        let raster = try DepthAnythingRasterImage(image: fixedImage)
        let imageWidth = raster.width
        let imageHeight = raster.height
        guard depthMap.count == imageHeight,
              depthMap.allSatisfy({ $0.count == imageWidth }) else {
            throw DepthAnythingRoomError.depthImageSizeMismatch
        }

        var depthMax: Float = 0
        for row in depthMap {
            for depth in row where depth.isFinite {
                depthMax = max(depthMax, depth)
            }
        }
        guard depthMax > 0 else {
            throw DepthAnythingRoomError.invalidDepthOutput
        }

        let sampledRows = Array(stride(from: 0, to: imageHeight, by: pixelStep))
        let sampledColumns = Array(stride(from: 0, to: imageWidth, by: pixelStep))
        let rowCount = sampledRows.count
        let columnCount = sampledColumns.count
        var vertexIndices = [Int32](repeating: -1, count: rowCount * columnCount)
        var vertexData = Data()
        vertexData.reserveCapacity(rowCount * columnCount * DepthAnythingVertex.byteStride)
        var vertexCount = 0

        let pixelScale = roomWidthMeters / Float(imageWidth)
        let centerX = Float(imageWidth) / 2.0
        let centerY = Float(imageHeight) / 2.0

        for (sampledRowIndex, row) in sampledRows.enumerated() {
            for (sampledColumnIndex, column) in sampledColumns.enumerated() {
                let depth = depthMap[row][column]
                guard depth.isFinite, depth > 0 else { continue }

                let x = -(Float(column) - centerX) * pixelScale
                let y = (Float(row) - centerY) * pixelScale
                let z = -(depthMax - depth)
                let color = raster.color(x: column, y: row).floatRGB
                let u = Float(column) / Float(max(imageWidth - 1, 1))
                let v = 1.0 - Float(row) / Float(max(imageHeight - 1, 1))

                vertexData.appendFloat32LE(x)
                vertexData.appendFloat32LE(y)
                vertexData.appendFloat32LE(z)
                vertexData.appendFloat32LE(color.x)
                vertexData.appendFloat32LE(color.y)
                vertexData.appendFloat32LE(color.z)
                vertexData.appendFloat32LE(u)
                vertexData.appendFloat32LE(v)
                vertexIndices[sampledRowIndex * columnCount + sampledColumnIndex] = Int32(vertexCount)
                vertexCount += 1
            }
        }

        guard vertexCount > 0 else {
            throw DepthAnythingRoomError.emptyMesh
        }

        var indexData = Data()
        indexData.reserveCapacity(max(0, (rowCount - 1) * (columnCount - 1) * 6 * MemoryLayout<UInt32>.size))
        var indexCount = 0

        func sampledIndex(_ row: Int, _ column: Int) -> Int {
            row * columnCount + column
        }

        for rowIndex in 0..<(rowCount - 1) {
            for columnIndex in 0..<(columnCount - 1) {
                let i00 = sampledIndex(rowIndex, columnIndex)
                let i10 = sampledIndex(rowIndex, columnIndex + 1)
                let i01 = sampledIndex(rowIndex + 1, columnIndex)
                let i11 = sampledIndex(rowIndex + 1, columnIndex + 1)
                let v00 = vertexIndices[i00]
                let v10 = vertexIndices[i10]
                let v01 = vertexIndices[i01]
                let v11 = vertexIndices[i11]
                guard v00 >= 0, v10 >= 0, v01 >= 0, v11 >= 0 else { continue }

                let r0 = sampledRows[rowIndex]
                let r1 = sampledRows[rowIndex + 1]
                let c0 = sampledColumns[columnIndex]
                let c1 = sampledColumns[columnIndex + 1]
                let d00 = depthMap[r0][c0]
                let d10 = depthMap[r0][c1]
                let d01 = depthMap[r1][c0]
                let d11 = depthMap[r1][c1]
                guard Self.depthsAreContinuous(
                    d00,
                    d10,
                    d01,
                    d11,
                    threshold: depthDiscontinuityThresholdMeters
                ) else {
                    continue
                }

                indexData.appendUInt32LE(UInt32(v00))
                indexData.appendUInt32LE(UInt32(v10))
                indexData.appendUInt32LE(UInt32(v11))
                indexData.appendUInt32LE(UInt32(v00))
                indexData.appendUInt32LE(UInt32(v11))
                indexData.appendUInt32LE(UInt32(v01))
                indexCount += 6
            }
        }

        guard indexCount >= 3 else {
            throw DepthAnythingRoomError.emptyMesh
        }

        let allocator = MDLMeshBufferDataAllocator()
        let vertexBuffer = allocator.newBuffer(with: vertexData, type: .vertex)
        let indexBuffer = allocator.newBuffer(with: indexData, type: .index)

        let descriptor = MDLVertexDescriptor()
        descriptor.attributes[0] = MDLVertexAttribute(
            name: MDLVertexAttributePosition,
            format: .float3,
            offset: 0,
            bufferIndex: 0
        )
        descriptor.attributes[1] = MDLVertexAttribute(
            name: MDLVertexAttributeColor,
            format: .float3,
            offset: 12,
            bufferIndex: 0
        )
        descriptor.attributes[2] = MDLVertexAttribute(
            name: MDLVertexAttributeTextureCoordinate,
            format: .float2,
            offset: 24,
            bufferIndex: 0
        )
        descriptor.layouts[0] = MDLVertexBufferLayout(stride: DepthAnythingVertex.byteStride)

        let submesh = MDLSubmesh(
            indexBuffer: indexBuffer,
            indexCount: indexCount,
            indexType: .uInt32,
            geometryType: .triangles,
            material: nil
        )
        let mesh = MDLMesh(
            vertexBuffer: vertexBuffer,
            vertexCount: vertexCount,
            descriptor: descriptor,
            submeshes: [submesh]
        )
        mesh.name = "DepthAnythingMetricRoom"
        return mesh
    }

    func exportUSDZ(mesh: MDLMesh, textureImage: UIImage) throws -> URL {
        let directory = try resolvedOutputDirectory()
        let url = directory.appendingPathComponent("DepthAnythingRoom_\(Self.outputStamp()).usdz")

        var exportErrors: [String] = []
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
            let scene = try Self.makeSceneKitScene(from: mesh, textureImage: textureImage)
            let didWrite = scene.write(to: url, options: nil, delegate: nil, progressHandler: nil)
            guard didWrite, FileManager.default.fileExists(atPath: url.path) else {
                throw DepthAnythingRoomError.exportFailed("SceneKit writer returned false.")
            }
            return url
        } catch {
            exportErrors.append("SceneKit: \(error.localizedDescription)")
        }

        if MDLAsset.canExportFileExtension("usdz") {
            do {
                let asset = MDLAsset(bufferAllocator: MDLMeshBufferDataAllocator())
                asset.add(mesh)
                try asset.export(to: url)
                return url
            } catch {
                exportErrors.append("ModelIO: \(error.localizedDescription)")
            }
        } else {
            exportErrors.append("ModelIO cannot export USDZ on this platform")
        }

        throw DepthAnythingRoomError.exportFailed(exportErrors.joined(separator: "; "))
    }

    private static func makeSceneKitScene(from mesh: MDLMesh, textureImage: UIImage) throws -> SCNScene {
        guard let vertexBuffer = mesh.vertexBuffers.first else {
            throw DepthAnythingRoomError.emptyMesh
        }
        let vertexData = Data(
            bytes: vertexBuffer.map().bytes,
            count: mesh.vertexCount * DepthAnythingVertex.byteStride
        )
        let positionSource = SCNGeometrySource(
            data: vertexData,
            semantic: .vertex,
            vectorCount: mesh.vertexCount,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: DepthAnythingVertex.byteStride
        )
        let colorSource = SCNGeometrySource(
            data: vertexData,
            semantic: .color,
            vectorCount: mesh.vertexCount,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 3 * MemoryLayout<Float>.size,
            dataStride: DepthAnythingVertex.byteStride
        )
        let texcoordSource = SCNGeometrySource(
            data: vertexData,
            semantic: .texcoord,
            vectorCount: mesh.vertexCount,
            usesFloatComponents: true,
            componentsPerVector: 2,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 6 * MemoryLayout<Float>.size,
            dataStride: DepthAnythingVertex.byteStride
        )

        let elements = try (mesh.submeshes as? [MDLSubmesh] ?? []).map { submesh -> SCNGeometryElement in
            guard submesh.indexType == .uInt32 else {
                throw DepthAnythingRoomError.exportFailed("SceneKit fallback expected UInt32 mesh indices.")
            }
            let indexData = Data(
                bytes: submesh.indexBuffer.map().bytes,
                count: submesh.indexCount * MemoryLayout<UInt32>.size
            )
            return SCNGeometryElement(
                data: indexData,
                primitiveType: .triangles,
                primitiveCount: submesh.indexCount / 3,
                bytesPerIndex: MemoryLayout<UInt32>.size
            )
        }
        guard !elements.isEmpty else {
            throw DepthAnythingRoomError.emptyMesh
        }

        let geometry = SCNGeometry(sources: [positionSource, colorSource, texcoordSource], elements: elements)
        let material = SCNMaterial()
        material.diffuse.contents = textureImage
        material.emission.contents = textureImage
        material.lightingModel = .constant
        material.isDoubleSided = true
        material.diffuse.wrapS = .clamp
        material.diffuse.wrapT = .clamp
        geometry.materials = [material]

        let scene = SCNScene()
        let node = SCNNode(geometry: geometry)
        node.name = "DepthAnythingMetricRoom"
        scene.rootNode.addChildNode(node)
        return scene
    }

    private func runVisionDepthRequest(cgImage: CGImage) async throws -> VNObservation {
        try await withCheckedThrowingContinuation { continuation in
            let request = VNCoreMLRequest(model: model) { request, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let observation = request.results?.first else {
                    continuation.resume(throwing: DepthAnythingRoomError.invalidDepthOutput)
                    return
                }
                continuation.resume(returning: observation)
            }
            request.imageCropAndScaleOption = .scaleFill

            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:]).perform([request])
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func loadDepthAnythingModel(configuration: MLModelConfiguration) throws -> (model: MLModel, name: String) {
        let candidates = [
            "DepthAnythingV2MetricIndoorSmall",
            "DepthAnythingV2MetricIndoorSmallF16",
            "depthanythingv2metricindoorsmall",
            "depthanythingv2metricindoorsmallf16",
        ]
        let extensions = ["mlmodelc", "mlpackage", "mlmodel"]

        for sourceURL in candidateModelURLs(baseNames: candidates, extensions: extensions) {
            let ext = sourceURL.pathExtension
            guard !ext.isEmpty else { continue }
            do {
                let modelURL = ext == "mlpackage" || ext == "mlmodel"
                    ? try MLModel.compileModel(at: sourceURL)
                    : sourceURL
                return (try MLModel(contentsOf: modelURL, configuration: configuration), sourceURL.lastPathComponent)
            } catch {
                continue
            }
        }

        throw DepthAnythingRoomError.modelNotFound
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

    private static func depthGrid(from observation: VNObservation) throws -> DenseDepthGrid {
        if let featureObservation = observation as? VNCoreMLFeatureValueObservation,
           let multiArray = featureObservation.featureValue.multiArrayValue {
            let dense = try denseArray(from: multiArray)
            guard dense.shape.count >= 2 else {
                throw DepthAnythingRoomError.invalidDepthOutput
            }
            let height = dense.shape[dense.shape.count - 2]
            let width = dense.shape[dense.shape.count - 1]
            guard width > 0, height > 0, dense.values.count >= width * height else {
                throw DepthAnythingRoomError.invalidDepthOutput
            }
            let suffix = Array(dense.values.suffix(width * height))
            return DenseDepthGrid(width: width, height: height, values: suffix)
        }

        if let pixelObservation = observation as? VNPixelBufferObservation {
            return try denseImage(from: pixelObservation.pixelBuffer)
        }

        throw DepthAnythingRoomError.invalidDepthOutput
    }

    private static func denseArray(from multiArray: MLMultiArray) throws -> DenseDepthArray {
        let shape = multiArray.shape.map(\.intValue)
        let count = multiArray.count
        guard count > 0 else {
            throw DepthAnythingRoomError.invalidDepthOutput
        }

        if isRowMajorContiguous(multiArray) {
            switch multiArray.dataType {
            case .float32:
                let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
                return DenseDepthArray(shape: shape, values: Array(UnsafeBufferPointer(start: ptr, count: count)))
            case .float16:
                let ptr = multiArray.dataPointer.bindMemory(to: UInt16.self, capacity: count)
                let values = UnsafeBufferPointer(start: ptr, count: count).map { Float(Float16(bitPattern: $0)) }
                return DenseDepthArray(shape: shape, values: values)
            case .double:
                let ptr = multiArray.dataPointer.bindMemory(to: Double.self, capacity: count)
                return DenseDepthArray(shape: shape, values: UnsafeBufferPointer(start: ptr, count: count).map(Float.init))
            default:
                break
            }
        }

        var indices = [Int](repeating: 0, count: shape.count)
        var values: [Float] = []
        values.reserveCapacity(count)

        func visit(_ dimension: Int) {
            if dimension == shape.count {
                values.append(multiArray[indices.map(NSNumber.init(value:))].floatValue)
                return
            }
            for index in 0..<shape[dimension] {
                indices[dimension] = index
                visit(dimension + 1)
            }
        }
        visit(0)
        return DenseDepthArray(shape: shape, values: values)
    }

    private static func isRowMajorContiguous(_ multiArray: MLMultiArray) -> Bool {
        let shape = multiArray.shape.map(\.intValue)
        let strides = multiArray.strides.map(\.intValue)
        guard shape.count == strides.count else { return false }
        var expected = 1
        for index in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[index] != expected { return false }
            expected *= max(shape[index], 1)
        }
        return true
    }

    private static func denseImage(from pixelBuffer: CVPixelBuffer) throws -> DenseDepthGrid {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width > 0, height > 0,
              let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DepthAnythingRoomError.invalidDepthOutput
        }

        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        var values = [Float](repeating: 0, count: width * height)

        switch format {
        case kCVPixelFormatType_DepthFloat32, kCVPixelFormatType_OneComponent32Float:
            for y in 0..<height {
                let row = baseAddress.advanced(by: y * rowBytes).assumingMemoryBound(to: Float.self)
                for x in 0..<width {
                    values[y * width + x] = row[x]
                }
            }
        case kCVPixelFormatType_DepthFloat16, kCVPixelFormatType_OneComponent16Half:
            for y in 0..<height {
                let row = baseAddress.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt16.self)
                for x in 0..<width {
                    values[y * width + x] = Float(Float16(bitPattern: row[x]))
                }
            }
        default:
            throw DepthAnythingRoomError.invalidDepthOutput
        }

        return DenseDepthGrid(width: width, height: height, values: values)
    }

    private static func resizeBilinear(
        values: [Float],
        width: Int,
        height: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> [Float] {
        guard width != targetWidth || height != targetHeight else { return values }
        var output = [Float](repeating: 0, count: targetWidth * targetHeight)
        for y in 0..<targetHeight {
            let sourceY = (Float(y) + 0.5) * Float(height) / Float(targetHeight) - 0.5
            for x in 0..<targetWidth {
                let sourceX = (Float(x) + 0.5) * Float(width) / Float(targetWidth) - 0.5
                output[y * targetWidth + x] = bilinearSample(
                    values: values,
                    width: width,
                    height: height,
                    x: sourceX,
                    y: sourceY
                )
            }
        }
        return output
    }

    private static func downsampledImage(_ image: UIImage, maxDimension: Int) throws -> UIImage {
        guard let cgImage = image.cgImage else {
            throw DepthAnythingRoomError.invalidImage
        }

        let width = cgImage.width
        let height = cgImage.height
        let longestSide = max(width, height)
        guard longestSide > maxDimension else {
            return image
        }

        let scale = CGFloat(maxDimension) / CGFloat(longestSide)
        let targetWidth = max(1, Int((CGFloat(width) * scale).rounded()))
        let targetHeight = max(1, Int((CGFloat(height) * scale).rounded()))
        guard let context = CGContext(
            data: nil,
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw DepthAnythingRoomError.invalidImage
        }

        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
        guard let resized = context.makeImage() else {
            throw DepthAnythingRoomError.invalidImage
        }
        return UIImage(cgImage: resized, scale: 1.0, orientation: .up)
    }

    private static func bilinearSample(values: [Float], width: Int, height: Int, x: Float, y: Float) -> Float {
        let clampedX = min(max(x, 0), Float(width - 1))
        let clampedY = min(max(y, 0), Float(height - 1))
        let x0 = Int(floor(clampedX))
        let y0 = Int(floor(clampedY))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        let tx = clampedX - Float(x0)
        let ty = clampedY - Float(y0)
        let v00 = values[y0 * width + x0]
        let v10 = values[y0 * width + x1]
        let v01 = values[y1 * width + x0]
        let v11 = values[y1 * width + x1]
        let top = v00 * (1 - tx) + v10 * tx
        let bottom = v01 * (1 - tx) + v11 * tx
        return top * (1 - ty) + bottom * ty
    }

    private static func depthsAreContinuous(
        _ d00: Float,
        _ d10: Float,
        _ d01: Float,
        _ d11: Float,
        threshold: Float
    ) -> Bool {
        let depths = [d00, d10, d01, d11]
        guard depths.allSatisfy({ $0.isFinite && $0 > 0 }) else { return false }
        for lhs in 0..<depths.count {
            for rhs in (lhs + 1)..<depths.count {
                if abs(depths[lhs] - depths[rhs]) > threshold {
                    return false
                }
            }
        }
        return true
    }

    private func resolvedOutputDirectory() throws -> URL {
        let directory = outputDirectory ?? FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DepthAnythingRooms", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func outputStamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss_SSS"
        return formatter.string(from: Date())
    }

    private static func flattenDepthForMesh(_ depthMap: [[Float]]) -> [[Float]] {
        var validDepths: [Float] = []
        validDepths.reserveCapacity(depthMap.count * max(depthMap.first?.count ?? 0, 0))
        for row in depthMap {
            for depth in row where depth.isFinite && depth > 0 {
                validDepths.append(depth)
            }
        }
        guard !validDepths.isEmpty else { return depthMap }
        let planeDepth = median(validDepths)
        return depthMap.map { row in
            row.map { depth in
                depth.isFinite && depth > 0 ? planeDepth : depth
            }
        }
    }

    private static func measureWall(
        depthMap: [[Float]],
        imageWidth: Int,
        imageHeight: Int,
        fx: Float,
        fy: Float,
        wallMargin: Float
    ) -> (width: Float, height: Float, depth: Float) {
        let margin = min(max(wallMargin, 0), 0.45)
        let rectX = margin * Float(imageWidth)
        let rectY = margin * Float(imageHeight)
        let rectWidth = (1.0 - 2.0 * margin) * Float(imageWidth)
        let rectHeight = (1.0 - 2.0 * margin) * Float(imageHeight)

        let centerX = Float(imageWidth - 1) * 0.5
        let centerY = Float(imageHeight - 1) * 0.5
        let leftX = Int(round(rectX))
        let rightX = Int(round(rectX + rectWidth - 1))
        let topY = Int(round(rectY))
        let bottomY = Int(round(rectY + rectHeight - 1))
        let sampleCenterX = Int(round(rectX + rectWidth * 0.5))
        let sampleCenterY = Int(round(rectY + rectHeight * 0.5))

        guard let centerDepth = medianAt(
            depthMap: depthMap,
            x: sampleCenterX,
            y: sampleCenterY,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        ) else {
            let fallbackDepth = centerDepthFallback(depthMap)
            let fallbackWidth = max(minimumRoomWidthMeters, Float(imageWidth) / fx * fallbackDepth)
            let fallbackHeight = fallbackWidth * Float(imageHeight) / Float(max(imageWidth, 1))
            return (fallbackWidth, fallbackHeight, fallbackDepth)
        }

        let leftPlane = (Float(leftX) - centerX) * centerDepth / fx
        let rightPlane = (Float(rightX) - centerX) * centerDepth / fx
        let topPlane = (Float(topY) - centerY) * centerDepth / fy
        let bottomPlane = (Float(bottomY) - centerY) * centerDepth / fy
        return (
            abs(rightPlane - leftPlane),
            abs(bottomPlane - topPlane),
            centerDepth
        )
    }

    private static func centerDepthFallback(_ depthMap: [[Float]]) -> Float {
        var validDepths: [Float] = []
        for row in depthMap {
            for depth in row where depth.isFinite && depth > 0 {
                validDepths.append(depth)
            }
        }
        return validDepths.isEmpty ? 3.0 : median(validDepths)
    }

    private static func medianAt(
        depthMap: [[Float]],
        x: Int,
        y: Int,
        imageWidth: Int,
        imageHeight: Int,
        radius: Int = 5
    ) -> Float? {
        let clampedX = min(max(x, 0), imageWidth - 1)
        let clampedY = min(max(y, 0), imageHeight - 1)
        var samples: [Float] = []
        let y0 = max(0, clampedY - radius)
        let y1 = min(imageHeight, clampedY + radius + 1)
        let x0 = max(0, clampedX - radius)
        let x1 = min(imageWidth, clampedX + radius + 1)
        for row in y0..<y1 {
            for column in x0..<x1 {
                let depth = depthMap[row][column]
                if depth.isFinite, depth > 0 {
                    samples.append(depth)
                }
            }
        }
        guard !samples.isEmpty else { return nil }
        return median(samples)
    }

    private static func focalPixels(image: UIImage, imageWidth: Int, imageHeight: Int) -> (fx: Float, fy: Float) {
        if let focal35mm = focal35mmEquivalent(from: image), focal35mm > 1 {
            let fx = (focal35mm / 36.0) * Float(imageWidth)
            let fy = fx * Float(imageHeight) / Float(max(imageWidth, 1))
            return (fx, fy)
        }
        let fx = (fallbackFocal35mmEquivalent / 36.0) * Float(imageWidth)
        let fy = fx * Float(imageHeight) / Float(max(imageWidth, 1))
        return (fx, fy)
    }

    private static func focal35mmEquivalent(from image: UIImage) -> Float? {
        guard let data = image.jpegData(compressionQuality: 0.95) else { return nil }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any] else {
            return nil
        }
        if let value = exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? NSNumber, value.floatValue > 1 {
            return value.floatValue
        }
        if let value = exif["FocalLenIn35mmFilm" as CFString] as? NSNumber, value.floatValue > 1 {
            return value.floatValue
        }
        return nil
    }

    private static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) * 0.5
        }
        return sorted[middle]
    }
}

private enum DepthAnythingVertex {
    static let byteStride = 8 * MemoryLayout<Float>.size
}

private struct DenseDepthArray {
    let shape: [Int]
    let values: [Float]
}

private struct DenseDepthGrid {
    let width: Int
    let height: Int
    let values: [Float]
}

private struct DepthAnythingRasterImage {
    let width: Int
    let height: Int
    let pixels: [UInt8]

    init(image: UIImage) throws {
        guard let cgImage = image.cgImage else {
            throw DepthAnythingRoomError.invalidImage
        }
        width = cgImage.width
        height = cgImage.height
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
            throw DepthAnythingRoomError.invalidImage
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        pixels = rgba
    }

    func color(x: Int, y: Int) -> DepthAnythingRGB {
        let clampedX = min(max(x, 0), width - 1)
        let clampedY = min(max(y, 0), height - 1)
        let offset = (clampedY * width + clampedX) * 4
        return DepthAnythingRGB(r: pixels[offset], g: pixels[offset + 1], b: pixels[offset + 2])
    }
}

private struct DepthAnythingRGB {
    let r: UInt8
    let g: UInt8
    let b: UInt8

    var floatRGB: SIMD3<Float> {
        SIMD3(Float(r) / 255, Float(g) / 255, Float(b) / 255)
    }
}

enum DepthAnythingRoomError: LocalizedError, Equatable {
    case modelNotFound
    case invalidImage
    case invalidDepthOutput
    case depthImageSizeMismatch
    case emptyMesh
    case exportFailed(String? = nil)

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "DepthAnythingV2MetricIndoorSmall was not found in the app bundle. Add the .mlpackage or .mlmodelc to the iOS target."
        case .invalidImage:
            return "The selected image could not be read."
        case .invalidDepthOutput:
            return "Depth Anything returned an invalid depth output."
        case .depthImageSizeMismatch:
            return "Depth map size does not match the source image."
        case .emptyMesh:
            return "Depth Anything did not produce enough connected room geometry."
        case .exportFailed(let reason):
            if let reason, !reason.isEmpty {
                return "Could not export the reconstructed room as USDZ. \(reason)"
            }
            return "Could not export the reconstructed room as USDZ."
        }
    }
}

private extension Data {
    mutating func appendFloat32LE(_ value: Float) {
        var bits = value.bitPattern.littleEndian
        Swift.withUnsafeBytes(of: &bits) { append(contentsOf: $0) }
    }

    mutating func appendUInt32LE(_ value: UInt32) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }
}
