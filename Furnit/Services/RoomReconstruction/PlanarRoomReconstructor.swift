import Accelerate
import Foundation
import SceneKit
import simd
import UIKit

struct PlanarRoomResult: Sendable {
    let plyURL: URL
    let objURL: URL
    let pointCount: Int
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
            format: "points=%d image=%dx%d dims=%.2fx%.2fx%.2fm obj=%@",
            pointCount,
            imageWidth,
            imageHeight,
            roomWidthMeters,
            roomHeightMeters,
            roomDepthMeters,
            objURL.lastPathComponent
        )
    }
}

enum PlanarRoomReconstructor {
    struct Configuration: Sendable {
        var longestSideMeters: Float = 3.0
        var pixelStep: Int = 2
        var voxelSizeMeters: Float = 0.005
        var quadSizeMeters: Float = 0.005
        var previewDepthMeters: Float = 0.06
        var outputDirectory: URL? = nil

        static let `default` = Configuration()
    }

    static func reconstruct(
        image: UIImage,
        configuration: Configuration = .default
    ) throws -> PlanarRoomResult {
        let raster = try RasterImage(image: image.fixedOrientation())
        let width = raster.width
        let height = raster.height
        guard width > 0, height > 0 else {
            throw PlanarRoomError.invalidImage
        }

        let longestSide = max(width, height)
        let metersPerPixel = configuration.longestSideMeters / Float(longestSide)
        let centerX = Float(width) / 2.0
        let centerY = Float(height) / 2.0
        let step = max(1, configuration.pixelStep)

        let sampledColumns = Array(stride(from: 0, to: width, by: step)).map(Float.init)
        let sampledRows = Array(stride(from: 0, to: height, by: step)).map(Float.init)
        var xCoordinates = [Float](repeating: 0, count: sampledColumns.count)
        var yCoordinates = [Float](repeating: 0, count: sampledRows.count)

        var xScale = metersPerPixel
        var xOffset = -centerX * metersPerPixel
        var yScale = -metersPerPixel
        var yOffset = centerY * metersPerPixel
        vDSP_vsmsa(
            sampledColumns,
            1,
            &xScale,
            &xOffset,
            &xCoordinates,
            1,
            vDSP_Length(sampledColumns.count)
        )
        vDSP_vsmsa(
            sampledRows,
            1,
            &yScale,
            &yOffset,
            &yCoordinates,
            1,
            vDSP_Length(sampledRows.count)
        )

        var occupied = Set<PlanarRoomVoxelKey>()
        occupied.reserveCapacity(sampledColumns.count * sampledRows.count)
        var points: [PlanarRoomPoint] = []
        points.reserveCapacity(min(sampledColumns.count * sampledRows.count, 300_000))

        let voxelSize = max(configuration.voxelSizeMeters, 0.0001)
        for (rowIndex, rowFloat) in sampledRows.enumerated() {
            let yPixel = Int(rowFloat)
            let yWorld = yCoordinates[rowIndex]
            for (columnIndex, columnFloat) in sampledColumns.enumerated() {
                let xPixel = Int(columnFloat)
                let xWorld = xCoordinates[columnIndex]
                let key = PlanarRoomVoxelKey(
                    x: Int(floor(xWorld / voxelSize)),
                    y: Int(floor(yWorld / voxelSize)),
                    z: 0
                )
                guard occupied.insert(key).inserted else {
                    continue
                }
                points.append(
                    PlanarRoomPoint(
                        position: SIMD3<Float>(xWorld, yWorld, 0),
                        color: raster.average2x2Color(x: xPixel, y: yPixel)
                    )
                )
            }
        }

        guard !points.isEmpty else {
            throw PlanarRoomError.noPoints
        }

        let outputDirectory = try resolvedOutputDirectory(configuration.outputDirectory)
        let stamp = outputStamp()
        let plyURL = outputDirectory.appendingPathComponent("PlanarRoom_\(stamp).ply")
        let objURL = outputDirectory.appendingPathComponent("PlanarRoom_\(stamp).obj")

        try writeGaussianPLY(points: points, quadSizeMeters: configuration.quadSizeMeters, to: plyURL)
        try writeOBJ(points: points, quadSizeMeters: configuration.quadSizeMeters, to: objURL)

        let roomWidth = Float(width) * metersPerPixel
        let roomHeight = Float(height) * metersPerPixel
        return PlanarRoomResult(
            plyURL: plyURL,
            objURL: objURL,
            pointCount: points.count,
            imageWidth: width,
            imageHeight: height,
            roomWidthMeters: roomWidth,
            roomHeightMeters: roomHeight,
            roomDepthMeters: max(configuration.previewDepthMeters, 0.051)
        )
    }

    static func makeNode(fromOBJAt url: URL) throws -> SCNNode {
        let scene = try SCNScene(url: url, options: nil)
        let node = SCNNode()
        for child in scene.rootNode.childNodes {
            node.addChildNode(child)
        }
        return node
    }

    private static func resolvedOutputDirectory(_ override: URL?) throws -> URL {
        let directory = override ?? FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("PlanarRooms", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func outputStamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss_SSS"
        return formatter.string(from: Date())
    }

    private static func writeGaussianPLY(
        points: [PlanarRoomPoint],
        quadSizeMeters: Float,
        to url: URL
    ) throws {
        let header = """
        ply
        format binary_little_endian 1.0
        element vertex \(points.count)
        property float x
        property float y
        property float z
        property float scale_0
        property float scale_1
        property float scale_2
        property float rot_0
        property float rot_1
        property float rot_2
        property float rot_3
        property float opacity
        property uchar red
        property uchar green
        property uchar blue
        end_header

        """
        var data = Data()
        data.reserveCapacity(header.utf8.count + points.count * 47)
        data.append(contentsOf: header.utf8)

        let logScale = log(max(quadSizeMeters, 0.0001))
        let opacity = logit(0.92)
        for point in points {
            data.appendFloat32LE(point.position.x)
            data.appendFloat32LE(point.position.y)
            data.appendFloat32LE(point.position.z)
            data.appendFloat32LE(logScale)
            data.appendFloat32LE(logScale)
            data.appendFloat32LE(logScale)
            data.appendFloat32LE(1)
            data.appendFloat32LE(0)
            data.appendFloat32LE(0)
            data.appendFloat32LE(0)
            data.appendFloat32LE(opacity)
            data.append(point.color.r)
            data.append(point.color.g)
            data.append(point.color.b)
        }
        try data.write(to: url, options: [.atomic])
    }

    private static func writeOBJ(
        points: [PlanarRoomPoint],
        quadSizeMeters: Float,
        to url: URL
    ) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        defer {
            try? handle.close()
        }

        var writer = BufferedOBJWriter(fileHandle: handle)
        writer.write("# Furnit Planar Room flat room mesh\n")
        writer.write("# Vertex colors are written as OBJ extension: v x y z r g b\n")

        let size = max(quadSizeMeters, 0.0001)
        for point in points {
            let x = point.position.x
            let y = point.position.y
            let z = point.position.z
            let color = point.color.objColor

            writeOBJVertex(x - size, y - size, z, color: color, writer: &writer)
            writeOBJVertex(x + size, y - size, z, color: color, writer: &writer)
            writeOBJVertex(x + size, y + size, z, color: color, writer: &writer)
            writeOBJVertex(x - size, y + size, z, color: color, writer: &writer)
        }

        for index in points.indices {
            let base = index * 4 + 1
            writer.write("f \(base) \(base + 1) \(base + 2)\n")
            writer.write("f \(base) \(base + 2) \(base + 3)\n")
        }
        writer.flush()
    }

    private static func writeOBJVertex(
        _ x: Float,
        _ y: Float,
        _ z: Float,
        color: SIMD3<Float>,
        writer: inout BufferedOBJWriter
    ) {
        // Apply -90 degrees around X for Y-up OBJ viewers: new_y = z, new_z = -y.
        writer.write(
            String(
                format: "v %.6f %.6f %.6f %.6f %.6f %.6f\n",
                x,
                z,
                -y,
                color.x,
                color.y,
                color.z
            )
        )
    }

    private static func logit(_ value: Float) -> Float {
        let clamped = min(max(value, 1e-4), 1 - 1e-4)
        return log(clamped / (1 - clamped))
    }
}

private struct PlanarRoomPoint {
    let position: SIMD3<Float>
    let color: PlanarRoomRGB
}

private struct PlanarRoomVoxelKey: Hashable {
    let x: Int
    let y: Int
    let z: Int
}

private struct PlanarRoomRGB {
    let r: UInt8
    let g: UInt8
    let b: UInt8

    var objColor: SIMD3<Float> {
        SIMD3(Float(r) / 255.0, Float(g) / 255.0, Float(b) / 255.0)
    }
}

private struct RasterImage {
    let width: Int
    let height: Int
    let pixels: [UInt8]

    init(image: UIImage) throws {
        guard let cgImage = image.cgImage else {
            throw PlanarRoomError.invalidImage
        }
        width = cgImage.width
        height = cgImage.height
        guard width > 0, height > 0 else {
            throw PlanarRoomError.invalidImage
        }

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
            throw PlanarRoomError.imageRasterizationFailed
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        pixels = rgba
    }

    func average2x2Color(x: Int, y: Int) -> PlanarRoomRGB {
        let x1 = min(x + 1, width - 1)
        let y1 = min(y + 1, height - 1)
        let i00 = (y * width + x) * 4
        let i10 = (y * width + x1) * 4
        let i01 = (y1 * width + x) * 4
        let i11 = (y1 * width + x1) * 4
        let r = (Int(pixels[i00]) + Int(pixels[i10]) + Int(pixels[i01]) + Int(pixels[i11]) + 2) / 4
        let g = (Int(pixels[i00 + 1]) + Int(pixels[i10 + 1]) + Int(pixels[i01 + 1]) + Int(pixels[i11 + 1]) + 2) / 4
        let b = (Int(pixels[i00 + 2]) + Int(pixels[i10 + 2]) + Int(pixels[i01 + 2]) + Int(pixels[i11 + 2]) + 2) / 4
        return PlanarRoomRGB(r: UInt8(r), g: UInt8(g), b: UInt8(b))
    }
}

private struct BufferedOBJWriter {
    private let fileHandle: FileHandle
    private var buffer = Data()
    private let flushThreshold = 1_048_576

    init(fileHandle: FileHandle) {
        self.fileHandle = fileHandle
        buffer.reserveCapacity(flushThreshold)
    }

    mutating func write(_ string: String) {
        guard let data = string.data(using: .utf8) else { return }
        buffer.append(data)
        if buffer.count >= flushThreshold {
            flush()
        }
    }

    mutating func flush() {
        guard !buffer.isEmpty else { return }
        fileHandle.write(buffer)
        buffer.removeAll(keepingCapacity: true)
    }
}

enum PlanarRoomError: LocalizedError, Equatable {
    case invalidImage
    case imageRasterizationFailed
    case noPoints

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "The selected image could not be read."
        case .imageRasterizationFailed:
            return "The selected image could not be converted to pixels."
        case .noPoints:
            return "The selected image did not produce any room points."
        }
    }
}

private extension Data {
    mutating func appendFloat32LE(_ value: Float) {
        var bits = value.bitPattern.littleEndian
        Swift.withUnsafeBytes(of: &bits) { append(contentsOf: $0) }
    }
}
