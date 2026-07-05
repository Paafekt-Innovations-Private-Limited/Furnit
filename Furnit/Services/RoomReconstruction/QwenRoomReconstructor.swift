import Foundation
import SceneKit
import UIKit

struct QwenRoomResult: Sendable {
    let usdzURL: URL
    let roomWidthMeters: Float
    let roomHeightMeters: Float
    let roomDepthMeters: Float
    let objectCount: Int

    var summary: String {
        "Qwen room: \(String(format: "%.2f", roomWidthMeters))m x \(String(format: "%.2f", roomHeightMeters))m x \(String(format: "%.2f", roomDepthMeters))m, objects=\(objectCount)"
    }
}

final class QwenRoomReconstructor {
    private let baseURL: URL
    private let modelName: String
    private let session: URLSession
    private let outputDirectory: URL?

    init(
        baseURL: URL,
        modelName: String = "qwen2.5vl:7b",
        session: URLSession = .shared,
        outputDirectory: URL? = nil
    ) {
        self.baseURL = baseURL
        self.modelName = modelName
        self.session = session
        self.outputDirectory = outputDirectory
    }

    func reconstructWithResult(image: UIImage) async throws -> QwenRoomResult {
        let fixedImage = image.fixedOrientation()
        let payload = try await queryQwen(image: fixedImage)
        let scene = makeScene(from: payload)
        let url = try exportUSDZ(scene: scene)
        return QwenRoomResult(
            usdzURL: url,
            roomWidthMeters: payload.roomStructure.width,
            roomHeightMeters: payload.roomStructure.height,
            roomDepthMeters: payload.roomStructure.depth,
            objectCount: payload.assets.count
        )
    }

    private func queryQwen(image: UIImage) async throws -> QwenRoomPayload {
        let resized = Self.resizedImage(image, maxEdge: 1024)
        guard let jpeg = resized.jpegData(compressionQuality: 0.85) else {
            throw QwenRoomError.invalidImage
        }
        let imageWidth = Int(resized.size.width.rounded())
        let imageHeight = Int(resized.size.height.rounded())
        print("📡 [QwenRoom] sending \(imageWidth)×\(imageHeight) image (\(jpeg.count / 1024)KB)")

        let requestPayload = OllamaGenerateRequest(
            model: modelName,
            prompt: Self.prompt(imageWidth: imageWidth, imageHeight: imageHeight),
            images: [jpeg.base64EncodedString()],
            stream: true,
            options: OllamaGenerateOptions(numCtx: 8192)
        )

        var request = URLRequest(url: baseURL.appendingPathComponent("api/generate"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 600
        request.httpBody = try JSONEncoder().encode(requestPayload)

        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw QwenRoomError.requestFailed("HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)")
        }

        var fullResponse = ""
        var tokenCount = 0
        for try await line in bytes.lines {
            guard let chunk = try? JSONDecoder().decode(OllamaStreamChunk.self, from: Data(line.utf8)) else { continue }
            fullResponse += chunk.response
            tokenCount += 1
            if tokenCount % 50 == 0 {
                print("📡 [QwenRoom] \(tokenCount) tokens received...")
            }
            if chunk.done { break }
        }
        print("📡 [QwenRoom] generation complete, \(tokenCount) tokens")

        let jsonText = Self.extractJSON(from: fullResponse)
        guard let jsonData = jsonText.data(using: .utf8) else {
            print("❌ [QwenRoom] could not extract JSON from: \(fullResponse.prefix(500))")
            throw QwenRoomError.invalidResponse
        }
        return try JSONDecoder().decode(QwenRoomPayload.self, from: jsonData)
    }

    private static func resizedImage(_ image: UIImage, maxEdge: CGFloat = 1024) -> UIImage {
        let size = image.size
        let scale = Swift.min(maxEdge / size.width, maxEdge / size.height)
        guard scale < 1 else { return image }
        let newSize = CGSize(width: (size.width * scale).rounded(), height: (size.height * scale).rounded())
        let renderer = UIGraphicsImageRenderer(size: newSize)
        return renderer.image { _ in image.draw(in: CGRect(origin: .zero, size: newSize)) }
    }

    private func makeScene(from payload: QwenRoomPayload) -> SCNScene {
        let scene = SCNScene()
        let room = payload.roomStructure.sanitized
        addRoomShell(to: scene, room: room)
        for asset in payload.assets {
            addAsset(asset.sanitized, to: scene)
        }
        return scene
    }

    private func addRoomShell(to scene: SCNScene, room: QwenRoomStructure) {
        let wallThickness: Float = 0.025
        let width = room.width
        let height = room.height
        let depth = room.depth

        addBox(
            name: "floor",
            size: SCNVector3(width, wallThickness, depth),
            position: SCNVector3(0, -wallThickness * 0.5, 0),
            color: UIColor(white: 0.72, alpha: 1),
            to: scene
        )
        addBox(
            name: "ceiling",
            size: SCNVector3(width, wallThickness, depth),
            position: SCNVector3(0, height + wallThickness * 0.5, 0),
            color: UIColor(white: 0.88, alpha: 1),
            to: scene
        )
        addBox(
            name: "wall_back",
            size: SCNVector3(width, height, wallThickness),
            position: SCNVector3(0, height * 0.5, -depth * 0.5),
            color: UIColor(white: 0.82, alpha: 1),
            to: scene
        )
        addBox(
            name: "wall_left",
            size: SCNVector3(wallThickness, height, depth),
            position: SCNVector3(-width * 0.5, height * 0.5, 0),
            color: UIColor(white: 0.78, alpha: 1),
            to: scene
        )
        addBox(
            name: "wall_right",
            size: SCNVector3(wallThickness, height, depth),
            position: SCNVector3(width * 0.5, height * 0.5, 0),
            color: UIColor(white: 0.78, alpha: 1),
            to: scene
        )
    }

    private func addAsset(_ asset: QwenRoomAsset, to scene: SCNScene) {
        let color = asset.color.map(UIColor.init(qwenRGB:)) ?? Self.color(for: asset.label)
        addBox(
            name: asset.id,
            size: SCNVector3(asset.scale[0], asset.scale[1], asset.scale[2]),
            position: SCNVector3(asset.position[0], asset.position[1], asset.position[2]),
            rotationDegrees: asset.rotation,
            color: color,
            to: scene
        )
    }

    private func addBox(
        name: String,
        size: SCNVector3,
        position: SCNVector3,
        rotationDegrees: [Float] = [0, 0, 0],
        color: UIColor,
        to scene: SCNScene
    ) {
        let box = SCNBox(width: CGFloat(size.x), height: CGFloat(size.y), length: CGFloat(size.z), chamferRadius: 0)
        let material = SCNMaterial()
        material.diffuse.contents = color
        material.lightingModel = .constant
        material.isDoubleSided = true
        box.materials = [material]

        let node = SCNNode(geometry: box)
        node.name = name
        node.position = position
        node.eulerAngles = SCNVector3(
            rotationDegrees[0] * .pi / 180,
            rotationDegrees[1] * .pi / 180,
            rotationDegrees[2] * .pi / 180
        )
        scene.rootNode.addChildNode(node)
    }

    private func exportUSDZ(scene: SCNScene) throws -> URL {
        let directory = try resolvedOutputDirectory()
        let url = directory.appendingPathComponent("QwenRoom_\(Self.outputStamp()).usdz")
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        let didWrite = scene.write(to: url, options: nil, delegate: nil, progressHandler: nil)
        guard didWrite, FileManager.default.fileExists(atPath: url.path) else {
            throw QwenRoomError.exportFailed
        }
        return url
    }

    private func resolvedOutputDirectory() throws -> URL {
        let directory = outputDirectory ?? FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("QwenRooms", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static func extractJSON(from text: String) -> String {
        if let range = text.range(of: "```json") {
            let afterFence = text[range.upperBound...]
            if let end = afterFence.range(of: "```") {
                return String(afterFence[..<end.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        if let first = text.firstIndex(of: "{"), let last = text.lastIndex(of: "}"), first <= last {
            return String(text[first...last])
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func outputStamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss_SSS"
        return formatter.string(from: Date())
    }

    private static func color(for label: String) -> UIColor {
        let lower = label.lowercased()
        if lower.contains("chair") { return UIColor(red: 0.18, green: 0.19, blue: 0.23, alpha: 1) }
        if lower.contains("curtain") || lower.contains("covering") { return UIColor(red: 0.55, green: 0.35, blue: 0.18, alpha: 1) }
        if lower.contains("desk") || lower.contains("table") { return UIColor(red: 0.40, green: 0.26, blue: 0.13, alpha: 1) }
        if lower.contains("screen") || lower.contains("monitor") { return UIColor(white: 0.08, alpha: 1) }
        return UIColor(white: 0.65, alpha: 1)
    }

    private static func prompt(imageWidth: Int, imageHeight: Int) -> String {
        """
        Analyze the room image. Return raw JSON only, no markdown.
        Units are meters/degrees. Coordinate: +X right, +Y up, +Z toward viewer, floor center origin.
        Image size \(imageWidth)x\(imageHeight).

        Schema:
        {
          "room_structure": {"width": float, "height": float, "depth": float},
          "assets": [
            {"id": string, "label": string, "position": [x,y,z], "scale": [w,h,d], "rotation": [rx,ry,rz], "color": [r,g,b]}
          ]
        }
        Include only major furniture. Use realistic proportions.
        """
    }
}

private struct OllamaGenerateRequest: Encodable {
    let model: String
    let prompt: String
    let images: [String]
    let stream: Bool
    let options: OllamaGenerateOptions
}

private struct OllamaGenerateOptions: Encodable {
    let numCtx: Int

    enum CodingKeys: String, CodingKey {
        case numCtx = "num_ctx"
    }
}

private struct OllamaGenerateResponse: Decodable {
    let response: String
}

private struct OllamaStreamChunk: Decodable {
    let response: String
    let done: Bool
}

private struct QwenRoomPayload: Decodable {
    let roomStructure: QwenRoomStructure
    let assets: [QwenRoomAsset]

    enum CodingKeys: String, CodingKey {
        case roomStructure = "room_structure"
        case assets
    }
}

private struct QwenRoomStructure: Decodable {
    let width: Float
    let height: Float
    let depth: Float

    var sanitized: QwenRoomStructure {
        QwenRoomStructure(
            width: width.clamped(to: 1.5...12),
            height: height.clamped(to: 2.0...5),
            depth: depth.clamped(to: 1.5...12)
        )
    }
}

private struct QwenRoomAsset: Decodable {
    let id: String
    let label: String
    let position: [Float]
    let scale: [Float]
    let rotation: [Float]
    let color: [Int]?

    var sanitized: QwenRoomAsset {
        QwenRoomAsset(
            id: id.isEmpty ? UUID().uuidString : id,
            label: label,
            position: Self.fixedVector(position, count: 3, fallback: [0, 0.5, 0]).map { $0.clamped(to: -12...12) },
            scale: Self.fixedVector(scale, count: 3, fallback: [0.4, 0.4, 0.4]).map { $0.clamped(to: 0.05...5) },
            rotation: Self.fixedVector(rotation, count: 3, fallback: [0, 0, 0]),
            color: color
        )
    }

    enum CodingKeys: String, CodingKey {
        case id
        case label
        case position
        case scale
        case rotation
        case color
    }

    init(
        id: String,
        label: String,
        position: [Float],
        scale: [Float],
        rotation: [Float],
        color: [Int]?
    ) {
        self.id = id
        self.label = label
        self.position = position
        self.scale = scale
        self.rotation = rotation
        self.color = color
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? container.decode(String.self, forKey: .id)) ?? UUID().uuidString
        label = (try? container.decode(String.self, forKey: .label)) ?? "object"
        position = (try? container.decode([Float].self, forKey: .position)) ?? [0, 0.5, 0]
        scale = (try? container.decode([Float].self, forKey: .scale)) ?? [0.4, 0.4, 0.4]
        rotation = (try? container.decode([Float].self, forKey: .rotation)) ?? [0, 0, 0]
        color = try? container.decode([Int].self, forKey: .color)
    }

    private static func fixedVector(_ values: [Float], count: Int, fallback: [Float]) -> [Float] {
        guard values.count == count else { return fallback }
        return values
    }
}

enum QwenRoomError: LocalizedError {
    case invalidImage
    case invalidResponse
    case requestFailed(String)
    case exportFailed

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Could not encode the selected image for Qwen."
        case .invalidResponse:
            return "Qwen did not return valid room JSON."
        case .requestFailed(let message):
            return "Qwen request failed: \(message)"
        case .exportFailed:
            return "Could not export Qwen room USDZ."
        }
    }
}

private extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        Swift.min(range.upperBound, Swift.max(range.lowerBound, self))
    }
}

private extension UIColor {
    convenience init(qwenRGB values: [Int]) {
        let r = CGFloat((values[safe: 0] ?? 160).clamped(to: 0...255)) / 255.0
        let g = CGFloat((values[safe: 1] ?? 160).clamped(to: 0...255)) / 255.0
        let b = CGFloat((values[safe: 2] ?? 160).clamped(to: 0...255)) / 255.0
        self.init(red: r, green: g, blue: b, alpha: 1)
    }
}

private extension Int {
    func clamped(to range: ClosedRange<Int>) -> Int {
        Swift.min(range.upperBound, Swift.max(range.lowerBound, self))
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
