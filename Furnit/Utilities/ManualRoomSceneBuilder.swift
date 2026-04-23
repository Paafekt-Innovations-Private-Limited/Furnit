import SceneKit
import UIKit

enum ManualRoomSceneBuilder {
    static func buildScene(
        roomWidth: Float,
        roomHeight: Float,
        roomDepth: Float,
        frontWallImage: UIImage,
        leftX: CGFloat = 0.12,
        rightX: CGFloat = 0.88,
        ceilingY: CGFloat = 0.15,
        floorY: CGFloat = 0.85
    ) -> SCNScene {
        let scene = SCNScene()
        let roomNode = SCNNode()
        logDebug("🏗️ [ManualRoom] buildScene start w=\(roomWidth) h=\(roomHeight) d=\(roomDepth) left=\(leftX) right=\(rightX) ceil=\(ceilingY) floor=\(floorY)")
        logDebug("🖼️ [ManualRoom] source image size=\(frontWallImage.size) scale=\(frontWallImage.scale) orientation=\(frontWallImage.imageOrientation.rawValue)")
        let normalizedImage = frontWallImage.fixedOrientation()
        logDebug("🖼️ [ManualRoom] normalized image size=\(normalizedImage.size) scale=\(normalizedImage.scale) orientation=\(normalizedImage.imageOrientation.rawValue)")
        if let cgImage = normalizedImage.cgImage {
            logDebug("🖼️ [ManualRoom] normalized cgImage=\(cgImage.width)x\(cgImage.height)")
        } else {
            logDebug("⚠️ [ManualRoom] normalized image missing cgImage")
        }

        let floorTexture = cropNormalized(normalizedImage, x: 0, y: floorY, width: 1, height: max(0.02, 1 - floorY)) ?? normalizedImage
        let ceilingTexture = cropNormalized(normalizedImage, x: 0, y: 0, width: 1, height: max(0.02, ceilingY)) ?? normalizedImage
        let leftWallTexture = cropNormalized(normalizedImage, x: 0, y: 0, width: max(0.02, leftX), height: 1) ?? normalizedImage
        let rightWallTexture = cropNormalized(normalizedImage, x: rightX, y: 0, width: max(0.02, 1 - rightX), height: 1) ?? normalizedImage
        let frontTexture = cropNormalized(normalizedImage, x: leftX, y: ceilingY, width: max(0.02, rightX - leftX), height: max(0.02, floorY - ceilingY)) ?? normalizedImage

        let floor = SCNBox(width: CGFloat(roomWidth), height: 0.01, length: CGFloat(roomDepth), chamferRadius: 0)
        floor.materials = [texturedMaterial(floorTexture)]
        let floorNode = SCNNode(geometry: floor)
        floorNode.position = SCNVector3(0, 0, 0)
        roomNode.addChildNode(floorNode)

        let ceiling = SCNBox(width: CGFloat(roomWidth), height: 0.01, length: CGFloat(roomDepth), chamferRadius: 0)
        ceiling.materials = [texturedMaterial(ceilingTexture)]
        let ceilingNode = SCNNode(geometry: ceiling)
        ceilingNode.position = SCNVector3(0, roomHeight, 0)
        roomNode.addChildNode(ceilingNode)

        let frontWall = SCNBox(width: CGFloat(roomWidth), height: CGFloat(roomHeight), length: 0.01, chamferRadius: 0)
        frontWall.materials = [texturedMaterial(frontTexture)]
        let frontNode = SCNNode(geometry: frontWall)
        frontNode.position = SCNVector3(0, roomHeight / 2, -roomDepth / 2)
        roomNode.addChildNode(frontNode)

        let leftWall = SCNBox(width: 0.01, height: CGFloat(roomHeight), length: CGFloat(roomDepth), chamferRadius: 0)
        leftWall.materials = [texturedMaterial(leftWallTexture)]
        let leftNode = SCNNode(geometry: leftWall)
        leftNode.position = SCNVector3(-roomWidth / 2, roomHeight / 2, 0)
        roomNode.addChildNode(leftNode)

        let rightWall = SCNBox(width: 0.01, height: CGFloat(roomHeight), length: CGFloat(roomDepth), chamferRadius: 0)
        rightWall.materials = [texturedMaterial(rightWallTexture)]
        let rightNode = SCNNode(geometry: rightWall)
        rightNode.position = SCNVector3(roomWidth / 2, roomHeight / 2, 0)
        roomNode.addChildNode(rightNode)

        scene.rootNode.addChildNode(roomNode)
        let (minBounds, maxBounds) = roomNode.boundingBox
        logDebug("📦 [ManualRoom] roomNode bounds min=\(minBounds) max=\(maxBounds)")
        return scene
    }

    private static func texturedMaterial(_ texture: UIImage) -> SCNMaterial {
        let material = SCNMaterial()
        material.diffuse.contents = texture
        material.emission.contents = texture
        material.emission.intensity = 0.35
        material.multiply.contents = UIColor(white: 1.08, alpha: 1.0)
        material.lightingModel = .physicallyBased
        material.isDoubleSided = true
        material.writesToDepthBuffer = true
        return material
    }

    private static func cropNormalized(
        _ image: UIImage,
        x: CGFloat,
        y: CGFloat,
        width: CGFloat,
        height: CGFloat
    ) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }

        let clampedX = min(max(x, 0), 0.98)
        let clampedY = min(max(y, 0), 0.98)
        let clampedWidth = min(max(width, 0.02), 1 - clampedX)
        let clampedHeight = min(max(height, 0.02), 1 - clampedY)

        let pixelRect = CGRect(
            x: CGFloat(cgImage.width) * clampedX,
            y: CGFloat(cgImage.height) * clampedY,
            width: CGFloat(cgImage.width) * clampedWidth,
            height: CGFloat(cgImage.height) * clampedHeight
        ).integral
        logDebug("✂️ [ManualRoom] crop x=\(clampedX) y=\(clampedY) w=\(clampedWidth) h=\(clampedHeight) pixelRect=\(pixelRect)")

        guard let cropped = cgImage.cropping(to: pixelRect) else { return nil }
        return UIImage(cgImage: cropped, scale: image.scale, orientation: image.imageOrientation)
    }
}
