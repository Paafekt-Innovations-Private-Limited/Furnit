import UIKit
import CoreImage
import Vision

// MARK: - Room Structure Analyzer
class RoomStructureAnalyzer {
    func analyzeRoom(image: UIImage, depthMap: CIImage) async -> RoomStructure {
        print("🔍 [RoomAnalyzer] Analyzing room structure")
        var structure = RoomStructure()
        
        print("   - Detecting wall lines...")
        structure.wallLines = await detectWallLines(in: image)
        
        print("   - Detecting floor region...")
        structure.floorRegion = detectFloorRegion(in: image, depthMap: depthMap)
        print("   - Detecting ceiling region...")
        structure.ceilingRegion = detectCeilingRegion(in: image, depthMap: depthMap)
        
        print("   - Calculating vanishing point...")
        structure.vanishingPoint = calculateVanishingPoint(from: structure.wallLines)
        
        print("✅ [RoomAnalyzer] Analysis complete")
        return structure
    }
    
    private func detectWallLines(in image: UIImage) async -> [RoomStructure.Line] {
        print("📐 [WallLineDetector] Detecting wall lines")
        guard let cgImage = image.cgImage else {
            print("❌ [WallLineDetector] Failed to get CGImage")
            return []
        }
        
        let request = VNDetectContoursRequest()
        request.contrastAdjustment = 1.0
        request.detectsDarkOnLight = true
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        do {
            try handler.perform([request])
            print("✅ [WallLineDetector] Vision request performed")
        } catch {
            print("❌ [WallLineDetector] Vision request failed: \(error)")
            return []
        }
        
        // results is already [VNContoursObservation]?
        guard let observations = request.results else {
            print("⚠️ [WallLineDetector] No contours found")
            return []
        }
        
        var lines: [RoomStructure.Line] = []
        print("   - Found \(observations.count) contours")
        
        for contour in observations {
            let path = contour.normalizedPath
            let bounds = path.boundingBox
            let conf = contour.confidence
            // Two axis-aligned lines from bounds (simple placeholder)
            lines.append(RoomStructure.Line(
                start: CGPoint(x: bounds.minX, y: bounds.maxY),
                end: CGPoint(x: bounds.maxX, y: bounds.maxY),
                confidence: Float(conf)
            ))
            lines.append(RoomStructure.Line(
                start: CGPoint(x: bounds.minX, y: bounds.minY),
                end: CGPoint(x: bounds.maxX, y: bounds.minY),
                confidence: Float(conf)
            ))
        }
        
        print("✅ [WallLineDetector] Extracted \(lines.count) lines from contours")
        return lines
    }
    
    private func detectFloorRegion(in image: UIImage, depthMap: CIImage) -> CGRect? {
        let region = CGRect(x: 0, y: 0.8, width: 1.0, height: 0.2)
        print("   - Floor region: \(region)")
        return region
    }
    
    private func detectCeilingRegion(in image: UIImage, depthMap: CIImage) -> CGRect? {
        let region = CGRect(x: 0, y: 0, width: 1.0, height: 0.15)
        print("   - Ceiling region: \(region)")
        return region
    }
    
    private func calculateVanishingPoint(from lines: [RoomStructure.Line]) -> CGPoint? {
        let point = CGPoint(x: 0.5, y: 0.4)
        print("   - Vanishing point: \(point)")
        return point
    }
}
