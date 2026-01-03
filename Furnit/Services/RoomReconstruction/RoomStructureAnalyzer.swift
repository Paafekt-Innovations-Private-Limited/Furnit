//import UIKit
//import CoreImage
//import Vision
//
//// MARK: - Room Structure Analyzer
//class RoomStructureAnalyzer {
//    func analyzeRoom(image: UIImage, depthMap: CIImage) async -> RoomStructure {
//        logDebug("🔍 [RoomAnalyzer] Analyzing room structure")
//        var structure = RoomStructure()
//        
//        logDebug("   - Detecting wall lines...")
//        structure.wallLines = await detectWallLines(in: image)
//        
//        logDebug("   - Detecting floor region...")
//        structure.floorRegion = detectFloorRegion(in: image, depthMap: depthMap)
//        logDebug("   - Detecting ceiling region...")
//        structure.ceilingRegion = detectCeilingRegion(in: image, depthMap: depthMap)
//        
//        logDebug("   - Calculating vanishing point...")
//        structure.vanishingPoint = calculateVanishingPoint(from: structure.wallLines)
//        
//        logDebug("✅ [RoomAnalyzer] Analysis complete")
//        return structure
//    }
//    
//    private func detectWallLines(in image: UIImage) async -> [RoomStructure.Line] {
//        logDebug("📐 [WallLineDetector] Detecting wall lines")
//        guard let cgImage = image.cgImage else {
//            logDebug("❌ [WallLineDetector] Failed to get CGImage")
//            return []
//        }
//        
//        let request = VNDetectContoursRequest()
//        request.contrastAdjustment = 1.0
//        request.detectsDarkOnLight = true
//        
//        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
//        
//        do {
//            try handler.perform([request])
//            logDebug("✅ [WallLineDetector] Vision request performed")
//        } catch {
//            logDebug("❌ [WallLineDetector] Vision request failed: \(error)")
//            return []
//        }
//        
//        // results is already [VNContoursObservation]?
//        guard let observations = request.results else {
//            logDebug("⚠️ [WallLineDetector] No contours found")
//            return []
//        }
//        
//        var lines: [RoomStructure.Line] = []
//        logDebug("   - Found \(observations.count) contours")
//        
//        for contour in observations {
//            let path = contour.normalizedPath
//            let bounds = path.boundingBox
//            let conf = contour.confidence
//            // Two axis-aligned lines from bounds (simple placeholder)
//            lines.append(RoomStructure.Line(
//                start: CGPoint(x: bounds.minX, y: bounds.maxY),
//                end: CGPoint(x: bounds.maxX, y: bounds.maxY),
//                confidence: Float(conf)
//            ))
//            lines.append(RoomStructure.Line(
//                start: CGPoint(x: bounds.minX, y: bounds.minY),
//                end: CGPoint(x: bounds.maxX, y: bounds.minY),
//                confidence: Float(conf)
//            ))
//        }
//        
//        logDebug("✅ [WallLineDetector] Extracted \(lines.count) lines from contours")
//        return lines
//    }
//    
//    private func detectFloorRegion(in image: UIImage, depthMap: CIImage) -> CGRect? {
//        let region = CGRect(x: 0, y: 0.8, width: 1.0, height: 0.2)
//        logDebug("   - Floor region: \(region)")
//        return region
//    }
//    
//    private func detectCeilingRegion(in image: UIImage, depthMap: CIImage) -> CGRect? {
//        let region = CGRect(x: 0, y: 0, width: 1.0, height: 0.15)
//        logDebug("   - Ceiling region: \(region)")
//        return region
//    }
//    
//    private func calculateVanishingPoint(from lines: [RoomStructure.Line]) -> CGPoint? {
//        let point = CGPoint(x: 0.5, y: 0.4)
//        logDebug("   - Vanishing point: \(point)")
//        return point
//    }
//}
