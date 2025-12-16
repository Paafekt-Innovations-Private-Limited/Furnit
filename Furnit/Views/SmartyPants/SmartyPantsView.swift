// SmartyPantsView.swift
// Two-Stage Detection: Full frame -> Crop to primary bbox -> Re-detect -> UNION BOTH
// With timing logs at crucial stages + real progress bar until first detection

import SwiftUI
import UIKit
import CoreML
import Accelerate
import AVFoundation
import Photos
import Accelerate.vImage
import Foundation
//import CoreML
//import Accelerate
import CoreVideo


// MARK: - SwiftUI Wrapper
struct SmartyPantsViewSwiftUI: UIViewRepresentable {
    let mlModel: MLModel?
    var processInterval: TimeInterval = 0.1
    var confidenceThreshold: Float = 0.5
    
    var detectAllObjects: Bool = true
    var useBilinearUpscaling: Bool = true
    var maskThreshold: Float = 0.0
    var debugMode: Bool = true
    var active: Bool = false
    var edgeFillMode: EdgeFillMode = .chairType

    func makeUIView(context: Context) -> SmartyPantsContainerView {
        let v = SmartyPantsContainerView()
        v.processInterval = processInterval
        v.confidenceThreshold = confidenceThreshold
        v.detectAllObjects = detectAllObjects
        v.useBilinearUpscaling = useBilinearUpscaling
        v.maskThreshold = maskThreshold
        v.debugMode = debugMode
        v.edgeFillMode = edgeFillMode
        v.setModel(mlModel)
        if active { v.startIfNeeded() }
        return v
    }

    func updateUIView(_ uiView: SmartyPantsContainerView, context: Context) {
        uiView.setModel(mlModel)
        uiView.processInterval = processInterval
        uiView.confidenceThreshold = confidenceThreshold
        uiView.detectAllObjects = detectAllObjects
        uiView.useBilinearUpscaling = useBilinearUpscaling
        uiView.maskThreshold = maskThreshold
        uiView.debugMode = debugMode
        uiView.edgeFillMode = edgeFillMode
        if active { uiView.startIfNeeded() } else { uiView.stop() }
    }

    static func dismantleUIView(_ uiView: SmartyPantsContainerView, coordinator: ()) {
        uiView.stop()
    }
}
// MARK: - Edge Fill Mode (user selectable)
enum EdgeFillMode {
    case furniMaterial  // Preserve fine edges using scanline - for chairs, tables, solid items
    case clothBased     // Solid fill using hull - for beds, sofas, fabric items with gaps
    case chairType      // Morphological close - fills small gaps, preserves general shape
}

// MARK: - Alpha Shape (Concave Hull) using Delaunay Triangulation
struct Point2D: Hashable {
    let x: Float
    let y: Float

    static func == (lhs: Point2D, rhs: Point2D) -> Bool {
        return abs(lhs.x - rhs.x) < 1e-6 && abs(lhs.y - rhs.y) < 1e-6
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(Int(x * 1000))
        hasher.combine(Int(y * 1000))
    }
}

struct Edge2D: Hashable {
    let p1: Point2D
    let p2: Point2D

    // Normalized edge (order-independent for hashing)
    var normalized: Edge2D {
        if p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y) {
            return Edge2D(p1: p1, p2: p2)
        } else {
            return Edge2D(p1: p2, p2: p1)
        }
    }

    static func == (lhs: Edge2D, rhs: Edge2D) -> Bool {
        let l = lhs.normalized
        let r = rhs.normalized
        return l.p1 == r.p1 && l.p2 == r.p2
    }

    func hash(into hasher: inout Hasher) {
        let n = normalized
        hasher.combine(n.p1)
        hasher.combine(n.p2)
    }
}

struct Triangle2D {
    let p1: Point2D
    let p2: Point2D
    let p3: Point2D

    var edges: [Edge2D] {
        return [
            Edge2D(p1: p1, p2: p2),
            Edge2D(p1: p2, p2: p3),
            Edge2D(p1: p3, p2: p1)
        ]
    }

    // Circumcenter and circumradius
    func circumcircle() -> (center: Point2D, radius: Float) {
        let ax = p1.x, ay = p1.y
        let bx = p2.x, by = p2.y
        let cx = p3.x, cy = p3.y

        let d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        guard abs(d) > 1e-10 else {
            // Degenerate triangle
            return (Point2D(x: ax, y: ay), Float.infinity)
        }

        let ax2 = ax * ax + ay * ay
        let bx2 = bx * bx + by * by
        let cx2 = cx * cx + cy * cy

        let ux = (ax2 * (by - cy) + bx2 * (cy - ay) + cx2 * (ay - by)) / d
        let uy = (ax2 * (cx - bx) + bx2 * (ax - cx) + cx2 * (bx - ax)) / d

        let center = Point2D(x: ux, y: uy)
        let radius = sqrtf((ax - ux) * (ax - ux) + (ay - uy) * (ay - uy))

        return (center, radius)
    }

    // Check if point is inside circumcircle
    func circumcircleContains(_ p: Point2D) -> Bool {
        let (center, radius) = circumcircle()
        let dx = p.x - center.x
        let dy = p.y - center.y
        return sqrtf(dx * dx + dy * dy) <= radius
    }
}

final class AlphaShape {

    /// Maximum points for Delaunay (O(n²) complexity)
    private static let maxPointsForDelaunay = 300

    /// Compute alpha shape from points using Delaunay triangulation
    /// - Parameters:
    ///   - points: Array of (x, y) tuples
    ///   - alpha: Alpha parameter (higher = more concave, 0 = convex hull)
    /// - Returns: Array of boundary edges forming the concave hull
    static func compute(points: [(x: Int, y: Int)], alpha: Float = 0.5) -> [(start: (x: Int, y: Int), end: (x: Int, y: Int))] {
        guard points.count >= 3 else {
            return []
        }

        // Downsample points if too many (Delaunay is O(n²))
        let sampledPoints: [(x: Int, y: Int)]
        if points.count > maxPointsForDelaunay {
            sampledPoints = downsamplePoints(points, targetCount: maxPointsForDelaunay)
        } else {
            sampledPoints = points
        }

        // Convert to Point2D
        let pts = sampledPoints.map { Point2D(x: Float($0.x), y: Float($0.y)) }

        // Remove duplicates
        let uniquePts = Array(Set(pts))
        guard uniquePts.count >= 3 else {
            return []
        }

        // Step 1: Delaunay triangulation using Bowyer-Watson
        let triangles = delaunayTriangulation(points: uniquePts)

        guard !triangles.isEmpty else {
            return []
        }

        // Step 2: Calculate circumradius for each triangle and filter by alpha
        let alphaThreshold = 1.0 / max(alpha, 1e-6)
        var keptTriangles: [Triangle2D] = []

        for tri in triangles {
            let (_, radius) = tri.circumcircle()
            if radius < alphaThreshold {
                keptTriangles.append(tri)
            }
        }

        guard !keptTriangles.isEmpty else {
            // Fall back to convex hull if no triangles kept
            return convexHullEdges(points: points)
        }

        // Step 3: Find boundary edges (edges that appear in exactly one triangle)
        var edgeCount: [Edge2D: Int] = [:]
        for tri in keptTriangles {
            for edge in tri.edges {
                let normalized = edge.normalized
                edgeCount[normalized, default: 0] += 1
            }
        }

        // Boundary edges appear exactly once
        let boundaryEdges = edgeCount.filter { $0.value == 1 }.map { $0.key }

        guard !boundaryEdges.isEmpty else {
            return []
        }

        // Step 4: Order edges to form polygon(s)
        let orderedEdges = orderEdges(boundaryEdges)

        // Convert back to integer tuples
        return orderedEdges.map { edge in
            (start: (x: Int(edge.p1.x), y: Int(edge.p1.y)),
             end: (x: Int(edge.p2.x), y: Int(edge.p2.y)))
        }
    }

    /// Bowyer-Watson algorithm for Delaunay triangulation
    private static func delaunayTriangulation(points: [Point2D]) -> [Triangle2D] {
        guard points.count >= 3 else { return [] }

        // Find bounding box
        var minX = Float.infinity, maxX = -Float.infinity
        var minY = Float.infinity, maxY = -Float.infinity
        for p in points {
            minX = min(minX, p.x)
            maxX = max(maxX, p.x)
            minY = min(minY, p.y)
            maxY = max(maxY, p.y)
        }

        let dx = maxX - minX
        let dy = maxY - minY
        let deltaMax = max(dx, dy) * 2

        // Create super-triangle that contains all points
        let superP1 = Point2D(x: minX - deltaMax, y: minY - deltaMax)
        let superP2 = Point2D(x: minX + deltaMax * 2, y: minY - deltaMax)
        let superP3 = Point2D(x: minX + dx / 2, y: maxY + deltaMax)
        let superTriangle = Triangle2D(p1: superP1, p2: superP2, p3: superP3)

        var triangles: [Triangle2D] = [superTriangle]

        // Add points one at a time
        for point in points {
            var badTriangles: [Triangle2D] = []

            // Find all triangles whose circumcircle contains the point
            for tri in triangles {
                if tri.circumcircleContains(point) {
                    badTriangles.append(tri)
                }
            }

            // Find boundary of polygonal hole
            var polygon: [Edge2D] = []
            for tri in badTriangles {
                for edge in tri.edges {
                    // Edge is on boundary if it's not shared by another bad triangle
                    let isShared = badTriangles.contains { other in
                        if other.p1 == tri.p1 && other.p2 == tri.p2 && other.p3 == tri.p3 {
                            return false // Same triangle
                        }
                        return other.edges.contains { $0 == edge }
                    }
                    if !isShared {
                        polygon.append(edge)
                    }
                }
            }

            // Remove bad triangles
            triangles.removeAll { tri in
                badTriangles.contains { bad in
                    bad.p1 == tri.p1 && bad.p2 == tri.p2 && bad.p3 == tri.p3
                }
            }

            // Re-triangulate the hole
            for edge in polygon {
                let newTri = Triangle2D(p1: edge.p1, p2: edge.p2, p3: point)
                triangles.append(newTri)
            }
        }

        // Remove triangles that share vertices with super-triangle
        triangles.removeAll { tri in
            let superVertices = [superP1, superP2, superP3]
            return superVertices.contains { sv in
                tri.p1 == sv || tri.p2 == sv || tri.p3 == sv
            }
        }

        return triangles
    }

    /// Order edges to form a continuous polygon
    private static func orderEdges(_ edges: [Edge2D]) -> [Edge2D] {
        guard !edges.isEmpty else { return [] }

        var remaining = edges
        var ordered: [Edge2D] = []

        // Start with first edge
        var current = remaining.removeFirst()
        ordered.append(current)
        var currentEnd = current.p2

        while !remaining.isEmpty {
            // Find edge that connects to current end
            var found = false
            for (i, edge) in remaining.enumerated() {
                if edge.p1 == currentEnd {
                    ordered.append(edge)
                    currentEnd = edge.p2
                    remaining.remove(at: i)
                    found = true
                    break
                } else if edge.p2 == currentEnd {
                    // Reverse the edge
                    let reversed = Edge2D(p1: edge.p2, p2: edge.p1)
                    ordered.append(reversed)
                    currentEnd = reversed.p2
                    remaining.remove(at: i)
                    found = true
                    break
                }
            }

            if !found {
                // Disconnected polygon - start new chain
                if !remaining.isEmpty {
                    current = remaining.removeFirst()
                    ordered.append(current)
                    currentEnd = current.p2
                }
            }
        }

        return ordered
    }

    /// Compute convex hull edges as fallback
    private static func convexHullEdges(points: [(x: Int, y: Int)]) -> [(start: (x: Int, y: Int), end: (x: Int, y: Int))] {
        guard points.count >= 3 else { return [] }

        // Graham scan for convex hull
        let sorted = points.sorted { a, b in
            if a.y != b.y { return a.y < b.y }
            return a.x < b.x
        }

        let start = sorted[0]
        let rest = sorted.dropFirst().sorted { a, b in
            let angle1 = atan2(Float(a.y - start.y), Float(a.x - start.x))
            let angle2 = atan2(Float(b.y - start.y), Float(b.x - start.x))
            return angle1 < angle2
        }

        var hull = [start]
        for p in rest {
            while hull.count > 1 && cross(hull[hull.count - 2], hull[hull.count - 1], p) <= 0 {
                hull.removeLast()
            }
            hull.append(p)
        }

        // Create edges from hull
        var edges: [(start: (x: Int, y: Int), end: (x: Int, y: Int))] = []
        for i in 0..<hull.count {
            let next = (i + 1) % hull.count
            edges.append((start: hull[i], end: hull[next]))
        }

        return edges
    }

    private static func cross(_ o: (x: Int, y: Int), _ a: (x: Int, y: Int), _ b: (x: Int, y: Int)) -> Int {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }

    /// Grid-based downsampling to reduce point count
    /// Keeps one representative point per grid cell
    private static func downsamplePoints(_ points: [(x: Int, y: Int)], targetCount: Int) -> [(x: Int, y: Int)] {
        guard points.count > targetCount else { return points }

        // Find bounding box
        var minX = Int.max, maxX = Int.min
        var minY = Int.max, maxY = Int.min
        for p in points {
            minX = min(minX, p.x)
            maxX = max(maxX, p.x)
            minY = min(minY, p.y)
            maxY = max(maxY, p.y)
        }

        let width = maxX - minX + 1
        let height = maxY - minY + 1

        // Calculate grid cell size to achieve target count
        // gridCells ≈ targetCount, so cellSize = sqrt(area / targetCount)
        let area = Float(width * height)
        let cellSize = max(1, Int(sqrtf(area / Float(targetCount))))

        // Grid-based sampling: keep one point per cell
        var grid: [Int: (x: Int, y: Int)] = [:]  // cellKey -> point

        for p in points {
            let cellX = (p.x - minX) / cellSize
            let cellY = (p.y - minY) / cellSize
            let key = cellY * 10000 + cellX  // Unique cell key

            // Keep first point in each cell (or could average)
            if grid[key] == nil {
                grid[key] = p
            }
        }

        return Array(grid.values)
    }
}
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

// MARK: - Main Container View
final class SmartyPantsContainerView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate, UIGestureRecognizerDelegate {
    
    // MARK: Config
    var processInterval: TimeInterval = 0.1
    var confidenceThreshold: Float = 0.3
    var debugMode: Bool = true  // Enable debug prints and image saves
    var edgeFillMode: EdgeFillMode = .chairType
    
    // Detection mode: true = detect ALL objects, false = furniture classes only
    var detectAllObjects: Bool = true
    
//    private var protoFloatBuf: [Float] = []
//    private var protoPlanar8: [UInt8] = []
//    private var tmpPlanar8: [UInt8] = []
//    private var closedProto8: [UInt8] = []
//    private var upscaledPlanar8: [UInt8] = []
    
    // Preallocated reusable buffers
    private var detFloatBuf: UnsafeMutablePointer<Float>? = nil
    private var protoFloatBuf: [Float] = []
    private var protoPlanar8: [UInt8] = []
    private var tmpPlanar8: [UInt8] = []
    private var closedProto8: [UInt8] = []
    private var upscaledPlanar8: [UInt8] = []
    
    private var detFloatBufCapacity: Int = 0

    // vDSP temp for argmax when numClasses large
    private var tempConf: [Float] = []

    
    // MARK: Brightness gate (prevent processing when phone is lying down / frame is dark)
    private var lumaThreshold: Float = 0.08          // 0.0 .. 1.0
    private var brightStreak: Int = 0
    private var requiredBrightStreak: Int = 3         // require a few bright frames before resuming
    private var isDarkGateActive: Bool = false
    
    // Mask upscaling: true = bilinear (smooth edges), false = nearest-neighbor (faster)
    var useBilinearUpscaling: Bool = true
    
    // Mask threshold: values above this are considered "object"
    var maskThreshold: Float = 0.0
    
    // Optional fast morphological closing (3x3) to strengthen edges and close small gaps
    private var enableMaskClosing: Bool = true
    
    private let bboxFont: CTFont = CTFontCreateWithName("Helvetica-Bold" as CFString, 28, nil)
    private lazy var bboxAttributes: [NSAttributedString.Key: Any] = [
        .font: bboxFont,
        .foregroundColor: UIColor.white
    ]

    // MARK: Camera
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "com.furnit.smarty.sample", qos: .userInitiated)

    // MARK: UI
    private let previewLayer = AVCaptureVideoPreviewLayer()
    private let maskImageView: UIImageView = {
        let iv = UIImageView()
        iv.contentMode = .scaleAspectFit
        iv.backgroundColor = .clear
        iv.isOpaque = false
        iv.clipsToBounds = true
        iv.alpha = 1.0
        iv.isUserInteractionEnabled = false
        return iv
    }()
    
    // Real progress bar until first detection
    private let progressView: UIProgressView = {
        let pv = UIProgressView(progressViewStyle: .default)
        pv.translatesAutoresizingMaskIntoConstraints = false
        pv.tintColor = .systemGreen
        pv.trackTintColor = UIColor(white: 1.0, alpha: 0.3)
        pv.isHidden = true
        pv.progress = 0.0
        return pv
    }()
    
    private let progressLabel: UILabel = {
        let l = UILabel()
        l.translatesAutoresizingMaskIntoConstraints = false
        l.textColor = .white
        l.font = .systemFont(ofSize: 14, weight: .medium)
        l.textAlignment = .center
        l.numberOfLines = 1
        l.isHidden = true
        l.text = "Preparing…"
        l.backgroundColor = UIColor.black.withAlphaComponent(0.4)
        l.layer.cornerRadius = 10
        l.clipsToBounds = true
        return l
    }()
    
    private var hasFirstDetection = false
    
    // MARK: Gesture state
    private var currentScale: CGFloat = 1.0

    // MARK: Model & Queues
    private var mlModel: MLModel?
    private let detectionQueue = DispatchQueue(label: "com.furnit.smarty.detection", qos: .userInitiated)
    private var lastProcessTime = Date.distantPast
    private var isProcessing = false
    private let ciContext = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])

    // MARK: Object Tracker
//    private let tracker = SimpleTracker()

    // MARK: Furniture & Household Classes (LVIS indices)
    private let furnitureClasses: [Int: String] = [
        // Seating
        132: "armchair", 276: "bar stool", 352: "beach chair", 364: "bean bag chair",
        402: "bench", 821: "chair", 1060: "computer chair", 1602: "feeding chair",
        1721: "folding chair", 2499: "loveseat", 2754: "music stool", 2834: "office chair",
        2939: "park bench", 3024: "church bench", 3423: "rocking chair", 3584: "seat",
        3888: "step stool", 3909: "stool", 4041: "swivel chair", 4473: "wheelchair",
        4506: "window seat",
        
        // Beds & Bedding
        375: "bed", 376: "bedcover", 377: "bed frame", 378: "bedsheet", 379: "bed sheet",
        632: "bunk bed", 714: "canopy bed", 823: "daybed", 1137: "infant bed",
        1270: "day bed", 1364: "dog bed", 2141: "hospital bed", 2599: "mattress",
        3049: "pillow", 455: "blanket", 1047: "comforter", 1425: "duvet",
        3625: "sheet", 3626: "sheets", 431: "bedspread", 2450: "linen",
        
        // Sofas & Couches
        1141: "couch", 1816: "futon", 4331: "vanity", 2936: "ottoman", 3728: "sofa",
        
        // Tables
        429: "billiard table", 1006: "cocktail table", 1061: "computer desk", 1301: "table",
        1325: "dining table", 1503: "side table", 1885: "glass table", 2247: "island",
        2319: "kitchen counter", 2322: "kitchen island",
        2324: "kitchen table", 2802: "nightstand", 2836: "office desk", 3045: "picnic table",
        3061: "table tennis table", 3145: "poker table", 3449: "round table",
        4055: "table top", 4545: "workbench", 4564: "writing desk", 1007: "coffee table",
        
        // Storage
        332: "bathroom cabinet", 517: "bookshelf", 567: "chest", 636: "bureau",
        670: "cabinet", 977: "closet", 996: "coatrack", 1396: "drawer", 1405: "dresser",
        1624: "file cabinet", 2318: "kitchen cabinet", 2614: "medicine cabinet",
        3621: "shelf", 3678: "side cabinet", 3812: "spice rack", 4004: "supermarket shelf",
        4294: "tv cabinet", 4513: "wine cabinet", 4516: "wine rack", 4433: "wardrobe",
        
        // Lighting
        382: "bedside lamp", 1302: "table lamp", 1619: "floor lamp", 2383: "lamp",
        2384: "lampshade", 732: "candle", 898: "chandelier",
        2449: "light bulb", 2451: "light fixture", 4210: "torch", 3862: "stand",
        
        // Mirrors & Decor
        334: "bathroom mirror", 2654: "mirror", 1214: "curtain", 3485: "rug",
        3046: "picture frame", 4056: "tablecloth", 4358: "vase", 3081: "plant",
        1750: "footrest", 749: "carpet", 1402: "drape", 1403: "drapery",
        
        // Electronics
        4161: "television", 4162: "tv", 1058: "computer monitor", 1059: "computer",
        3365: "remote control", 3802: "speaker",
        
        // Bathroom
        4179: "toilet seat", 4178: "toilet", 4213: "towel bar", 4212: "towel",
        386: "bathtub", 3635: "shower", 3636: "shower curtain", 387: "bath mat",
        
        // Kitchen
        3357: "refrigerator", 2914: "oven", 2637: "microwave", 3675: "sink",
        1350: "dishwasher", 3915: "stovetop", 1780: "freezer",
        
        // Misc
        213: "baby seat", 733: "car seat", 834: "changing table", 679: "cake stand",
        1143: "counter", 1144: "counter top", 1303: "desktop", 1733: "food stand",
        1801: "fruit stand", 2193: "ice shelf", 2219: "information desk",
        1099: "cot", 1183: "cradle", 3088: "playpen"
    ]
    
    private var isAppActive: Bool = true

    private func installAppStateObservers() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appWillResignActive),
            name: UIApplication.willResignActiveNotification,
            object: nil
        )
    }
    
    func startIfNeeded() {
        hasFirstDetection = false
        isDarkGateActive = false
        brightStreak = 0
        setProgress(0.05, text: "Starting camera…")
        requestCameraPermissionAndStart()
    }

    // MARK: - Brightness (average luma) estimation
    private func averageLuma(of pixelBuffer: CVPixelBuffer, sampleStride: Int = 8) -> Float {
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

    // MARK: - Capture Delegate
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        detectionQueue.async { [weak self] in
            guard let self = self else { return }

            // 🔦 Brightness validation: if the frame is too dark, pause detection until bright again
            let luma = self.averageLuma(of: pixelBuffer)
            if luma.isFinite && luma < self.lumaThreshold {
                // Enter/maintain dark gate state
                self.isDarkGateActive = true
                self.brightStreak = 0
                self.showDarkGate(message: "Lift phone and point at the scene…")
                // Clear any previous output to make state obvious
                DispatchQueue.main.async {
                    self.maskImageView.image = nil
                }
                return
            } else {
                // Count consecutive bright frames before resuming
                self.brightStreak += 1
                if self.isDarkGateActive && self.brightStreak < self.requiredBrightStreak {
                    // Still waiting for stability
                    self.showDarkGate(message: "Hold steady…")
                    return
                }
                if self.isDarkGateActive {
                    // We have enough bright frames; exit gate
                    self.isDarkGateActive = false
                    self.hideDarkGateIfNeeded()
                }
            }

            self.processFrame(pixelBuffer)
        }
    }


    private func showDarkGate(message: String) {
        DispatchQueue.main.async {
            self.progressView.isHidden = true
            self.progressLabel.isHidden = false
            self.progressLabel.text = "  \(message)  "
            self.progressLabel.alpha = 1.0
        }
    }

    private func hideDarkGateIfNeeded() {
        guard isDarkGateActive else { return }
        DispatchQueue.main.async {
            UIView.animate(withDuration: 0.2) {
                self.progressLabel.alpha = 0
            } completion: { _ in
                self.progressLabel.isHidden = true
                self.progressLabel.alpha = 1
            }
        }
    }

    @objc private func appDidBecomeActive() {
        isAppActive = true
    }

    @objc private func appWillResignActive() {
        isAppActive = false
        // Also stop any “in-flight” processing quickly.
        detectionQueue.async { [weak self] in
            self?.isProcessing = false
        }
    }


    override init(frame: CGRect) {
        super.init(frame: frame)
        commonInit()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    private func commonInit() {
        backgroundColor = .clear
        isUserInteractionEnabled = true

        // Camera preview (hidden, used only for capture)
        previewLayer.session = captureSession
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.isHidden = true
        layer.addSublayer(previewLayer)

        maskImageView.isUserInteractionEnabled = true
        addSubview(maskImageView)
        maskImageView.translatesAutoresizingMaskIntoConstraints = true
        maskImageView.frame = bounds
        maskImageView.autoresizingMask = [.flexibleWidth, .flexibleHeight]

        addSubview(progressView)
        addSubview(progressLabel)
        NSLayoutConstraint.activate([
            progressView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 40),
            progressView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -40),
            progressView.topAnchor.constraint(equalTo: safeAreaLayoutGuide.topAnchor, constant: 12),

            progressLabel.centerXAnchor.constraint(equalTo: progressView.centerXAnchor),
            progressLabel.bottomAnchor.constraint(equalTo: progressView.topAnchor, constant: -6),
            progressLabel.heightAnchor.constraint(equalToConstant: 24),
            progressLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 80)
        ])

        // Gestures (unchanged)
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        pinchGesture.delegate = self
        addGestureRecognizer(pinchGesture)

        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        panGesture.delegate = self
        maskImageView.addGestureRecognizer(panGesture)

        // 🔔 Observe app going to background / foreground
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAppDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleAppDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification,
            object: nil
        )

        setupCamera()
        installAppStateObservers()   // ← add this
        if self.debugMode { print("✅ SmartyPantsContainerView initialized") }
        
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        if Thread.isMainThread {
            self.maskImageView.image = nil
            self.layer.removeAllAnimations()
        } else {
            DispatchQueue.main.sync {
                self.maskImageView.image = nil
                self.layer.removeAllAnimations()
            }
        }
        videoOutput.setSampleBufferDelegate(nil, queue: nil)
        stopCamera()
    }


    @objc private func handleAppDidEnterBackground() {
        if debugMode { print("📵 App entered background – stopping camera & delegate") }
        // Stop delivering frames
        videoOutput.setSampleBufferDelegate(nil, queue: nil)
        stopCamera()
    }

    @objc private func handleAppDidBecomeActive() {
        if debugMode { print("📲 App became active – restarting camera if needed") }
        // Only restart if you want live detection when active
        videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
        requestCameraPermissionAndStart()
    }

    private func setProgress(_ value: Float, text: String) {
        guard !hasFirstDetection else { return }
        DispatchQueue.main.async {
            self.progressView.isHidden = false
            self.progressLabel.isHidden = false
            self.progressView.progress = value
            self.progressLabel.text = "  \(text)  "
        }
    }
    
    private func finishFirstDetectionIfNeeded() {
        guard !hasFirstDetection else { return }
        hasFirstDetection = true
        DispatchQueue.main.async {
            UIView.animate(withDuration: 0.25, animations: {
                self.progressView.alpha = 0
                self.progressLabel.alpha = 0
            }, completion: { _ in
                self.progressView.isHidden = true
                self.progressLabel.isHidden = true
                self.progressView.alpha = 1
                self.progressLabel.alpha = 1
                self.progressView.progress = 0
            })
        }
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
    }
    
    override func point(inside point: CGPoint, with event: UIEvent?) -> Bool {
        if point.y < 100 { return false }
        return true
    }
    
    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard maskImageView.image != nil else { return }
        
        switch gesture.state {
        case .began:
            break
        case .changed:
            let newScale = currentScale * gesture.scale
            let clampedScale = min(max(newScale, 0.3), 3.0)
            maskImageView.transform = CGAffineTransform(scaleX: clampedScale, y: clampedScale)
            currentScale = clampedScale
            gesture.scale = 1.0
        case .ended, .cancelled:
            if currentScale > 0.9 && currentScale < 1.1 {
                currentScale = 1.0
                UIView.animate(withDuration: 0.2) {
                    self.maskImageView.transform = .identity
                }
            }
        default:
            break
        }
    }
    
    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        guard let _ = maskImageView.image else { return }
        
        let translation = gesture.translation(in: self)
        
        switch gesture.state {
        case .began, .changed:
            // move normally
            maskImageView.center = CGPoint(
                x: maskImageView.center.x + translation.x,
                y: maskImageView.center.y + translation.y
            )
            gesture.setTranslation(.zero, in: self)
            
//        case .ended, .cancelled:
//            // proper clamping using REAL frame (after transforms!)
//            let frame = maskImageView.frame
//
//            var newCenter = maskImageView.center
//
//            let maxLeft = frame.width / 2
//            let maxRight = bounds.width - frame.width / 2
//            let maxTop = frame.height / 2
//            let maxBottom = bounds.height - frame.height / 2
//
//            // Slight padding so user can push "off-screen" a bit
//            let pad: CGFloat = 150
//
//            newCenter.x = min(max(newCenter.x, maxLeft - pad), maxRight + pad)
//            newCenter.y = min(max(newCenter.y, maxTop - pad), maxBottom + pad)
//
//            UIView.animate(withDuration: 0.2, delay: 0, options: [.curveEaseOut]) {
//                self.maskImageView.center = newCenter
//            }
            
        default:
            break
        }
    }


    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        return true
    }

    
    // MARK: - UIGestureRecognizerDelegate
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
        let location = touch.location(in: self)
        if location.y < 100 { return false }
        return true
    }
    
    // MARK: - Public
    func setModel(_ model: MLModel?) {
        detectionQueue.sync {
            self.mlModel = model
        }
    }
    
    func stop() {
        stopCamera()
    }

    // MARK: - Camera Setup
    private func setupCamera() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1280x720
        captureSession.inputs.forEach { captureSession.removeInput($0) }
        captureSession.outputs.forEach { captureSession.removeOutput($0) }

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            captureSession.commitConfiguration()
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
            videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            if captureSession.canAddOutput(videoOutput) {
                captureSession.addOutput(videoOutput)
            }
            if let conn = videoOutput.connection(with: .video) {
                conn.videoRotationAngle = 90
            }
            captureSession.commitConfiguration()
            DispatchQueue.global(qos: .userInitiated).async {
                self.captureSession.startRunning()
            }
        } catch {
            captureSession.commitConfiguration()
        }
    }

    private func stopCamera() {
        DispatchQueue.global(qos: .userInitiated).async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
        }
    }

    private func requestCameraPermissionAndStart() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            if !captureSession.isRunning {
                DispatchQueue.global(qos: .userInitiated).async {
                    self.captureSession.startRunning()
                }
            }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    DispatchQueue.global(qos: .userInitiated).async { self.captureSession.startRunning() }
                }
            }
        default:
            break
        }
    }

    // MARK: - Capture Delegate
//    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
//        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
//        detectionQueue.async { [weak self] in self?.processFrame(pixelBuffer) }
//    }
    

    // MARK: - Crop Pixel Buffer to BBox (vImage copy)
    private func cropPixelBuffer(_ pixelBuffer: CVPixelBuffer, toBBox det: DetectionSmarty, padding: Float = 0.1) -> CVPixelBuffer? {
        let cropStart = Date()
        
        let fullWf = Float(CVPixelBufferGetWidth(pixelBuffer))
        let fullHf = Float(CVPixelBufferGetHeight(pixelBuffer))
        
        let scaleX = fullWf / 1280.0
        let scaleY = fullHf / 1280.0
        
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
        
        if self.debugMode {
            let dt = Date().timeIntervalSince(cropStart) * 1000.0
            print(String(format: "⏱ cropPixelBuffer: %.2f ms (rect %dx%d)", dt, cropW, cropH))
        }
        
        return dst
    }
    
    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        let frameStart = Date()
        guard let model = mlModel else { return }
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !isProcessing else { return }
        lastProcessTime = now
        isProcessing = true
        if debugMode { print("\n🕒 ===== NEW FRAME @ \(now.timeIntervalSince1970) =====") }

        setProgress(0.2, text: "Preprocessing frame…")
        // Preprocess -> resize + MLMultiArray
        let stage1PreStart = Date()
        guard let resized = resizePixelBufferToSquare(pixelBuffer, size: 1280),
              let inputArray = pixelBufferToMLMultiArray(resized) else { isProcessing = false; return }
        if debugMode { print(String(format: "⏱ Stage1 preprocess: %.2f ms", Date().timeIntervalSince(stage1PreStart)*1000)) }

        // Inference
        setProgress(0.35, text: "Running detection…")
        let stage1InfStart = Date()
        guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]),
              let output = try? model.prediction(from: inputProvider) else { isProcessing = false; return }
        if debugMode { print(String(format: "⏱ Stage1 model.prediction: %.2f ms", Date().timeIntervalSince(stage1InfStart)*1000)) }

        // Extract outputs (detections + prototypes)
        guard let detArray = output.featureValue(for: "var_2497")?.multiArrayValue,
              let prototypesArray = output.featureValue(for: "p")?.multiArrayValue else {
            isProcessing = false; return
        }

        // Quick decode (low-threshold for masks)
        setProgress(0.55, text: "Refining crop…")
//        let rawDetections = extractDetections(from: detArray, confThreshold: 0.01)
//        let rawDetections = extractDetections(from: detArray, confThreshold: 0.01, topK: min(50, 10000))
        
        let rawDetections = extractDetections(from: detArray, confThreshold: 0.01, topK: min(50, /*some upper bound*/ 10000))

        if rawDetections.isEmpty {
            DispatchQueue.main.async { self.maskImageView.image = nil; self.isProcessing = false }
            return
        }

        // Compute masks once (fast path)
        setProgress(0.8, text: "Building mask…")
        let maskCoeffs = rawDetections.map { $0.maskCoeffs }
        let imageW = CVPixelBufferGetWidth(pixelBuffer), imageH = CVPixelBufferGetHeight(pixelBuffer)
        let masks = processMasks(protos: prototypesArray, maskCoeffs: maskCoeffs, boxes: rawDetections, outSize: (height: imageH, width: imageW), upsample: true)

        // Fast generateCutout which accepts precomputed masks
        let cutoutStart = Date()
        generateCutout(stage1Prototypes: prototypesArray, primaryBBoxes: rawDetections, masks: masks, originalImage: pixelBuffer)
        if debugMode {
            let cutoutEnd = Date()
            print(String(format: "⏱ generateCutout: %.2f ms", cutoutEnd.timeIntervalSince(cutoutStart)*1000))
            print(String(format: "🕒 Frame total: %.2f ms", cutoutEnd.timeIntervalSince(frameStart)*1000))
        }
        isProcessing = false
    }



    private func applyNMS(_ detections: [DetectionSmarty], iouThreshold: Float) -> [DetectionSmarty] {
        // Guard against empty or invalid input
        guard !detections.isEmpty else { return [] }
        guard iouThreshold >= 0 && iouThreshold <= 1 else {
            if self.debugMode { print("⚠️ applyNMS: Invalid IoU threshold: \(iouThreshold)") }
            return detections
        }
        
        // Filter out detections with invalid dimensions before sorting
        let validDetections = detections.filter { det in
            guard det.width > 0, det.height > 0,
                  det.width.isFinite, det.height.isFinite,
                  det.x.isFinite, det.y.isFinite,
                  det.confidence >= 0 && det.confidence <= 1 else {
                if self.debugMode {
                    print("⚠️ applyNMS: Filtering invalid detection: w=\(det.width), h=\(det.height), x=\(det.x), y=\(det.y), conf=\(det.confidence)")
                }
                return false
            }
            return true
        }
        
        guard !validDetections.isEmpty else { return [] }
        
        let sorted = validDetections.sorted { $0.confidence > $1.confidence }
        var kept: [DetectionSmarty] = []
        kept.reserveCapacity(sorted.count)

        for det in sorted {
            var dominated = false
            for k in kept {
                let iou = bboxIoU(det, k)
                if iou.isFinite && iou > iouThreshold {
                    dominated = true
                    break
                }
            }
            if !dominated { kept.append(det) }
        }
        return kept
    }

    private func bboxIoU(_ a: DetectionSmarty, _ b: DetectionSmarty) -> Float {
        // Guard against invalid inputs
        guard a.width > 0 && a.height > 0 && b.width > 0 && b.height > 0 else { return 0 }
        guard a.width.isFinite && a.height.isFinite && b.width.isFinite && b.height.isFinite else { return 0 }
        guard a.x.isFinite && a.y.isFinite && b.x.isFinite && b.y.isFinite else { return 0 }
        
        let aLeft = a.x - a.width * 0.5
        let aRight = a.x + a.width * 0.5
        let aTop = a.y - a.height * 0.5
        let aBottom = a.y + a.height * 0.5

        let bLeft = b.x - b.width * 0.5
        let bRight = b.x + b.width * 0.5
        let bTop = b.y - b.height * 0.5
        let bBottom = b.y + b.height * 0.5

        let ix1 = max(aLeft, bLeft)
        let ix2 = min(aRight, bRight)
        let iy1 = max(aTop, bTop)
        let iy2 = min(aBottom, bBottom)
        
        let iw = max(0, ix2 - ix1)
        let ih = max(0, iy2 - iy1)
        let inter = iw * ih
        
        let areaA = a.width * a.height
        let areaB = b.width * b.height
        let union = areaA + areaB - inter
        
        // Prevent division by zero and ensure result is valid
        guard union > 0 && union.isFinite && inter.isFinite else { return 0 }
        
        let iou = inter / union
        return iou.isFinite ? iou : 0
    }

    
    private func resizePixelBufferToSquare(_ src: CVPixelBuffer, size: Int = 1280) -> CVPixelBuffer? {
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

        if self.debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ letterbox %dx%d → %dx%d: %.2f ms",
                         srcW, srcH, size, size, dt))
        }

        return dst
    }

    private func keepOverlappingDetections(_ detections: [DetectionSmarty]) -> [DetectionSmarty] {
        guard detections.count > 0 else { return [] }
        if detections.count == 1 { return detections }

        let sorted = detections.sorted { $0.confidence > $1.confidence }
        let primary = sorted[0]
        let pLeft = primary.x - primary.width / 2
        let pRight = primary.x + primary.width / 2
        let pTop = primary.y - primary.height / 2
        let pBottom = primary.y + primary.height / 2

        var kept: [DetectionSmarty] = []
        kept.reserveCapacity(sorted.count)

        for det in sorted {
            let aLeft = det.x - det.width / 2
            let aRight = det.x + det.width / 2
            let aTop = det.y - det.height / 2
            let aBottom = det.y + det.height / 2

            if aRight < pLeft || pRight < aLeft { continue }
            if aBottom < pTop || pBottom < aTop { continue }
            kept.append(det)
        }
        return kept
    }
    
    // MARK: - Print 20x20 Binary Grid
    private func print20x20BinaryGrid(_ title: String, mask: [UInt8], width: Int, height: Int) {
        guard self.debugMode else { return }
        
        print("\n🔢 [\(title)] (20x20 binary, * = object, . = background):")
        for gy in 0..<20 {
            var rowSymbols = ""
            for gx in 0..<20 {
                let y = gy * 8 + 7
                let x = gx * 8 + 7
                if y < height && x < width {
                    let idx = y * width + x
                    rowSymbols += mask[idx] > 0 ? "*" : "."
                } else {
                    rowSymbols += " "
                }
            }
            print("   \(rowSymbols)")
        }
    }
    
    // MARK: - Save Mask to File (with Accelerate normalization)
    private func saveMaskToFile(rawMask: [Float], width: Int, height: Int, detection: DetectionSmarty) {
        let timestamp = Int(Date().timeIntervalSince1970)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        
        var minVal: Float = 0
        var maxVal: Float = 0
        vDSP_minv(rawMask, 1, &minVal, vDSP_Length(rawMask.count))
        vDSP_maxv(rawMask, 1, &maxVal, vDSP_Length(rawMask.count))
        let range = maxVal - minVal
        
        let count = rawMask.count
        var normalized = [Float](repeating: 0, count: count)
        
        if range > 0 {
            var negMin = -minVal
            rawMask.withUnsafeBufferPointer { src in
                normalized.withUnsafeMutableBufferPointer { dst in
                    vDSP_vsadd(src.baseAddress!, 1, &negMin, dst.baseAddress!, 1, vDSP_Length(count))
                }
            }
            var invRange: Float = 1.0 / range
            vDSP_vsmul(normalized, 1, &invRange, &normalized, 1, vDSP_Length(count))
        } else {
            normalized = [Float](repeating: 0.5, count: count)
        }
        
        var scale255: Float = 255.0
        vDSP_vsmul(normalized, 1, &scale255, &normalized, 1, vDSP_Length(count))
        
        var clipLow: Float = 0
        var clipHigh: Float = 255
        vDSP_vclip(normalized, 1, &clipLow, &clipHigh, &normalized, 1, vDSP_Length(count))
        
        var grayPixels = [UInt8](repeating: 0, count: count)
        normalized.withUnsafeBufferPointer { src in
            grayPixels.withUnsafeMutableBufferPointer { dst in
                vDSP_vfixu8(src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(count))
            }
        }
        
        if let provider = CGDataProvider(data: Data(grayPixels) as CFData),
           let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
                                  bytesPerRow: width, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: 0),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) {
            let grayImage = UIImage(cgImage: cgImage)
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: grayImage)
            }) { success, error in
                if success {
                    print("💾 Saved GRAYSCALE mask to Photos @ \(timestamp)")
                } else {
                    print("❌ Failed to save grayscale: \(error?.localizedDescription ?? "unknown")")
                }
            }
        }
        
        let scale = Float(width) / 1280.0
        let mx1 = max(0, Int((detection.x - detection.width / 2) * scale))
        let my1 = max(0, Int((detection.y - detection.height / 2) * scale))
        let mx2 = min(width, Int((detection.x + detection.width / 2) * scale))
        let my2 = min(height, Int((detection.y + detection.height / 2) * scale))
        
        var binaryPixels = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                if x >= mx1 && x < mx2 && y >= my1 && y < my2 && rawMask[idx] > maskThreshold {
                    binaryPixels[idx] = 255
                }
            }
        }
        
        if let provider = CGDataProvider(data: Data(binaryPixels) as CFData),
           let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
                                  bytesPerRow: width, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: 0),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) {
            let binaryImage = UIImage(cgImage: cgImage)
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: binaryImage)
            }) { success, error in
                if success {
                    print("💾 Saved BINARY mask to Photos (threshold: \(self.maskThreshold)) @ \(timestamp)")
                } else {
                    print("❌ Failed to save binary: \(error?.localizedDescription ?? "unknown")")
                }
            }
        }
    }
    
    // MARK: - Morphological Closing (Dilate then Erode) on binary mask at prototype resolution
    // MARK: - Fill Convex Hull of Mask
    /// Computes convex hull of mask pixels and fills the entire hull area
    /// This guarantees no gaps within each detection's boundary
    private func fillConvexHullOfMask(
        mask: [Float],
        into globalMask: inout [Float],
        width: Int,
        height: Int,
        bboxX1: Int, bboxY1: Int, bboxX2: Int, bboxY2: Int,
        threshold: Float
    ) -> Int {
        // Collect mask pixels above threshold within bbox
        var points = [(x: Int, y: Int)]()
        for py in bboxY1..<bboxY2 {
            let rowStart = py * width
            for px in bboxX1..<bboxX2 {
                let idx = rowStart + px
                if mask[idx] > threshold {
                    points.append((px, py))
                }
            }
        }

        guard points.count >= 3 else {
            // Not enough points for hull, just copy mask directly
            var added = 0
            for py in bboxY1..<bboxY2 {
                let rowStart = py * width
                for px in bboxX1..<bboxX2 {
                    let idx = rowStart + px
                    if mask[idx] > threshold && globalMask[idx] == 0 {
                        globalMask[idx] = 1.0
                        added += 1
                    }
                }
            }
            return added
        }

        // Compute convex hull using Graham scan
        let hull = convexHull(points: points)
        guard hull.count >= 3 else {
            return 0
        }

        // Fill the convex hull polygon using scanline
        var added = 0
        let minY = hull.min(by: { $0.y < $1.y })!.y
        let maxY = hull.max(by: { $0.y < $1.y })!.y

        for y in minY...maxY {
            // Find all x-intersections with hull edges at this y
            var xIntersections = [Int]()

            for i in 0..<hull.count {
                let p1 = hull[i]
                let p2 = hull[(i + 1) % hull.count]

                // Check if edge crosses this scanline
                if (p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y) {
                    // Calculate x intersection
                    let t = Float(y - p1.y) / Float(p2.y - p1.y)
                    let x = Int(Float(p1.x) + t * Float(p2.x - p1.x))
                    xIntersections.append(x)
                }
            }

            xIntersections.sort()

            // Fill between pairs of intersections
            var i = 0
            while i + 1 < xIntersections.count {
                let x1 = max(bboxX1, xIntersections[i])
                let x2 = min(bboxX2 - 1, xIntersections[i + 1])
                if x2 >= x1 && y >= 0 && y < height {
                    let rowStart = y * width
                    for x in x1...x2 {
                        if x >= 0 && x < width {
                            let idx = rowStart + x
                            if globalMask[idx] == 0 {
                                globalMask[idx] = 1.0
                                added += 1
                            }
                        }
                    }
                }
                i += 2
            }
        }

        return added
    }

    /// Graham scan convex hull algorithm
    private func convexHull(points: [(x: Int, y: Int)]) -> [(x: Int, y: Int)] {
        guard points.count >= 3 else { return points }

        // Find lowest point (and leftmost if tie)
        var sorted = points
        let start = sorted.min { a, b in
            if a.y != b.y { return a.y < b.y }
            return a.x < b.x
        }!

        // Sort by polar angle with respect to start point
        sorted.sort { a, b in
            let ax = a.x - start.x, ay = a.y - start.y
            let bx = b.x - start.x, by = b.y - start.y
            let cross = ax * by - ay * bx
            if cross != 0 { return cross > 0 }
            // Collinear - sort by distance
            return ax * ax + ay * ay < bx * bx + by * by
        }

        // Build hull
        var hull = [(x: Int, y: Int)]()
        for p in sorted {
            while hull.count >= 2 {
                let o = hull[hull.count - 2]
                let a = hull[hull.count - 1]
                let cross = (a.x - o.x) * (p.y - o.y) - (a.y - o.y) * (p.x - o.x)
                if cross <= 0 {
                    hull.removeLast()
                } else {
                    break
                }
            }
            hull.append(p)
        }

        return hull
    }

    private func applyMorphologicalClosing(to mask: inout [Float], width: Int, height: Int, kernelSize: Int = 9, iterations: Int = 2) {
        let count = width * height
        guard count > 0, mask.count == count else { return }

        // Convert Float (0/1) -> Planar8 (0/255)
        var buf1 = [UInt8](repeating: 0, count: count)
        for i in 0..<count { buf1[i] = mask[i] > 0 ? 255 : 0 }

        var buf2 = [UInt8](repeating: 0, count: count)
        var tmpBuf = [UInt8](repeating: 0, count: count)

        // Create kernel (all ones)
        let kSize = kernelSize
        var kernel = [UInt8](repeating: 1, count: kSize * kSize)

        for iter in 0..<iterations {
            let isEven = (iter % 2 == 0)

            if isEven {
                buf1.withUnsafeMutableBytes { srcPtr in
                    tmpBuf.withUnsafeMutableBytes { tmpPtr in
                        buf2.withUnsafeMutableBytes { dstPtr in
                            var srcVBuf = vImage_Buffer(data: srcPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                            var tmpVBuf = vImage_Buffer(data: tmpPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                            var dstVBuf = vImage_Buffer(data: dstPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)

                            kernel.withUnsafeMutableBufferPointer { kPtr in
                                vImageDilate_Planar8(&srcVBuf, &tmpVBuf, 0, 0, kPtr.baseAddress!, vImagePixelCount(kSize), vImagePixelCount(kSize), vImage_Flags(kvImageNoFlags))
                                vImageErode_Planar8(&tmpVBuf, &dstVBuf, 0, 0, kPtr.baseAddress!, vImagePixelCount(kSize), vImagePixelCount(kSize), vImage_Flags(kvImageNoFlags))
                            }
                        }
                    }
                }
            } else {
                buf2.withUnsafeMutableBytes { srcPtr in
                    tmpBuf.withUnsafeMutableBytes { tmpPtr in
                        buf1.withUnsafeMutableBytes { dstPtr in
                            var srcVBuf = vImage_Buffer(data: srcPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                            var tmpVBuf = vImage_Buffer(data: tmpPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
                            var dstVBuf = vImage_Buffer(data: dstPtr.baseAddress!, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)

                            kernel.withUnsafeMutableBufferPointer { kPtr in
                                vImageDilate_Planar8(&srcVBuf, &tmpVBuf, 0, 0, kPtr.baseAddress!, vImagePixelCount(kSize), vImagePixelCount(kSize), vImage_Flags(kvImageNoFlags))
                                vImageErode_Planar8(&tmpVBuf, &dstVBuf, 0, 0, kPtr.baseAddress!, vImagePixelCount(kSize), vImagePixelCount(kSize), vImage_Flags(kvImageNoFlags))
                            }
                        }
                    }
                }
            }
        }

        // Result is in buf2 if iterations is even, buf1 if odd
        let resultBuf = (iterations % 2 == 0) ? buf1 : buf2

        // Convert back Planar8 -> Float (0/1)
        for i in 0..<count { mask[i] = resultBuf[i] > 0 ? 1.0 : 0.0 }
    }

    
    func clearOutsideUsingIntCorners(x0: Int, y0: Int, x1: Int, y1: Int, in image: CGImage) -> CGImage? {
        let t0 = Date()
        
        let width = image.width
        let height = image.height
        let imageRect = CGRect(x: 0, y: 0, width: width, height: height)

        let minX0 = min(x0, x1)
        let maxX0 = max(x0, x1)
        let minY0 = min(y0, y1)
        let maxY0 = max(y0, y1)

        var bbox = CGRect(x: CGFloat(minX0),
                          y: CGFloat(minY0),
                          width: CGFloat(maxX0 - minX0),
                          height: CGFloat(maxY0 - minY0))

        if bbox.isNull || bbox.width <= 0 || bbox.height <= 0 {
            let bytesPerPixel = 4
            let bytesPerRow = bytesPerPixel * width
            let dataSize = bytesPerRow * height
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: dataSize)
            buffer.initialize(repeating: 0, count: dataSize)
            defer {
                buffer.deinitialize(count: dataSize)
                buffer.deallocate()
            }
            guard let ctx = CGContext(data: buffer,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return nil }
            let out = ctx.makeImage()
            if self.debugMode {
                let dt = Date().timeIntervalSince(t0) * 1000.0
                print(String(format: "⏱ clearOutsideUsingIntCorners (empty bbox): %.2f ms", dt))
            }
            return out
        }

        bbox = bbox.intersection(imageRect)

        if bbox.isNull || bbox.width <= 0 || bbox.height <= 0 {
            let bytesPerPixel = 4
            let bytesPerRow = bytesPerPixel * width
            let dataSize = bytesPerRow * height
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: dataSize)
            buffer.initialize(repeating: 0, count: dataSize)
            defer {
                buffer.deinitialize(count: dataSize)
                buffer.deallocate()
            }
            guard let ctx = CGContext(data: buffer,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return nil }
            let out = ctx.makeImage()
            if self.debugMode {
                let dt = Date().timeIntervalSince(t0) * 1000.0
                print(String(format: "⏱ clearOutsideUsingIntCorners (clipped empty): %.2f ms", dt))
            }
            return out
        }

        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let dataSize = bytesPerRow * height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }

        let rawData = UnsafeMutablePointer<UInt8>.allocate(capacity: dataSize)
        rawData.initialize(repeating: 0, count: dataSize)
        defer {
            rawData.deinitialize(count: dataSize)
            rawData.deallocate()
        }

        guard let ctx = CGContext(data: rawData,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }

        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1.0, y: -1.0)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let startX = 0
        let endX = width
        let startY = 0
        let endY = height

        var kx0 = Int(floor(bbox.minX))
        var ky0 = Int(floor(bbox.minY))
        var kx1 = Int(ceil(bbox.maxX))
        var ky1 = Int(ceil(bbox.maxY))

        kx0 = max(startX, min(kx0, endX))
        kx1 = max(startX, min(kx1, endX))
        ky0 = max(startY, min(ky0, endY))
        ky1 = max(startY, min(ky1, endY))

        if kx0 > kx1 { swap(&kx0, &kx1) }
        if ky0 > ky1 { swap(&ky0, &ky1) }

        for y in startY..<endY {
            let rowBase = rawData + y * bytesPerRow
            for x in startX..<endX {
                let px = rowBase + x * bytesPerPixel
                let inside = (x >= kx0 && x < kx1 && y >= ky0 && y < ky1)
                if !inside {
                    px[0] = 0
                    px[1] = 0
                    px[2] = 0
                    px[3] = 0
                }
            }
        }

        let out = ctx.makeImage()
        if self.debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ clearOutsideUsingIntCorners: %.2f ms", dt))
        }
        return out
    }

    private func makePrototypeBuffer(from array: MLMultiArray, C: Int, Hp: Int, Wp: Int) -> [Float] {
        let count = C * Hp * Wp
        var out = [Float](repeating: 0, count: count)
        
        // Validate array size matches expected count
        guard array.count >= count else {
            if self.debugMode {
                print("⚠️ makePrototypeBuffer: Array size mismatch! Expected: \(count), Got: \(array.count)")
            }
            return out
        }
        
        switch array.dataType {
        case .float32:
            // Safer memory copying with bounds checking
            array.dataPointer.withMemoryRebound(to: Float.self, capacity: array.count) { src in
                out.withUnsafeMutableBufferPointer { dst in
                    guard let dstPtr = dst.baseAddress else {
                        if self.debugMode { print("⚠️ makePrototypeBuffer: Null destination pointer") }
                        return
                    }
                    let safeCopyCount = min(count, array.count)
                    memcpy(dstPtr, src, safeCopyCount * MemoryLayout<Float>.size)
                }
            }
        case .float16:
            // Safer Float16 conversion with bounds checking
            let actualCount = min(count, array.count)
            let src = array.dataPointer.bindMemory(to: UInt16.self, capacity: actualCount)
            var srcBuf = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: src),
                height: 1,
                width: vImagePixelCount(actualCount),
                rowBytes: actualCount * MemoryLayout<UInt16>.size
            )
            out.withUnsafeMutableBufferPointer { dst in
                var dstBuf = vImage_Buffer(
                    data: dst.baseAddress,
                    height: 1,
                    width: vImagePixelCount(actualCount),
                    rowBytes: actualCount * MemoryLayout<Float>.size
                )
                let result = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
                if result != kvImageNoError && self.debugMode {
                    print("⚠️ makePrototypeBuffer: vImage conversion failed with error: \(result)")
                }
            }
        default:
            // Safe fallback with bounds checking
            let safeCopyCount = min(count, array.count)
            for i in 0..<safeCopyCount {
                out[i] = array[i].floatValue
            }
        }
        
        return out
    }

    
    private func makeBinaryMaskFromGlobalMask(_ globalMask: [Float], count: Int) -> [UInt8] {
        var scaled = [Float](repeating: 0, count: count)
        var scale255: Float = 255.0
        
        globalMask.withUnsafeBufferPointer { src in
            scaled.withUnsafeMutableBufferPointer { dst in
                vDSP_vsmul(src.baseAddress!, 1, &scale255, dst.baseAddress!, 1, vDSP_Length(count))
            }
        }
        
        var binary = [UInt8](repeating: 0, count: count)
        scaled.withUnsafeBufferPointer { src in
            binary.withUnsafeMutableBufferPointer { dst in
                vDSP_vfixu8(src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(count))
            }
        }
        return binary
    }
    
    func rasterizePolygonFill(polygon: [(Int,Int)],
                              outMask: inout [UInt8],
                              width: Int,
                              height: Int)
    {
        if polygon.count < 3 { return }

        // Compute bounding box
        var minY = height, maxY = 0
        for (x,y) in polygon {
            if y < minY { minY = y }
            if y > maxY { maxY = y }
        }
        minY = max(minY,0)
        maxY = min(maxY,height-1)

        // Scanline fill
        for fy in minY...maxY {
            var xs: [Int] = []
            xs.reserveCapacity(16)

            for i in 0..<polygon.count {
                let (x0,y0) = polygon[i]
                let (x1,y1) = polygon[(i+1)%polygon.count]
                if y0 == y1 { continue }
                let ymin = min(y0,y1)
                let ymax = max(y0,y1)
                if fy >= ymin && fy < ymax {
                    let dy = y1 - y0
                    let dx = x1 - x0
                    let t = Float(fy - y0) / Float(dy)
                    let xf = Float(x0) + t * Float(dx)
                    xs.append(Int(xf))
                }
            }

            if xs.count < 2 { continue }
            xs.sort()
            var i = 0
            while i + 1 < xs.count {
                let xStart = max(0, min(xs[i], width-1))
                let xEnd   = max(0, min(xs[i+1], width-1))
                if xEnd > xStart {
                    for x in xStart..<xEnd {
                        outMask[fy*width + x] = 1
                    }
                }
                i += 2
            }
        }
    }

    
    func constructPolygonFromEdges(_ edges: [(start:(x:Int,y:Int),end:(x:Int,y:Int))])
        -> [(Int,Int)] {

        // Build adjacency
        var next: [String:(x:Int,y:Int)] = [:]
        func key(_ p:(x:Int,y:Int)) -> String { "\(p.x)_\(p.y)" }
        for e in edges {
            next[key(e.start)] = e.end
        }

        // Build chain
        var polygon: [(Int,Int)] = []
        guard let first = edges.first else { return [] }
        var cur = first.start
        polygon.append(cur)

        var safety = 0
        while let n = next[key(cur)], safety < 10000 {
            if n.x == polygon.first!.0 && n.y == polygon.first!.1 { break }
            polygon.append((n.x,n.y))
            cur = n
            safety += 1
        }

        return polygon
    }

    
    // MARK: cv2.findContours(RETR_EXTERNAL) Equivalent
    func findContoursExternal(binary: [UInt8], w: Int, h: Int) -> [[(x: Int, y: Int)]] {
        var visited = Array(repeating: false, count: w * h)
        var contours: [[(Int,Int)]] = []

        let dirs = [(1,0),(-1,0),(0,1),(0,-1)]

        func dfs(_ sx: Int, _ sy: Int) -> [(Int,Int)] {
            var stack = [(sx, sy)]
            var out: [(x: Int, y: Int)] = []
            while let (x,y) = stack.popLast() {
                let idx = y*w + x
                if visited[idx] { continue }
                visited[idx] = true
                if binary[idx] == 0 { continue }
                out.append((x,y))
                for (dx,dy) in dirs {
                    let nx=x+dx, ny=y+dy
                    if nx>=0 && nx<w && ny>=0 && ny<h {
                        if binary[ny*w+nx] > 0 && !visited[ny*w+nx] {
                            stack.append((nx,ny))
                        }
                    }
                }
            }
            return out
        }

        for y in 0..<h {
            for x in 0..<w {
                let idx = y*w + x
                if binary[idx] > 0 && !visited[idx] {
                    let c = dfs(x,y)
                    if !c.isEmpty { contours.append(c) }
                }
            }
        }

        return contours
    }

    
    //
    // MARK: - Python-Exact Concave Hull + Raster Fill
    //
    func buildConcaveHullMaskPythonExact(unionMask: UnsafeMutablePointer<UInt8>,
                                         width: Int,
                                         height: Int,
                                         alpha: Float = 2.5) -> [UInt8] {

        // ---------------------------------------------------------------
        // 1. Convert unionMask → UInt8 2D array for contour detection
        // ---------------------------------------------------------------
        var bin: [UInt8] = Array(repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            bin[i] = unionMask[i] > 0 ? 1 : 0
        }

        // ---------------------------------------------------------------
        // 2. Extract contours EXACTLY like Python cv2.findContours(..., RETR_EXTERNAL)
        // ---------------------------------------------------------------
        let contours = findContoursExternal(binary: bin, w: width, h: height)
        if contours.isEmpty {
            return bin  // nothing to fill
        }

        // ---------------------------------------------------------------
        // 3. Subsample contour points like Python
        // Python code:
        //   step = 1 if len(c) < 500 else max(1, len(c)//500)
        // ---------------------------------------------------------------
        var allPts: [(Int,Int)] = []

        for c in contours {
            let n = c.count
            let step = (n < 500) ? 1 : max(1, n / 500)
            var idx = 0
            for p in c {
                if idx % step == 0 {
                    allPts.append((p.x, p.y))
                }
                idx += 1
            }
        }

        if allPts.count < 4 {
            return bin   // identical to python fallback
        }

        // ---------------------------------------------------------------
        // 4. Compute alpha shape using your working Delaunay + hull code
        // returns edges forming polygon boundaries
        // ---------------------------------------------------------------
        let edges = AlphaShape.compute(points: allPts, alpha: alpha)
        if edges.isEmpty {
            return bin
        }

        // ---------------------------------------------------------------
        // 5. Reconstruct polygon(s) EXACTLY like Python polygonize
        // (We treat edges as one exterior polygon – same behavior your Python gave)
        // ---------------------------------------------------------------
        let polygon = constructPolygonFromEdges(edges)

        // ---------------------------------------------------------------
        // 6. Rasterize the polygon to a filled mask
        // MATCHES cv2.fillPoly (scanline fill)
        // ---------------------------------------------------------------
        var outMask = Array(repeating: UInt8(0), count: width * height)
        rasterizePolygonFill(polygon: polygon, outMask: &outMask, width: width, height: height)

        return outMask
    }
    
    
    func generateCutout(
        stage1Prototypes: MLMultiArray,
        primaryBBoxes: [DetectionSmarty],
        masks: [[UInt8]]?,                 // optional precomputed full-image masks (0/255)
        originalImage: CVPixelBuffer,
        outlineWidth: Int = 1,
        outlineColor: (r: UInt8, g: UInt8, b: UInt8, a: UInt8) = (255, 0, 0, 255),
        debugMode: Bool = false
    ) {
        CVPixelBufferLockBaseAddress(originalImage, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(originalImage, .readOnly) }

        let W = CVPixelBufferGetWidth(originalImage)
        let H = CVPixelBufferGetHeight(originalImage)
        let fullCount = W * H
        let dstRowBytes = W * 4

        // Validate masks
        var useMasks = masks
        if let m = masks {
            if m.count != primaryBBoxes.count || m.first?.count != fullCount {
                if debugMode { print("⚠️ generateCutout: masks mismatch; falling back to computed masks") }
                useMasks = nil
            }
        }

        // Source BGRA bytes
        guard let srcBase = CVPixelBufferGetBaseAddress(originalImage) else { return }
        let srcRowBytes = CVPixelBufferGetBytesPerRow(originalImage)
        let srcPtr = srcBase.assumingMemoryBound(to: UInt8.self)

        // Result buffer (premultiplied ARGB interleaved)
        guard let resultData = malloc(H * dstRowBytes) else { return }
        defer { free(resultData) }
        let resultPtr = resultData.assumingMemoryBound(to: UInt8.self)
        memset(resultPtr, 0, H * dstRowBytes)

        // Convert source BGRA -> premultiplied ARGB buffer
        guard let srcARGBData = malloc(H * dstRowBytes) else { return }
        defer { free(srcARGBData) }
        let srcARGB = srcARGBData.assumingMemoryBound(to: UInt8.self)
        for y in 0..<H {
            let srcRow = srcPtr.advanced(by: y * srcRowBytes)
            let dstRow = srcARGB.advanced(by: y * dstRowBytes)
            for x in 0..<W {
                let b = srcRow[x*4 + 0]
                let g = srcRow[x*4 + 1]
                let r = srcRow[x*4 + 2]
                let a = srcRow[x*4 + 3]
                dstRow[x*4 + 0] = a
                dstRow[x*4 + 1] = UInt8((UInt16(r) * UInt16(a) + 127) / 255)
                dstRow[x*4 + 2] = UInt8((UInt16(g) * UInt16(a) + 127) / 255)
                dstRow[x*4 + 3] = UInt8((UInt16(b) * UInt16(a) + 127) / 255)
            }
        }

        // Compute masks if not provided
        var computedMasks: [[UInt8]]? = useMasks
        if useMasks == nil {
            computedMasks = processMasks(protos: stage1Prototypes,
                                         maskCoeffs: primaryBBoxes.map { $0.maskCoeffs },
                                         boxes: primaryBBoxes,
                                         outSize: (height: H, width: W),
                                         upsample: true)
        }
        guard let masksToUse = computedMasks else { return }

        // Dilate helper via integral-image existence test
        func dilateMask(boxRadius: Int, srcAlpha: UnsafePointer<UInt8>, dstOut: UnsafeMutablePointer<UInt8>) {
            var integral = [UInt32](repeating: 0, count: (W+1)*(H+1))
            for y in 0..<H {
                var rowSum: UInt32 = 0
                for x in 0..<W {
                    rowSum += UInt32(srcAlpha[y*W + x])
                    integral[(y+1)*(W+1) + (x+1)] = integral[y*(W+1) + (x+1)] + rowSum
                }
            }
            for y in 0..<H {
                let y0 = max(0, y - boxRadius)
                let y1 = min(H-1, y + boxRadius)
                for x in 0..<W {
                    let x0 = max(0, x - boxRadius)
                    let x1 = min(W-1, x + boxRadius)
                    let sum = integral[(y1+1)*(W+1) + (x1+1)]
                            - integral[(y0)*(W+1) + (x1+1)]
                            - integral[(y1+1)*(W+1) + (x0)]
                            + integral[(y0)*(W+1) + (x0)]
                    dstOut[y*W + x] = (sum > 0) ? 255 : 0
                }
            }
        }

        // Temp buffers
        var dilated = [UInt8](repeating: 0, count: fullCount)
        var outlinePlane = [UInt8](repeating: 0, count: fullCount)
        let srcARGBPtr = srcARGB
        let resultARGBPtr = resultPtr
        let rC = outlineColor.r, gC = outlineColor.g, bC = outlineColor.b

        // Per-detection processing
        for (i, _) in primaryBBoxes.enumerated() {
            let mask = masksToUse[i]

            // Build premultiplied ARGB mask (alpha from mask, RGB from outlineColor)
            guard let maskARGBData = malloc(H * dstRowBytes) else { return }
            defer { free(maskARGBData) }
            let maskARGB = maskARGBData.assumingMemoryBound(to: UInt8.self)
            mask.withUnsafeBufferPointer { mPtr in
                let aPtr = mPtr.baseAddress!
                for p in 0..<fullCount {
                    let a = aPtr[p]
                    maskARGB[p*4 + 0] = a
                    maskARGB[p*4 + 1] = UInt8((UInt16(rC) * UInt16(a) + 127) / 255)
                    maskARGB[p*4 + 2] = UInt8((UInt16(gC) * UInt16(a) + 127) / 255)
                    maskARGB[p*4 + 3] = UInt8((UInt16(bC) * UInt16(a) + 127) / 255)
                }
            }

            // Blend source into result using mask (both premultiplied)
            for p in 0..<fullCount {
                let ma = maskARGB[p*4 + 0]
                if ma == 0 { continue }
                if ma == 255 {
                    resultARGBPtr[p*4 + 0] = srcARGBPtr[p*4 + 0]
                    resultARGBPtr[p*4 + 1] = srcARGBPtr[p*4 + 1]
                    resultARGBPtr[p*4 + 2] = srcARGBPtr[p*4 + 2]
                    resultARGBPtr[p*4 + 3] = srcARGBPtr[p*4 + 3]
                    continue
                }
                let invA = 255 - ma
                let sA = Int(srcARGBPtr[p*4 + 0])
                let dA = Int(resultARGBPtr[p*4 + 0])
                let outA = (sA * Int(ma) + dA * Int(invA) + 127) / 255
                resultARGBPtr[p*4 + 0] = UInt8(outA)
                for c in 1...3 {
                    let sC = Int(srcARGBPtr[p*4 + c])
                    let dC = Int(resultARGBPtr[p*4 + c])
                    let oC = (sC * Int(ma) + dC * Int(invA) + 127) / 255
                    resultARGBPtr[p*4 + c] = UInt8(oC)
                }
            }

            // Dilate mask
            mask.withUnsafeBufferPointer { mPtr in
                dilateMask(boxRadius: outlineWidth, srcAlpha: mPtr.baseAddress!, dstOut: &dilated)
            }

            // outline = dilated - mask (clamped)
            mask.withUnsafeBufferPointer { mPtr in
                let aPtr = mPtr.baseAddress!
                for p in 0..<fullCount {
                    let val = Int(dilated[p]) - Int(aPtr[p])
                    outlinePlane[p] = UInt8(clamping: val)
                }
            }

            // Build premultiplied ARGB outline and blend over result
            guard let outlineARGBData = malloc(H * dstRowBytes) else { return }
            defer { free(outlineARGBData) }
            let outlineARGB = outlineARGBData.assumingMemoryBound(to: UInt8.self)
            for p in 0..<fullCount {
                let a = outlinePlane[p]
                outlineARGB[p*4 + 0] = a
                outlineARGB[p*4 + 1] = UInt8((UInt16(rC) * UInt16(a) + 127) / 255)
                outlineARGB[p*4 + 2] = UInt8((UInt16(gC) * UInt16(a) + 127) / 255)
                outlineARGB[p*4 + 3] = UInt8((UInt16(bC) * UInt16(a) + 127) / 255)
            }
            for p in 0..<fullCount {
                let ma = outlineARGB[p*4 + 0]
                if ma == 0 { continue }
                if ma == 255 {
                    resultARGBPtr[p*4 + 0] = outlineARGB[p*4 + 0]
                    resultARGBPtr[p*4 + 1] = outlineARGB[p*4 + 1]
                    resultARGBPtr[p*4 + 2] = outlineARGB[p*4 + 2]
                    resultARGBPtr[p*4 + 3] = outlineARGB[p*4 + 3]
                    continue
                }
                let invA = 255 - ma
                let sA = Int(outlineARGB[p*4 + 0])
                let dA = Int(resultARGBPtr[p*4 + 0])
                let outA = (sA * Int(ma) + dA * Int(invA) + 127) / 255
                resultARGBPtr[p*4 + 0] = UInt8(outA)
                for c in 1...3 {
                    let sC = Int(outlineARGB[p*4 + c])
                    let dC = Int(resultARGBPtr[p*4 + c])
                    let oC = (sC * Int(ma) + dC * Int(invA) + 127) / 255
                    resultARGBPtr[p*4 + c] = UInt8(oC)
                }
            }
        }

        if debugMode { print("generateCutout finished; processed \(primaryBBoxes.count) detections") }

        // resultPtr contains premultiplied ARGB. If you need BGRA CVPixelBuffer output, I can append unpremultiply+permute + buffer creation.
    }




    // maskPlaneBuf: vImage_Buffer with Planar8 alpha (rowBytes = width)
    // outARGBBuf: vImage_Buffer pointing to allocated ARGB8888 buffer (rowBytes = width*4)
    // outlineColor: (r,g,b,a) as UInt8 but we only need r,g,b here (we'll premultiply)
    func buildPremultipliedARGBMask(maskPlaneBuf: inout vImage_Buffer, outARGBBuf: inout vImage_Buffer, outlineColor: (r: UInt8,g: UInt8,b: UInt8)) -> vImage_Error {
        // Create constant RGB planes filled with outlineColor components
        // vImage expects pointers to planar sources (A,R,G,B). We'll pass alpha as maskPlaneBuf and use constant fills for RGB.
        // Create temporary planar buffers for R,G,B filled with respective color values scaled to 255 (we'll premultiply later)
        let width = Int(maskPlaneBuf.width)
        let height = Int(maskPlaneBuf.height)
        let planeBytes = width * height

        // allocate small temporary constant planes (reuse outside loop ideally)
        let rPlane = malloc(planeBytes)!
        let gPlane = malloc(planeBytes)!
        let bPlane = malloc(planeBytes)!
        defer { free(rPlane); free(gPlane); free(bPlane) }

        memset(rPlane, Int32(outlineColor.r), planeBytes)
        memset(gPlane, Int32(outlineColor.g), planeBytes)
        memset(bPlane, Int32(outlineColor.b), planeBytes)

        var rBuf = vImage_Buffer(data: rPlane, height: maskPlaneBuf.height, width: maskPlaneBuf.width, rowBytes: width)
        var gBuf = vImage_Buffer(data: gPlane, height: maskPlaneBuf.height, width: maskPlaneBuf.width, rowBytes: width)
        var bBuf = vImage_Buffer(data: bPlane, height: maskPlaneBuf.height, width: maskPlaneBuf.width, rowBytes: width)

        // vImageConvert_Planar8toARGB8888 expects source planes in order A,R,G,B and writes interleaved ARGB8888
        let err = vImageConvert_Planar8toARGB8888(&maskPlaneBuf, &rBuf, &gBuf, &bBuf, &outARGBBuf, vImage_Flags(kvImageNoFlags))
        if err != kvImageNoError { return err }

        // Now premultiply (makes RGB := RGB * (A/255))
        return vImagePremultiplyData_ARGB8888(&outARGBBuf, &outARGBBuf, vImage_Flags(kvImageNoFlags))
    }

    // maskPlaneBuf: source Planar8 alpha (in), tmpDilatedBuf: Planar8 buffer (out), outlineARGBBuf: ARGB8888 buffer (out)
    // kernelSize = 2*outlineWidth + 1
    func buildOutlineARGB(maskPlaneBuf: inout vImage_Buffer, tmpDilatedBuf: inout vImage_Buffer, outlineARGBBuf: inout vImage_Buffer, outlineColor: (r: UInt8,g: UInt8,b: UInt8), kernelSize: Int) -> vImage_Error {
        // Dilate mask into tmpDilatedBuf
        // Create kernel filled with 1s
        var kernel = [UInt8](repeating: 1, count: kernelSize * kernelSize)
        var kerr: vImage_Error = kvImageNoError
        kernel.withUnsafeBufferPointer { kptr in
            kerr = vImageDilate_Planar8(&maskPlaneBuf, &tmpDilatedBuf, 0, 0, kptr.baseAddress!, vImagePixelCount(kernelSize), vImagePixelCount(kernelSize), vImage_Flags(kvImageNoFlags))
        }
        if kerr != kvImageNoError { return kerr }

        // outlinePlane = tmpDilated - maskPlane (clamped)
        // Use vImageSubtract_Planar8: dest = src2 - src1 -> we want (dilated - mask)
        // vImageSubtract_Planar8(src1, src2, dest, flags) does dest = src2 - src1
//        var outlinePlaneData = malloc(Int(maskPlaneBuf.height) * Int(maskPlaneBuf.width))!
//        defer { free(outlinePlaneData) }
        
        // outlinePlane = tmpDilated - maskPlane (clamped)
        // Manual per-pixel subtraction to avoid unavailable vImageSubtract_Planar8
        let planeWidth = Int(maskPlaneBuf.width)
        let planeHeight = Int(maskPlaneBuf.height)
        let planeBytes = planeWidth * planeHeight

        var outlinePlaneData = malloc(planeBytes)!
        defer { free(outlinePlaneData) }
        var outlinePlaneBuf = vImage_Buffer(data: outlinePlaneData,
                                            height: maskPlaneBuf.height,
                                            width: maskPlaneBuf.width,
                                            rowBytes: planeWidth)

        let mPtr = maskPlaneBuf.data!.assumingMemoryBound(to: UInt8.self)
        let dPtr = tmpDilatedBuf.data!.assumingMemoryBound(to: UInt8.self)
        let oPtr = outlinePlaneBuf.data!.assumingMemoryBound(to: UInt8.self)
        for i in 0..<planeBytes {
            let val = Int(dPtr[i]) - Int(mPtr[i])
            oPtr[i] = UInt8(clamping: val)
        }
//        var outlinePlaneBuf = vImage_Buffer(data: outlinePlaneData, height: maskPlaneBuf.height, width: maskPlaneBuf.width, rowBytes: Int(maskPlaneBuf.width))
//        let subErr = vImageSubtract_Planar8(&maskPlaneBuf, &tmpDilatedBuf, &outlinePlaneBuf, vImage_Flags(kvImageNoFlags))
//        if subErr != kvImageNoError { return subErr }
        // Note: vImageSubtract_Planar8 order: dest = src2 - src1 -> so passed (mask, dilated, dest) gives dilated - mask.

        // Convert outlinePlane (A) + constant RGB into ARGB8888
        // Create constant RGB planes filled with outlineColor
        let width = Int(maskPlaneBuf.width)
        let height = Int(maskPlaneBuf.height)
        let planeBytess = width * height
        let rPlane = malloc(planeBytess)!
        let gPlane = malloc(planeBytess)!
        let bPlane = malloc(planeBytess)!
        defer { free(rPlane); free(gPlane); free(bPlane) }
        memset(rPlane, Int32(outlineColor.r), planeBytess)
        memset(gPlane, Int32(outlineColor.g), planeBytess)
        memset(bPlane, Int32(outlineColor.b), planeBytess)
        var rBuf = vImage_Buffer(data: rPlane, height: maskPlaneBuf.height, width: maskPlaneBuf.width, rowBytes: width)
        var gBuf = vImage_Buffer(data: gPlane, height: maskPlaneBuf.height, width: maskPlaneBuf.width, rowBytes: width)
        var bBuf = vImage_Buffer(data: bPlane, height: maskPlaneBuf.height, width: maskPlaneBuf.width, rowBytes: width)

        let convErr = vImageConvert_Planar8toARGB8888(&outlinePlaneBuf, &rBuf, &gBuf, &bBuf, &outlineARGBBuf, vImage_Flags(kvImageNoFlags))
        if convErr != kvImageNoError { return convErr }

        // Premultiply outlineARGB
        return vImagePremultiplyData_ARGB8888(&outlineARGBBuf, &outlineARGBBuf, vImage_Flags(kvImageNoFlags))
    }

    // ======================================================
    // SAFE LABEL + BOX DRAWING (NO CoreText, NO CTLineDraw)
    // ======================================================
    // ======================================================
    // SAFE LABEL + BOX DRAWING  (CYAN, BIG FONT, FIXED FLIP)
    // ======================================================
    private func drawLabelsAndBoxes(
        ctx: CGContext,
        stage1: [DetectionSmarty],
        stage2: [DetectionSmarty],
        imageWidth: Int,
        imageHeight: Int,
        drawBoxes: Bool
    ) {
        let all = stage1 + stage2
        guard !all.isEmpty else { return }

        let W = CGFloat(imageWidth)
        let H = CGFloat(imageHeight)
        let modelSize: CGFloat = 1280.0
        let sx = W / modelSize
        let sy = H / modelSize

        // Fix UIKit upside-down drawing in CGContexts
        ctx.saveGState()
        ctx.translateBy(x: 0, y: H)
        ctx.scaleBy(x: 1, y: -1)

        UIGraphicsPushContext(ctx)

        let font = UIFont.boldSystemFont(ofSize: 38)

        for det in all {

            let cx = CGFloat(det.x)
            let cy = CGFloat(det.y)
            let w  = CGFloat(det.width)
            let h  = CGFloat(det.height)

            let left = (cx - w / 2) * sx
            let top  = (cy - h / 2) * sy
            let rect = CGRect(x: left, y: top, width: w * sx, height: h * sy)

            // ---- Cyan Box (optional) ----
            if drawBoxes {
                UIColor.cyan.setStroke()
                let b = UIBezierPath(rect: rect)
                b.lineWidth = 4
                b.stroke()
            }

            // ---- Label text ----
            let textString = "\(det.className) \(Int(det.confidence * 100))%"
            let attributes: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: UIColor.cyan,                                  // CYAN TEXT
                .backgroundColor: UIColor.black.withAlphaComponent(0.6),        // DARK BACK
                .shadow: {
                    let sh = NSShadow()
                    sh.shadowBlurRadius = 6
                    sh.shadowOffset = CGSize(width: 2, height: -2)
                    sh.shadowColor = UIColor.black.withAlphaComponent(0.8)
                    return sh
                }()
            ]

            let text = NSAttributedString(string: textString, attributes: attributes)
            let size = text.size()

            var tx = max(0, min(left, W - size.width - 4))
            var ty = top - size.height - 6

            if ty < 0 { ty = top + 6 }

            let drawRect = CGRect(x: tx, y: ty, width: size.width, height: size.height)

            text.draw(in: drawRect)
        }

        UIGraphicsPopContext()
        ctx.restoreGState()
    }


    // ======================================================
    // FINAL LABEL RENDERING — SAFE, CRASH-PROOF
    // Draws both Stage 1 + Stage 2 labels on final CGImage
    // ======================================================
    private func renderLabelsOnFinalImage(
        baseCGImage: CGImage,
        width: Int,
        height: Int,
        stage1: [DetectionSmarty],
        stage2: [DetectionSmarty]
    ) -> CGImage {

        let colorSpace = CGColorSpaceCreateDeviceRGB()

        guard let ctx = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            print("❌ renderLabelsOnFinalImage: failed to create CGContext")
            return baseCGImage
        }

        // Draw the already-rendered cutout mask
        ctx.draw(baseCGImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Return the final composited CGImage
        return ctx.makeImage() ?? baseCGImage
    }

    private func renderFinalMaskAndLabels(
        mergedMaskUpscaled: CGImage,
        outputWidth: Int,
        outputHeight: Int,
        stage1Detections: [DetectionSmarty],
        stage2DetectionsMapped: [DetectionSmarty],
        tightBBoxRect: CGRect
    ) {
        let startTime = Date()

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil,
            width: outputWidth,
            height: outputHeight,
            bitsPerComponent: 8,
            bytesPerRow: outputWidth * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            print("❌ CGContext failed")
            return
        }

        // ---------------------------------------
        // 1️⃣ Draw the cutout mask (already upscaled)
        // ---------------------------------------
        ctx.draw(mergedMaskUpscaled, in: CGRect(x: 0, y: 0,
                                                width: outputWidth,
                                                height: outputHeight))

        // ---------------------------------------
        // 2️⃣ Clip to the tight bounding box for cutout,
        //    but labels must be drawn OUTSIDE clip.
        // ---------------------------------------
        ctx.saveGState()
        ctx.clip(to: tightBBoxRect)

        // (Your cutout drawing happens here if required)

        ctx.restoreGState()   // Important: remove clipping before labels.

        // ---------------------------------------
        // 3️⃣ Labels must be drawn AFTER clipping is removed
        // ---------------------------------------
        ctx.resetClip()   // ← ***THE CRITICAL FIX***

        // Draw labels for Stage 1 detections
        for det in stage1Detections {
            drawDetectionLabelOnly(
                ctx: ctx,
                detection: det,
                imageWidth: outputWidth,
                imageHeight: outputHeight
            )
        }

        // Draw labels for Stage 2 detections
        for det in stage2DetectionsMapped {
            drawDetectionLabelOnly(
                ctx: ctx,
                detection: det,
                imageWidth: outputWidth,
                imageHeight: outputHeight
            )
        }

        let endTime = Date()
        if self.debugMode {
            print(String(format: "⏱ renderFinalMaskAndLabels: %.2f ms",
                         (endTime.timeIntervalSince(startTime) * 1000)))
        }

        // ---------------------------------------
        // 4️⃣ Export final image
        // ---------------------------------------
        if let result = ctx.makeImage() {
            DispatchQueue.main.async {
                self.maskImageView.image = UIImage(cgImage: result)
            }
        }
    }

    private func drawDetectionLabelOnly(
        ctx: CGContext,
        detection: DetectionSmarty,
        imageWidth: Int,
        imageHeight: Int
    ) {
        // ---------------------------------------
        // Convert Float → CGFloat
        // ---------------------------------------
        let cx = CGFloat(detection.x)
        let cy = CGFloat(detection.y)
        let w  = CGFloat(detection.width)
        let h  = CGFloat(detection.height)

        // YOLOE outputs are scaled to 1280×1280 model input
        let sx = CGFloat(imageWidth) / 1280.0
        let sy = CGFloat(imageHeight) / 1280.0

        // ---------------------------------------
        // Compute top-left of bbox in output image
        // ---------------------------------------
        var rectX = (cx - w/2.0) * sx
        var rectY = (cy - h/2.0) * sy

        // Clamp inside final image
        rectX = max(0, min(rectX, CGFloat(imageWidth - 1)))
        rectY = max(0, min(rectY, CGFloat(imageHeight - 1)))

        // ---------------------------------------
        // Prepare label text (include track ID if available)
        // ---------------------------------------
//        let trackPrefix = detection.trackId != nil ? "ID\(detection.trackId!) " : ""
        let label = "\(detection.className) \(Int(detection.confidence * 100))%"
        let font = UIFont.boldSystemFont(ofSize: 26)

        let attrs: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: UIColor.white,
            .backgroundColor: UIColor.black.withAlphaComponent(0.65)
        ]

        let text = NSAttributedString(string: label, attributes: attrs)
        let textSize = text.size()

        // Draw label slightly above bbox
        let textRect = CGRect(
            x: rectX,
            y: max(0, rectY - textSize.height - 4),
            width: textSize.width,
            height: textSize.height
        )

        ctx.saveGState()
        text.draw(in: textRect)
        ctx.restoreGState()
    }



    private func drawBoundingBox(ctx: CGContext, detection: DetectionSmarty, imageWidth: Int, imageHeight: Int) {
        let originalWidth = CGFloat(imageWidth)
        let originalHeight = CGFloat(imageHeight)
        let modelSize: CGFloat = 1280.0
        let scaleX = originalWidth / modelSize
        let scaleY = originalHeight / modelSize

        let centerX = CGFloat(detection.x) * scaleX
        let centerY = CGFloat(detection.y) * scaleY
        let boxWidth = CGFloat(detection.width) * scaleX
        let boxHeight = CGFloat(detection.height) * scaleY

        let x = centerX - boxWidth / 2
        let y = centerY - boxHeight / 2

        // cyan box
        ctx.setStrokeColor(CGColor(red: 0, green: 1, blue: 1, alpha: 1))
        ctx.setLineWidth(3.0)
        let rect = CGRect(x: x, y: y, width: boxWidth, height: boxHeight)
        ctx.stroke(rect)

        // label text (include track ID if available)
        let confidence = Int(detection.confidence * 100)
//        let trackIdStr = detection.trackId != nil ? "ID\(detection.trackId!) " : ""
        let labelText = "\(detection.className) \(confidence)%"
        let attributed = NSAttributedString(string: labelText, attributes: bboxAttributes)
        let textSize = attributed.size()

        let labelPadding: CGFloat = 6
        let labelWidth = textSize.width + (labelPadding * 2)
        let labelHeight = textSize.height + (labelPadding * 2)

        var labelX: CGFloat
        var labelY: CGFloat

        if y - labelHeight - 5 >= 0 {
            labelX = max(0, min(x, originalWidth - labelWidth))
            labelY = y - labelHeight - 5
        } else if y + boxHeight + labelHeight + 5 <= originalHeight {
            labelX = max(0, min(x, originalWidth - labelWidth))
            labelY = y + boxHeight + 5
        } else {
            labelX = max(0, min(x + 5, originalWidth - labelWidth))
            labelY = max(0, y + 5)
        }

        ctx.setFillColor(CGColor(red: 0, green: 1, blue: 1, alpha: 1))
        let labelRect = CGRect(x: labelX, y: labelY, width: labelWidth, height: labelHeight)
        ctx.fill(labelRect)

        let textX = labelX + labelPadding
        let textY = labelY + labelPadding

        let line = CTLineCreateWithAttributedString(attributed)

        ctx.saveGState()
        ctx.textMatrix = .identity

        let ctm = ctx.ctm
        let isFlipped = ctm.d < 0 || ctm.ty != 0

        if isFlipped {
            ctx.translateBy(x: 0, y: CGFloat(imageHeight))
            ctx.scaleBy(x: 1.0, y: -1.0)
            let ascent = CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
            let flippedY = CGFloat(imageHeight) - textY - ascent
            ctx.textPosition = CGPoint(x: textX, y: flippedY)
        } else {
            ctx.textPosition = CGPoint(x: textX, y: textY)
        }

        CTLineDraw(line, ctx)
        ctx.restoreGState()
    }
//    private var detFloatBuf: UnsafeMutablePointer<Float>? = nil
    
//    private var tempConf: [Float] = []
    
    private func extractDetections(from detections: MLMultiArray, confThreshold: Float, topK: Int = 50) -> [DetectionSmarty] {
        let tAll = Date()
        var results: [DetectionSmarty] = []

        let shapeInts = detections.shape.map { $0.intValue }
        guard shapeInts.count >= 3 else {
            if debugMode { print("⚠️ extractDetections: unexpected shape \(shapeInts)") }
            return []
        }
        let numFeatures = shapeInts[1]
        let numAnchors = shapeInts[2]
        let protoDim = 32
        let numClasses = numFeatures - 4 - protoDim
        guard numFeatures >= 4 + numClasses + protoDim && numAnchors > 0 && numClasses > 0 else {
            if debugMode { print("⚠️ extractDetections: invalid dims F:\(numFeatures) A:\(numAnchors) nc:\(numClasses)") }
            return []
        }

        if debugMode {
            print("🔍 detections: features=\(numFeatures) anchors=\(numAnchors) classes=\(numClasses)")
        }

        let expectedCount = numFeatures * numAnchors

        // Ensure persistent buffer capacity
        if detFloatBuf == nil || detFloatBufCapacity < expectedCount {
            if let existing = detFloatBuf { existing.deallocate() }
            detFloatBuf = UnsafeMutablePointer<Float>.allocate(capacity: expectedCount)
            detFloatBufCapacity = expectedCount
        }
        guard let detBuf = detFloatBuf else { return [] }

        // Copy MLMultiArray to detBuf
        if detections.dataType == .float16 {
            let src = detections.dataPointer.bindMemory(to: UInt16.self, capacity: expectedCount)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src),
                                       height: 1, width: vImagePixelCount(expectedCount),
                                       rowBytes: expectedCount * MemoryLayout<UInt16>.size)
            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(detBuf),
                                       height: 1, width: vImagePixelCount(expectedCount),
                                       rowBytes: expectedCount * MemoryLayout<Float>.size)
            let r = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
            if r != kvImageNoError && debugMode { print("⚠️ vImage conv16->32 failed: \(r)") }
        } else if detections.dataType == .float32 {
            let src = detections.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(detBuf, src, expectedCount * MemoryLayout<Float>.size)
        } else {
            for i in 0..<expectedCount { detBuf[i] = detections[i].floatValue }
        }
        if debugMode {
            print(String(format: "⏱ copy dets: %.2f ms", Date().timeIntervalSince(tAll)*1000))
        }

        let detPtr = UnsafeMutableBufferPointer(start: detBuf, count: expectedCount)
        if tempConf.count < numClasses { tempConf = [Float](repeating: 0, count: numClasses) }

        let stride = numAnchors
        let confBaseOffset = 4 * stride
        let coeffsOffset = (4 + numClasses) * stride

        var intermediate: [DetectionSmarty] = []
        intermediate.reserveCapacity(min(512, numAnchors))

        let decodeStart = Date()
        for a in 0..<numAnchors {
            let ix = 0 * stride + a
            let iy = 1 * stride + a
            let iw = 2 * stride + a
            let ih = 3 * stride + a
            if ix >= expectedCount || iy >= expectedCount || iw >= expectedCount || ih >= expectedCount { continue }
            let x = detPtr[ix], y = detPtr[iy], w = detPtr[iw], h = detPtr[ih]
            if !x.isFinite || !y.isFinite || !w.isFinite || !h.isFinite || w <= 0 || h <= 0 { continue }

            // copy class confidences into tempConf (strided)
            let confStart = confBaseOffset + a
            for c in 0..<numClasses {
                let idx = confStart + c * stride
                tempConf[c] = (idx < expectedCount) ? detPtr[idx] : 0.0
            }

            // argmax via vDSP
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(tempConf, 1, &maxVal, &maxIdx, vDSP_Length(numClasses))
            let bestConf = maxVal
            let bestClass = Int(maxIdx)
            if bestConf <= confThreshold { continue }

            // read protoDim coeffs (strided)
            var coeffs = [Float](repeating: 0, count: protoDim)
            var coeffValid = true
            let coeffStart = coeffsOffset + a
            for k in 0..<protoDim {
                let idx = coeffStart + k * stride
                if idx < expectedCount { coeffs[k] = detPtr[idx] } else { coeffValid = false; break }
            }
            if !coeffValid { continue }

            let className = furnitureClasses[bestClass] ?? "object_\(bestClass)"
            let det = DetectionSmarty(x: x, y: y, width: w, height: h, confidence: bestConf, classIdx: bestClass, className: className, maskCoeffs: coeffs)
            intermediate.append(det)
        }
        if debugMode {
            print(String(format: "⏱ decode anchors: %.2f ms", Date().timeIntervalSince(decodeStart)*1000))
        }

        // keep top-K by confidence
        let kept = intermediate.sorted(by: { $0.confidence > $1.confidence }).prefix(topK)
        results = Array(kept)

        if debugMode {
            print("📊 extracted kept: \(results.count) (topK=\(topK))")
            print(String(format: "⏱ extractDetections total: %.2f ms", Date().timeIntervalSince(tAll)*1000))
        }

        return results
    }

    
    // Requires: import Accelerate, import CoreGraphics, import UIKit
    // Assumes originalImage is CVPixelBuffer in kCVPixelFormatType_32BGRA or similar.
    // Helper: convert CVPixelBuffer -> planar BGRA bytes
    private func cgImageFromPixelBuffer(_ pb: CVPixelBuffer) -> CGImage? {
        // existing helper you may already have; keep or replace with this minimal one
        let ci = CIImage(cvPixelBuffer: pb)
        let context = CIContext(options: nil)
        return context.createCGImage(ci, from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb)))
    }

//    import Accelerate
//    import Accelerate.vImage
//    import Foundation
//    import CoreML
//
//    struct DetectionSmarty { var x: Float; var y: Float; var width: Float; var height: Float }

    func processMasks(
        protos: MLMultiArray,
        maskCoeffs: [[Float]],
        boxes: [DetectionSmarty],
        outSize: (height: Int, width: Int),
        upsample: Bool,
        debugMode: Bool = false
    ) -> [[UInt8]] {

        let protoShape = protos.shape.map { $0.intValue }
        guard protoShape.count >= 3 else { return [] }

        let hasBatchDim = protoShape.count == 4
        let offset = hasBatchDim ? 1 : 0
        let C = protoShape[offset]
        let Hp = protoShape[offset + 1]
        let Wp = protoShape[offset + 2]
        let spatial = Hp * Wp
        let N = maskCoeffs.count
        guard N > 0 && C > 0 && boxes.count == N else { return [] }

        // Prepare protoMatrix
        var protoMatrix = [Float](repeating: 0, count: C * spatial)
        let t_proto_build_start = Date()

        // Fast path: bind data pointer (non-optional) and copy per-channel blocks concurrently
        let totalProtoCount = C * Hp * Wp
        let srcPtr = protos.dataPointer.bindMemory(to: Float.self, capacity: protos.count)

        if protos.count == totalProtoCount || protos.count == totalProtoCount * (hasBatchDim ? 1 : 1) {
            // Assume contiguous inner layout: either (1,C,Hp,Wp) or (C,Hp,Wp)
            DispatchQueue.concurrentPerform(iterations: C) { c in
                let dstBase = c * spatial
                let srcBase = (hasBatchDim ? (0 * C + c) : c) * spatial
                protoMatrix.withUnsafeMutableBufferPointer { dstBuf in
                    let dstPtr = dstBuf.baseAddress! + dstBase
                    let srcRow = srcPtr.advanced(by: srcBase)
                    dstPtr.assign(from: srcRow, count: spatial)
                }
            }
        } else {
            // Stride-aware fallback using MLMultiArray.strides
            let strides = protos.strides.map { $0.intValue }
            let bStride = hasBatchDim ? strides[0] : 0
            let cStride = strides[offset]
            let yStride = strides[offset + 1]
            let xStride = strides[offset + 2]

            for c in 0..<C {
                let dstBase = c * spatial
                for y in 0..<Hp {
                    for x in 0..<Wp {
                        let srcIndex = (hasBatchDim ? 0 * bStride : 0) + c * cStride + y * yStride + x * xStride
                        protoMatrix[dstBase + y * Wp + x] = srcPtr[srcIndex]
                    }
                }
            }
        }

        let t_proto_build_dt = Date().timeIntervalSince(t_proto_build_start) * 1000.0
        if debugMode { print(String(format: "⏱ processMasks: protoMatrix build: %.2f ms", t_proto_build_dt)) }

        // Buffers
        let fullCount = outSize.height * outSize.width
        var masksOut = Array(repeating: [UInt8](repeating: 0, count: fullCount), count: N)
        var protoFloatBuf = [Float](repeating: 0, count: spatial)
        var protoPlanar8 = [UInt8](repeating: 0, count: spatial)
        var upscaledPlanar8 = [UInt8](repeating: 0, count: fullCount)

        let vImageFlags = vImage_Flags(kvImageNoFlags)
        let upsampleFlags = vImage_Flags(kvImageHighQualityResampling)

        let m_blas = Int32(C)
        let n_blas = Int32(spatial)
        let lda = Int32(spatial)

        // timers
        var total_sgemv: Double = 0
        var total_f2p: Double = 0
        var total_upscale: Double = 0
        var total_finalize: Double = 0

        let t_loop_start = Date()
        for i in 0..<N {
            let coeffs = maskCoeffs[i]
            guard coeffs.count == C else { return [] }

            // SGEMV
            let t_sgemv_start = Date()
            protoFloatBuf.withUnsafeMutableBufferPointer { yPtr in
                protoMatrix.withUnsafeBufferPointer { aPtr in
                    coeffs.withUnsafeBufferPointer { xPtr in
                        cblas_sgemv(
                            CblasRowMajor,
                            CblasTrans,
                            m_blas,
                            n_blas,
                            1.0,
                            aPtr.baseAddress!,
                            lda,
                            xPtr.baseAddress!,
                            1,
                            0.0,
                            yPtr.baseAddress!,
                            1
                        )
                    }
                }
            }
            total_sgemv += Date().timeIntervalSince(t_sgemv_start) * 1000.0

            // bbox -> proto coords
            let det = boxes[i]
            let cx = det.x, cy = det.y, bw = det.width, bh = det.height
            var x1 = Int(floor((cx - bw/2) * Float(Wp)))
            var x2 = Int(ceil ((cx + bw/2) * Float(Wp)))
            var y1 = Int(floor((cy - bh/2) * Float(Hp)))
            var y2 = Int(ceil ((cy + bh/2) * Float(Hp)))
            x1 = max(0, min(Wp - 1, x1)); x2 = max(0, min(Wp, x2))
            y1 = max(0, min(Hp - 1, y1)); y2 = max(0, min(Hp, y2))

            // threshold negatives
            var thr: Float = 0.0
            protoFloatBuf.withUnsafeMutableBufferPointer { pPtr in
                vDSP_vthr(pPtr.baseAddress!, 1, &thr, pPtr.baseAddress!, 1, vDSP_Length(spatial))
            }

            // zero outside bbox
            protoFloatBuf.withUnsafeMutableBufferPointer { pPtr in
                let base = pPtr.baseAddress!
                if y1 > 0 {
                    memset(base, 0, y1 * Wp * MemoryLayout<Float>.size)
                }
                if y2 < Hp {
                    let start = y2 * Wp
                    let count = spatial - start
                    if count > 0 {
                        memset(base + start, 0, count * MemoryLayout<Float>.size)
                    }
                }
                for row in y1..<y2 {
                    let rowBase = base + row * Wp
                    if x1 > 0 {
                        memset(rowBase, 0, x1 * MemoryLayout<Float>.size)
                    }
                    if x2 < Wp {
                        memset(rowBase + x2, 0, (Wp - x2) * MemoryLayout<Float>.size)
                    }
                }
            }

            // Float -> Planar8
            let t_f2p_start = Date()
            protoFloatBuf.withUnsafeMutableBytes { fPtr in
                protoPlanar8.withUnsafeMutableBytes { pPtr in
                    var src = vImage_Buffer(data: fPtr.baseAddress!, height: vImagePixelCount(Hp), width: vImagePixelCount(Wp), rowBytes: Wp * MemoryLayout<Float>.size)
                    var dst = vImage_Buffer(data: pPtr.baseAddress!, height: vImagePixelCount(Hp), width: vImagePixelCount(Wp), rowBytes: Wp)
                    vImageConvert_PlanarFtoPlanar8(&src, &dst, 255.0, 0, vImageFlags)
                }
            }
            total_f2p += Date().timeIntervalSince(t_f2p_start) * 1000.0

            // Upsample
            let t_up_start = Date()
            protoPlanar8.withUnsafeMutableBytes { cPtr in
                upscaledPlanar8.withUnsafeMutableBytes { fPtr in
                    var src = vImage_Buffer(data: cPtr.baseAddress!, height: vImagePixelCount(Hp), width: vImagePixelCount(Wp), rowBytes: Wp)
                    var dst = vImage_Buffer(data: fPtr.baseAddress!, height: vImagePixelCount(outSize.height), width: vImagePixelCount(outSize.width), rowBytes: outSize.width)
                    vImageScale_Planar8(&src, &dst, nil, upsample ? upsampleFlags : vImageFlags)
                }
            }
            total_upscale += Date().timeIntervalSince(t_up_start) * 1000.0

            // Finalize
            let t_fin_start = Date()
            masksOut[i].withUnsafeMutableBufferPointer { oPtr in
                upscaledPlanar8.withUnsafeBufferPointer { iPtr in
                    if let dst = oPtr.baseAddress, let src = iPtr.baseAddress {
                        memcpy(dst, src, fullCount)
                    }
                }
            }
            total_finalize += Date().timeIntervalSince(t_fin_start) * 1000.0
        }

        let t_loop_dt = Date().timeIntervalSince(t_loop_start) * 1000.0
        if debugMode {
            print(String(format: "⏱ processMasks: loop total %.2f ms (N=%d)", t_loop_dt, N))
            print(String(format: "    sgemv total: %.2f ms (avg %.2f ms)", total_sgemv, total_sgemv / Double(N)))
            print(String(format: "    float->planar total: %.2f ms (avg %.2f ms)", total_f2p, total_f2p / Double(N)))
            print(String(format: "    upsample total: %.2f ms (avg %.2f ms)", total_upscale, total_upscale / Double(N)))
            print(String(format: "    finalize total: %.2f ms (avg %.2f ms)", total_finalize, total_finalize / Double(N)))
        }

        return masksOut
    }







    
    // MARK: - Pixel Buffer to MLMultiArray (Accelerate) — with timing
    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let t0 = Date()
        guard let array = try? MLMultiArray(shape: [1, 3, 1280, 1280], dataType: .float32) else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = 1280
        let height = 1280
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

        if self.debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ pixelBufferToMLMultiArray: %.2f ms", dt))
        }

        return array
    }
    
    // MARK: - Compute edge pixels (circumference) of a binary mask (4-neighborhood)
      private func computeMaskEdges(mask: [Float], width: Int, height: Int) -> [(x: Int, y: Int)] {
          let w = width, h = height
          guard mask.count == w * h, w > 0, h > 0 else { return [] }
          var edges: [(Int, Int)] = []
          edges.reserveCapacity(w * h / 8)
          for y in 0..<h {
              let rowOff = y * w
              for x in 0..<w {
                  let idx = rowOff + x
                  if mask[idx] <= 0 { continue }
                  // 4-neighborhood: if any neighbor is background or out of bounds, it's an edge
                  let leftEmpty   = (x == 0)        || mask[idx - 1] <= 0
                  let rightEmpty  = (x == w - 1)    || mask[idx + 1] <= 0
                  let topEmpty    = (y == 0)        || mask[idx - w] <= 0
                  let bottomEmpty = (y == h - 1)    || mask[idx + w] <= 0
                  if leftEmpty || rightEmpty || topEmpty || bottomEmpty {
                      edges.append((x, y))
                  }
              }
          }
          return edges
      }
    
    public func cutoutClearOutsideAccelerated(x0: Int, y0: Int, x1: Int, y1: Int, in image: CGImage) -> CGImage? {
        let t0 = Date()
        
        let width = image.width
        let height = image.height
        guard width > 0 && height > 0 else { return nil }

        var minX = min(x0, x1)
        var maxX = max(x0, x1)
        var minY = min(y0, y1)
        var maxY = max(y0, y1)

        minX = max(0, min(minX, width))
        maxX = max(0, min(maxX, width))
        minY = max(0, min(minY, height))
        maxY = max(0, min(maxY, height))

        if minX >= maxX || minY >= maxY {
            let out = makeTransparentImage(width: width, height: height)
            if self.debugMode {
                let dt = Date().timeIntervalSince(t0) * 1000.0
                print(String(format: "⏱ cutoutClearOutsideAccelerated (empty): %.2f ms", dt))
            }
            return out
        }

        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let bufSize = bytesPerRow * height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }

        guard let destData = malloc(bufSize) else { return nil }
        defer { free(destData) }

        guard let ctx = CGContext(data: destData,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            return nil
        }

        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1.0, y: -1.0)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var srcBuffer = vImage_Buffer(data: destData, height: vImagePixelCount(height),
                                      width: vImagePixelCount(width), rowBytes: bytesPerRow)

        guard let zeroRow = malloc(bytesPerRow) else { return nil }
        memset(zeroRow, 0, bytesPerRow)
        defer { free(zeroRow) }

        if minY > 0 {
            let dstPtr = destData
            for r in 0..<minY {
                let rowBase = dstPtr.advanced(by: r * bytesPerRow)
                memcpy(rowBase, zeroRow, bytesPerRow)
            }
        }

        if maxY < height {
            let dstPtr = destData.advanced(by: maxY * bytesPerRow)
            for r in 0..<(height - maxY) {
                let rowBase = dstPtr.advanced(by: r * bytesPerRow)
                memcpy(rowBase, zeroRow, bytesPerRow)
            }
        }

        if minX > 0 || maxX < width {
            let leftBytes = minX * bytesPerPixel
            let rightBytes = (width - maxX) * bytesPerPixel
            for row in minY..<maxY {
                let rowBase = destData.advanced(by: row * bytesPerRow)
                if leftBytes > 0 {
                    memset(rowBase, 0, leftBytes)
                }
                if rightBytes > 0 {
                    let rightPtr = rowBase.advanced(by: maxX * bytesPerPixel)
                    memset(rightPtr, 0, rightBytes)
                }
            }
        }

        guard let outCtx = CGContext(data: destData,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: 8,
                                     bytesPerRow: bytesPerRow,
                                     space: colorSpace,
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            return nil
        }

        let outImage = outCtx.makeImage()
        if self.debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ cutoutClearOutsideAccelerated: %.2f ms", dt))
        }
        return outImage
    }

    private func makeTransparentImage(width: Int, height: Int) -> CGImage? {
        guard width > 0 && height > 0 else { return nil }
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let bufSize = bytesPerRow * height
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        guard let data = calloc(1, bufSize) else { return nil }
        defer { free(data) }

        guard let ctx = CGContext(data: data,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }

        return ctx.makeImage()
    }

    public func cutoutClearOutsideAcceleratedUIImage(x0: Int, y0: Int, x1: Int, y1: Int, in image: UIImage) -> UIImage? {
        guard let cg = image.cgImage else { return nil }
        guard let outCG = cutoutClearOutsideAccelerated(x0: x0, y0: y0, x1: x1, y1: y1, in: cg) else { return nil }
        return UIImage(cgImage: outCG, scale: image.scale, orientation: image.imageOrientation)
    }

}
