// SmartyPantsView.swift
// Two-Stage Detection: Full frame -> Crop to primary bbox -> Re-detect -> UNION BOTH
// With timing logs at crucial stages + real progress bar until first detection

import SwiftUI
import UIKit
import CoreML
import Accelerate
import AVFoundation
import Photos

// MARK: - SwiftUI Wrapper
struct SmartyPantsViewSwiftUI: UIViewRepresentable {
    let mlModel: MLModel?
    var processInterval: TimeInterval = 0.1
    var confidenceThreshold: Float = 0.5
    
    var detectAllObjects: Bool = true
    var useBilinearUpscaling: Bool = true
    var maskThreshold: Float = 0.0
    var debugMode: Bool = false
    var active: Bool = false
    var edgeFillMode: EdgeFillMode = .chairType

    func makeUIView(context: Context) -> SmartyPantsContainerView {
        let v = SmartyPantsContainerView()
        v.processInterval = processInterval
        v.confidenceThreshold = confidenceThreshold
//        v.detectAllObjects = detectAllObjects
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
//        uiView.detectAllObjects = detectAllObjects
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

// MARK: - Union Detection Struct
struct UnionDet {
    let x: Float
    let y: Float
    let w: Float
    let h: Float
    let coeffs: [Float]
}

// MARK: - Main Container View
final class SmartyPantsContainerView: UIView, AVCaptureVideoDataOutputSampleBufferDelegate, UIGestureRecognizerDelegate {
    
    // MARK: Config
    var processInterval: TimeInterval = 0.1
    var confidenceThreshold: Float = 0.01
    var debugMode: Bool = false  // Enable debug prints and image saves
    var strongDebug: Bool = false  // Enable debug prints and image saves
    var edgeFillMode: EdgeFillMode = .chairType
    
    // Detection mode: true = detect ALL objects, false = furniture classes only
//    var detectAllObjects: Bool = true
    
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
        1099: "cot", 1183: "cradle", 3088: "playpen", 1234: "furniture"  // ← Add mystery class
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

            self.processFrameUnionCutout(pixelBuffer)
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
    
//    import Accelerate
//    import CoreML
//    import UIKit

    private func processFrameUnionCutout(_ pixelBuffer: CVPixelBuffer) {
        let frameStart = Date()

        guard let model = mlModel else { return }
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !isProcessing else { return }
        lastProcessTime = now
        isProcessing = true

        if debugMode {
            print("\n🕒 ===== NEW FRAME @ \(now.timeIntervalSince1970) =====")
            print("🔬 Single-stage: detect → union mask → cutout")
        }

        // 1) Letterbox + to MLMultiArray
        setProgress(0.25, text: "Preprocessing…")
        guard let lb = resizePixelBufferToSquare(pixelBuffer, size: 1280) else {
            isProcessing = false
            return
        }
        
        // 🔑 Bind letterbox parameters to local variables for scope access
        let letterboxGain: Float = lb.gain
        let letterboxPadX: Float = lb.padX
        let letterboxPadY: Float = lb.padY
        
        guard let inputArray = pixelBufferToMLMultiArray(lb.buffer) else {
            isProcessing = false
            return
        }

        // 2) Inference
        setProgress(0.45, text: "Running model…")
        guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
            isProcessing = false
            return
        }
        guard let output = try? model.prediction(from: inputProvider) else {
            isProcessing = false
            return
        }

        // 3) Grab detections + prototypes
        var detectionsArray: MLMultiArray?
        if let arr = output.featureValue(for: "var_2497")?.multiArrayValue {
            detectionsArray = arr
        }

        guard let detArray = detectionsArray,
              let protoArray = output.featureValue(for: "p")?.multiArrayValue
        else {
            isProcessing = false
            return
        }

        // 4) Read tensor dims
        let numFeatures = detArray.shape[1].intValue
        let numAnchors  = detArray.shape[2].intValue
        let numClasses  = numFeatures - 4 - 32
        guard numFeatures >= 36, numAnchors > 0, numClasses > 0 else {
            if debugMode { print("⚠️ Invalid det tensor dims") }
            isProcessing = false
            return
        }

        // 5) Copy detArray -> detBuf (Float32)
        let totalCount = detArray.count
        let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
        defer { detBuf.deallocate() }

        if detArray.dataType == .float32 {
            let src = detArray.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(detBuf, src, totalCount * MemoryLayout<Float>.size)
        }

        // 6) Parse prototypes -> planes + dims
        guard let protoInfo = parsePrototypes(protoArray) else {
            isProcessing = false
            return
        }
        let planes = protoInfo.planes
        let pCount = protoInfo.count
        let pH = protoInfo.height
        let pW = protoInfo.width
        guard pCount >= 32 else {
            isProcessing = false
            return
        }

        let planeSize = pH * pW

        // 7) Reorg prototypes to A: [planeSize x 32] row-major
        setProgress(0.60, text: "Building union mask…")
        var A = [Float](repeating: 0, count: planeSize * 32)
        var zero: Float = 0
        A.withUnsafeMutableBufferPointer { dstPtr in
            planes.withUnsafeBufferPointer { srcPtr in
                for k in 0..<32 {
                    let srcStart = srcPtr.baseAddress!.advanced(by: k * planeSize)
                    let dstStart = dstPtr.baseAddress!.advanced(by: k)
                    vDSP_vsadd(srcStart, 1, &zero, dstStart, 32, vDSP_Length(planeSize))
                }
            }
        }

        // 8) Select anchors that pass threshold AND store both bbox + coeffs
        let stride = numAnchors
        let coeffOffset = 4 + numClasses

        var selectedDets: [UnionDet] = []
        selectedDets.reserveCapacity(512)
        
        // FURNITURE-ONLY path — keep a tight loop over furniture indices
        let furnitureList = furnitureClasses.filter { $0.key < numClasses }

        // Debug: track best furniture confidence found
        var bestFurnitureConf: Float = 0
        var bestFurnitureClass = ""
        var bestFurnitureAnchor = -1
        var tempScores = [Float](repeating: 0, count: numClasses)

        for anchor in 0..<numAnchors {
            let x = detBuf[0 * stride + anchor]
            let y = detBuf[1 * stride + anchor]
            let w = detBuf[2 * stride + anchor]
            let h = detBuf[3 * stride + anchor]

            if !(x.isFinite && y.isFinite && w.isFinite && h.isFinite && w > 0 && h > 0) {
                continue
            }

            for (classIdx, className) in furnitureList {
                let conf = detBuf[(4 + classIdx) * stride + anchor]

                // Track best furniture confidence for debug
                if conf.isFinite && conf > bestFurnitureConf {
                    bestFurnitureConf = conf
                    bestFurnitureClass = className
                    bestFurnitureAnchor = anchor
                }

                if conf.isFinite && conf > confidenceThreshold {
                    var coeffs = [Float](repeating: 0, count: 32)
                    let coeffBase = detBuf.advanced(by: coeffOffset * stride + anchor)
                    cblas_scopy(32, coeffBase, Int32(stride), &coeffs, 1)

                    selectedDets.append(UnionDet(
                        x: x, y: y, w: w, h: h, coeffs: coeffs
                    ))
                }
            }
        }

        // Debug: find best detection across ALL classes (not just furniture) for comparison
        var bestOverallConf: Float = 0
        var bestOverallClass = -1
        var bestOverallAnchor = -1
        
        if self.debugMode && selectedDets.isEmpty {
            print("🔍 FURNITURE DEBUG: Best furniture conf=\(String(format: "%.3f", bestFurnitureConf)) for '\(bestFurnitureClass)' at anchor \(bestFurnitureAnchor)")
            print("   Threshold was: \(confidenceThreshold), furniture classes checked: \(furnitureList.count)")

            // Find best confidence across ALL classes for comparison
            for anchor in 0..<numAnchors {
                let x = detBuf[0 * stride + anchor]
                let y = detBuf[1 * stride + anchor]
                let w = detBuf[2 * stride + anchor]
                let h = detBuf[3 * stride + anchor]

                // Skip anchors with invalid geometry
                if !(x.isFinite && y.isFinite && w.isFinite && h.isFinite && w > 0 && h > 0) {
                    continue
                }

                // Copy class scores into temp buffer using BLAS
                let basePtr = detBuf.advanced(by: 4 * stride + anchor)
                cblas_scopy(Int32(numClasses), basePtr, Int32(stride), &tempScores, 1)
                
                // Find max confidence across all classes using vDSP
                var maxVal: Float = 0
                var maxIdx: vDSP_Length = 0
                vDSP_maxvi(tempScores, 1, &maxVal, &maxIdx, vDSP_Length(numClasses))
                
                if maxVal > bestOverallConf {
                    bestOverallConf = maxVal
                    bestOverallClass = Int(maxIdx)
                    bestOverallAnchor = anchor
                }
            }
            print("   Best OVERALL conf=\(String(format: "%.3f", bestOverallConf)) for class \(bestOverallClass) at anchor \(bestOverallAnchor)")
        }
        
        // Create a "best detection" for visualization if we found something but it was below threshold
        var bestDetectionForVisualization: DetectionSmarty?
        if selectedDets.isEmpty {
            // Prefer overall best if available, otherwise fall back to furniture best
            var useAnchor = -1
            var useConf: Float = 0
            var useClassName = ""
            
            if bestOverallAnchor >= 0 && bestOverallConf > 0.001 {
                useAnchor = bestOverallAnchor
                useConf = bestOverallConf
                // Look up class name from furniture classes if available, otherwise generic
                useClassName = furnitureClasses[bestOverallClass] ?? "object_\(bestOverallClass)"
            } else if bestFurnitureAnchor >= 0 && bestFurnitureConf > 0.001 {
                useAnchor = bestFurnitureAnchor
                useConf = bestFurnitureConf
                useClassName = bestFurnitureClass
            }
            
            if useAnchor >= 0 {
                let x = detBuf[0 * stride + useAnchor]
                let y = detBuf[1 * stride + useAnchor]
                let w = detBuf[2 * stride + useAnchor]
                let h = detBuf[3 * stride + useAnchor]
                
                if x.isFinite && y.isFinite && w.isFinite && h.isFinite && w > 0 && h > 0 {
                    bestDetectionForVisualization = DetectionSmarty(
                        x: x, y: y, width: w, height: h,
                        confidence: useConf,
                        classIdx: -1,  // Special flag for "best but below threshold"
                        className: useClassName,
                        maskCoeffs: [Float](repeating: 0, count: 32)
                    )
                }
            }
        }

        // Get image dimensions first (needed for both paths)
        let origW = CVPixelBufferGetWidth(pixelBuffer)
        let origH = CVPixelBufferGetHeight(pixelBuffer)
        
        if selectedDets.isEmpty {
            // If we have a best detection visualization, draw just the label without processing
            if let bestDet = bestDetectionForVisualization {
                let colorSpace = CGColorSpaceCreateDeviceRGB()
                guard let ctx = CGContext(
                    data: nil,
                    width: origW,
                    height: origH,
                    bitsPerComponent: 8,
                    bytesPerRow: origW * 4,
                    space: colorSpace,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                ) else {
                    isProcessing = false
                    return
                }
                
                // Draw best detection with special styling (red box to indicate below threshold)
                drawBestDetectionVisualization(ctx: ctx, detection: bestDet, 
                                             imageWidth: origW, imageHeight: origH,
                                             letterboxGain: letterboxGain, 
                                             letterboxPadX: letterboxPadX, letterboxPadY: letterboxPadY)
                
                if let out = ctx.makeImage() {
                    DispatchQueue.main.async {
                        self.maskImageView.image = UIImage(cgImage: out)
                        self.isProcessing = false
                    }
                } else {
                    DispatchQueue.main.async { self.isProcessing = false }
                }
                return
            }
            
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }
        
        // Step 1: Compute UNION BBOX in model space (1280x1280)
        var ux1: Float = .greatestFiniteMagnitude
        var uy1: Float = .greatestFiniteMagnitude
        var ux2: Float = -.greatestFiniteMagnitude
        var uy2: Float = -.greatestFiniteMagnitude

        for d in selectedDets {
            ux1 = min(ux1, d.x - d.w * 0.5)
            uy1 = min(uy1, d.y - d.h * 0.5)
            ux2 = max(ux2, d.x + d.w * 0.5)
            uy2 = max(uy2, d.y + d.h * 0.5)
        }
        
        // Step 2: Map UNION BBOX to original image space using letterbox params
        @inline(__always)
        func modelToOrigX(_ x: Float) -> Int {
            Int(round((x - letterboxPadX) / letterboxGain))
        }
        @inline(__always)
        func modelToOrigY(_ y: Float) -> Int {
            Int(round((y - letterboxPadY) / letterboxGain))
        }
        
        var bx1 = modelToOrigX(ux1)
        var by1 = modelToOrigY(uy1)
        var bx2 = modelToOrigX(ux2)
        var by2 = modelToOrigY(uy2)

        // Clamp to image bounds
        bx1 = max(0, min(origW - 1, bx1))
        by1 = max(0, min(origH - 1, by1))
        bx2 = max(0, min(origW, bx2))
        by2 = max(0, min(origH, by2))
        
        if debugMode {
            print("🔲 Union bbox (model space): [\(String(format: "%.1f", ux1)),\(String(format: "%.1f", uy1))] to [\(String(format: "%.1f", ux2)),\(String(format: "%.1f", uy2))]")
            print("🔲 Union bbox (image space): [\(bx1),\(by1)] to [\(bx2),\(by2)] → \((bx2-bx1))x\((by2-by1))")
        }

        // 9) UNION MASK via batched GEMM (safe RAM)
        //    Maintain maxLogits[planeSize] across batches.
        var maxLogits = [Float](repeating: -Float.greatestFiniteMagnitude, count: planeSize)

        let batchSize = 64 // tune 32/64/128
        let M = Int32(planeSize)
        let K = Int32(32)
        let alpha: Float = 1
        let beta: Float = 0

        var bStart = 0
        while bStart < selectedDets.count {
            let bEnd = min(selectedDets.count, bStart + batchSize)
            let Bn = bEnd - bStart
            let N = Int32(Bn)

            // B: [32 x Bn] row-major, ldb = N
            var B = [Float](repeating: 0, count: 32 * Bn)
            for j in 0..<Bn {
                let coeffs = selectedDets[bStart + j].coeffs
                for i in 0..<32 {
                    B[i * Bn + j] = coeffs[i]
                }
            }

            // C: [planeSize x Bn]
            var C = [Float](repeating: 0, count: planeSize * Bn)

            A.withUnsafeBufferPointer { aPtr in
                B.withUnsafeBufferPointer { bPtr in
                    C.withUnsafeMutableBufferPointer { cPtr in
                        cblas_sgemm(
                            CblasRowMajor,
                            CblasNoTrans, CblasNoTrans,
                            M, N, K,
                            alpha,
                            aPtr.baseAddress!, K,   // lda=32
                            bPtr.baseAddress!, N,   // ldb=Bn
                            beta,
                            cPtr.baseAddress!, N    // ldc=Bn
                        )
                    }
                }
            }

            // Reduce each pixel row -> update maxLogits[px]
            C.withUnsafeBufferPointer { cPtr in
                maxLogits.withUnsafeMutableBufferPointer { maxPtr in
                    for px in 0..<planeSize {
                        var localMax: Float = 0
                        vDSP_maxv(cPtr.baseAddress!.advanced(by: px * Bn), 1, &localMax, vDSP_Length(Bn))
                        if localMax > maxPtr[px] { maxPtr[px] = localMax }
                    }
                }
            }

            bStart = bEnd
        }

        // Threshold -> maskSmall Planar8
        var maskSmall = [UInt8](repeating: 0, count: planeSize)
        for i in 0..<planeSize {
            maskSmall[i] = (maxLogits[i] > 0.0) ? 255 : 0
        }

        // 10) Proper letterbox-aware upscaling using helper function
        let maskFull = makeFullMaskFromProtoWithResizeSquareFix(
            maskSmall: maskSmall,
            pW: pW,
            pH: pH,
            modelInput: 1280,
            origW: origW,
            origH: origH,
            letterboxGain: letterboxGain,
            letterboxPadX: letterboxPadX,
            letterboxPadY: letterboxPadY,
            useBilinear: self.useBilinearUpscaling
        )
        
        // ✅ SAVE FINAL MASK USED FOR CUTOUT (debug only)
        if self.debugMode {
            // Convert final mask (0/255) -> Float (0.0/1.0) so your save function can normalize it
            var rawMaskFloat = [Float](repeating: 0, count: maskFull.count)
            for i in 0..<maskFull.count {
                rawMaskFloat[i] = (maskFull[i] > 0) ? 1.0 : 0.0
            }

            // Dummy detection that covers full 1280x1280 model space (so your bbox-gating keeps everything)
            let fullDet = DetectionSmarty(
                x: 640, y: 640,
                width: 1280, height: 1280,
                confidence: 1.0,
                classIdx: 0,
                className: "UNION_MASK",
                maskCoeffs: [Float](repeating: 0, count: 32)
            )

            self.saveMaskToFile(
                rawMask: rawMaskFloat,
                width: origW,
                height: origH,
                detection: fullDet
            )
        }

        // 11) Composite ONCE: transparent bg, copy RGB where mask=1
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil,
            width: origW,
            height: origH,
            bitsPerComponent: 8,
            bytesPerRow: origW * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            isProcessing = false
            return
        }
        guard let outBase = ctx.data?.assumingMemoryBound(to: UInt8.self) else {
            isProcessing = false
            return
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let origBase = CVPixelBufferGetBaseAddress(pixelBuffer)?.assumingMemoryBound(to: UInt8.self) else {
            isProcessing = false
            return
        }
        let origBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        var totalSet = 0
        for y in 0..<origH {
            let origRow = y * origBytesPerRow
            let outRow  = y * origW * 4
            let mRow    = y * origW
            for x in 0..<origW {
                let outPx = outRow + x * 4
                
                // Step 3: Enforce transparency outside union bbox (CRITICAL)
                if x < bx1 || x >= bx2 || y < by1 || y >= by2 {
                    // ❌ Outside UNION BBOX → transparent
                    outBase[outPx + 3] = 0
                    continue
                }
                
                // Inside union bbox → apply mask
                let m = maskFull[mRow + x]
                if m > 0 {
                    let origPx = origRow + x * 4
                    outBase[outPx + 0] = origBase[origPx + 0]
                    outBase[outPx + 1] = origBase[origPx + 1]
                    outBase[outPx + 2] = origBase[origPx + 2]
                    outBase[outPx + 3] = 255
                    totalSet += 1
                } else {
                    outBase[outPx + 3] = 0
                }
            }
        }

        // 12) Draw MAIN UNION bbox (green) for live visualization
        let boxHeight = by2 - by1
        let flippedY = origH - by1 - boxHeight
        
        ctx.setStrokeColor(UIColor.green.cgColor)
        ctx.setLineWidth(4.0)
        ctx.stroke(CGRect(
            x: bx1,
            y: flippedY,
            width: bx2 - bx1,
            height: boxHeight
        ))

        // 13) UI update
        if let out = ctx.makeImage() {
            DispatchQueue.main.async {
                self.maskImageView.image = UIImage(cgImage: out)
                self.isProcessing = false
            }
        } else {
            DispatchQueue.main.async { self.isProcessing = false }
        }

        if debugMode {
            let ms = Date().timeIntervalSince(frameStart) * 1000
            print(String(format: "🕒 Frame total (furniture-only method): %.2f ms", ms))
            print("   Furniture detections: \(selectedDets.count)")
            print("   Opaque pixels: \(totalSet) (\(String(format: "%.2f", Float(totalSet)/Float(origW*origH)*100))%)")
            print("   Letterbox correction: gain=\(String(format: "%.3f", letterboxGain)), pad=(\(String(format: "%.1f", letterboxPadX)),\(String(format: "%.1f", letterboxPadY)))")
            print("   Union bbox coverage: \((bx2-bx1)*(by2-by1)) pixels (\(String(format: "%.2f", Float((bx2-bx1)*(by2-by1))/Float(origW*origH)*100))% of image)")
        }

        if totalSet > 0 {
            finishFirstDetectionIfNeeded()
        }
    }

    private func makeFullMaskFromProtoWithResizeSquareFix(
        maskSmall: [UInt8],   // size pW*pH
        pW: Int,
        pH: Int,
        modelInput: Int,      // 1280
        origW: Int,
        origH: Int,
        letterboxGain: Float,
        letterboxPadX: Float,
        letterboxPadY: Float,
        useBilinear: Bool
    ) -> [UInt8] {

        // 1) Scale proto mask -> modelInput x modelInput (1280x1280)
        var maskModel = [UInt8](repeating: 0, count: modelInput * modelInput)
        maskModel.withUnsafeMutableBufferPointer { dstPtr in
            maskSmall.withUnsafeBufferPointer { srcPtr in
                var s = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!),
                    height: vImagePixelCount(pH),
                    width:  vImagePixelCount(pW),
                    rowBytes: pW
                )
                var d = vImage_Buffer(
                    data: dstPtr.baseAddress!,
                    height: vImagePixelCount(modelInput),
                    width:  vImagePixelCount(modelInput),
                    rowBytes: modelInput
                )
                let flags: vImage_Flags = useBilinear
                    ? vImage_Flags(kvImageHighQualityResampling)
                    : vImage_Flags(kvImageNoFlags)
                _ = vImageScale_Planar8(&s, &d, nil, flags)
            }
        }

        // 2) Compute content rect in MODEL space (crop out padding)
        let contentW = Int(round(Float(origW) * letterboxGain))
        let contentH = Int(round(Float(origH) * letterboxGain))
        let x0 = Int(round(letterboxPadX))
        let y0 = Int(round(letterboxPadY))

        // Clamp defensively
        let cx0 = max(0, min(modelInput - 1, x0))
        let cy0 = max(0, min(modelInput - 1, y0))
        let cW  = max(1, min(modelInput - cx0, contentW))
        let cH  = max(1, min(modelInput - cy0, contentH))

        // 3) Copy crop into a tight buffer (cW x cH)
        var cropped = [UInt8](repeating: 0, count: cW * cH)
        for y in 0..<cH {
            let srcRow = (cy0 + y) * modelInput + cx0
            let dstRow = y * cW
            // manual copy (avoid memcpy if you prefer)
            for x in 0..<cW {
                cropped[dstRow + x] = maskModel[srcRow + x]
            }
        }

        // 4) Scale cropped mask -> original size (origW x origH)
        var maskFull = [UInt8](repeating: 0, count: origW * origH)
        maskFull.withUnsafeMutableBufferPointer { dstPtr in
            cropped.withUnsafeBufferPointer { srcPtr in
                var s = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!),
                    height: vImagePixelCount(cH),
                    width:  vImagePixelCount(cW),
                    rowBytes: cW
                )
                var d = vImage_Buffer(
                    data: dstPtr.baseAddress!,
                    height: vImagePixelCount(origH),
                    width:  vImagePixelCount(origW),
                    rowBytes: origW
                )
                let flags: vImage_Flags = useBilinear
                    ? vImage_Flags(kvImageHighQualityResampling)
                    : vImage_Flags(kvImageNoFlags)
                _ = vImageScale_Planar8(&s, &d, nil, flags)
            }
        }

        return maskFull
    }

    
    // MARK: - Pixel Buffer to MLMultiArray (BGRA -> [1,3,1280,1280] Float32)
    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width == 1280 && height == 1280 else {
            if self.debugMode { print("⚠️ pixelBufferToMLMultiArray: expected 1280x1280, got \(width)x\(height)") }
            return nil
        }
        guard let array = try? MLMultiArray(shape: [1, 3, 1280, 1280], dataType: .float32) else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        let pixelCount = width * height
        let floatSize = MemoryLayout<Float32>.size
        let planeStrideBytes = pixelCount * floatSize
        let rPtr = array.dataPointer.advanced(by: 0 * planeStrideBytes).assumingMemoryBound(to: Float32.self)
        let gPtr = array.dataPointer.advanced(by: 1 * planeStrideBytes).assumingMemoryBound(to: Float32.self)
        let bPtr = array.dataPointer.advanced(by: 2 * planeStrideBytes).assumingMemoryBound(to: Float32.self)

        let src = baseAddress.assumingMemoryBound(to: UInt8.self)
        var rowU8 = [UInt8](repeating: 0, count: width * 4)
        var rowF  = [Float](repeating: 0, count: width * 4)
        var scale: Float = 1.0 / 255.0

        var indicesR = [vDSP_Length](repeating: 0, count: width)
        var indicesG = [vDSP_Length](repeating: 0, count: width)
        var indicesB = [vDSP_Length](repeating: 0, count: width)
        for i in 0..<width {
            indicesR[i] = vDSP_Length(2 + i * 4) // BGRA -> R at +2
            indicesG[i] = vDSP_Length(1 + i * 4) // G at +1
            indicesB[i] = vDSP_Length(0 + i * 4) // B at +0
        }

        for y in 0..<height {
            let rowStart = src.advanced(by: y * bytesPerRow)
            memcpy(&rowU8, rowStart, width * 4)
            rowU8.withUnsafeBufferPointer { u8 in
                rowF.withUnsafeMutableBufferPointer { f in
                    vDSP_vfltu8(u8.baseAddress!, 1, f.baseAddress!, 1, vDSP_Length(width * 4))
                    vDSP_vsmul(f.baseAddress!, 1, &scale, f.baseAddress!, 1, vDSP_Length(width * 4))
                }
            }
            rowF.withUnsafeBufferPointer { rf in
                let baseF = rf.baseAddress!
                vDSP_vgathr(baseF, indicesR, 1, rPtr.advanced(by: y * width), 1, vDSP_Length(width))
                vDSP_vgathr(baseF, indicesG, 1, gPtr.advanced(by: y * width), 1, vDSP_Length(width))
                vDSP_vgathr(baseF, indicesB, 1, bPtr.advanced(by: y * width), 1, vDSP_Length(width))
            }
        }
        return array
    }
    
    // MARK: - Extract Detections (optimized)
    private func extractDetections(from detections: MLMultiArray, confThreshold: Float) -> [DetectionSmarty] {
        var results: [DetectionSmarty] = []

        let numFeatures = detections.shape[1].intValue
        let numAnchors = detections.shape[2].intValue
        let numClasses = numFeatures - 4 - 32
        guard numFeatures >= 36 && numAnchors > 0 && numClasses > 0 else { return [] }

        let totalCount = detections.count
        let expectedCount = numFeatures * numAnchors
        guard totalCount >= expectedCount else { return [] }

        // Copy/convert into contiguous Float buffer
        let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
        defer { detBuf.deallocate() }
//        if detections.dataType == .float16 {
//            let src = detections.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
//            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(totalCount), rowBytes: totalCount * MemoryLayout<UInt16>.size)
//            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(detBuf), height: 1, width: vImagePixelCount(totalCount), rowBytes: totalCount * MemoryLayout<Float>.size)
//            _ = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
//        } else if detections.dataType == .float32 {
            let src = detections.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(detBuf, src, totalCount * MemoryLayout<Float>.size)
//        } else {
//            for i in 0..<totalCount { detBuf[i] = detections[i].floatValue }
//        }

        let stride = numAnchors
        let coeffOffset = 4 + numClasses
        var tempScores = [Float](repeating: 0, count: numClasses)

        if true {//Kishore
            for anchor in 0..<numAnchors {
                let x = detBuf[0 * stride + anchor]
                let y = detBuf[1 * stride + anchor]
                let w = detBuf[2 * stride + anchor]
                let h = detBuf[3 * stride + anchor]
                if !(x.isFinite && y.isFinite && w.isFinite && h.isFinite && w > 0 && h > 0) { continue }

                // Copy class scores column -> tempScores
                let basePtr = detBuf.advanced(by: 4 * stride + anchor)
                cblas_scopy(Int32(numClasses), basePtr, Int32(stride), &tempScores, 1)

                var maxVal: Float = 0
                var maxIdx: vDSP_Length = 0
                vDSP_maxvi(tempScores, 1, &maxVal, &maxIdx, vDSP_Length(numClasses))
                if maxVal > confThreshold {
                    var coeffs = [Float](repeating: 0, count: 32)
                    let coeffBase = detBuf.advanced(by: coeffOffset * stride + anchor)
                    cblas_scopy(32, coeffBase, Int32(stride), &coeffs, 1)
                    let clsIdx = Int(maxIdx)
                    let className = furnitureClasses[clsIdx] ?? "object_\(clsIdx)"
                    results.append(DetectionSmarty(x: x, y: y, width: w, height: h, confidence: maxVal, classIdx: clsIdx, className: className, maskCoeffs: coeffs))
                }
            }
        } else {
            let furnitureList = furnitureClasses.filter { $0.key < numClasses }
            for anchor in 0..<numAnchors {
                let x = detBuf[0 * stride + anchor]
                let y = detBuf[1 * stride + anchor]
                let w = detBuf[2 * stride + anchor]
                let h = detBuf[3 * stride + anchor]
                if !(x.isFinite && y.isFinite && w.isFinite && h.isFinite && w > 0 && h > 0) { continue }

                for (classIdx, className) in furnitureList {
                    let conf = detBuf[(4 + classIdx) * stride + anchor]
                    if conf.isFinite && conf > confThreshold {
                        var coeffs = [Float](repeating: 0, count: 32)
                        let coeffBase = detBuf.advanced(by: coeffOffset * stride + anchor)
                        cblas_scopy(32, coeffBase, Int32(stride), &coeffs, 1)
                        results.append(DetectionSmarty(x: x, y: y, width: w, height: h, confidence: conf, classIdx: classIdx, className: className, maskCoeffs: coeffs))
                    }
                }
            }
        }
        return results
    }
    
    // Parse prototype tensor `p` into contiguous Float planes [32][H][W]
    private func parsePrototypes(_ proto: MLMultiArray) -> (planes: [Float], count: Int, height: Int, width: Int)? {
        // Normalize shape: drop leading batch dim if present
        var shape = proto.shape.map { $0.intValue }
        if shape.count == 4 && shape[0] == 1 { shape.removeFirst() }
        guard shape.count == 3 else { return nil }

        // Locate 32-channel dimension
        let cIdx: Int
        if shape[0] == 32 { cIdx = 0 }
        else if shape[2] == 32 { cIdx = 2 }
        else { cIdx = shape.firstIndex(of: 32) ?? -1 }
        guard cIdx >= 0 else { return nil }

        let count = 32
        let h: Int
        let w: Int
        if cIdx == 0 {
            h = shape[1]; w = shape[2]
        } else if cIdx == 2 {
            h = shape[0]; w = shape[1]
        } else {
            // Fallback assume [C,H,W]
            h = shape[1]; w = shape[2]
        }
        let planeSize = h * w
        let total = shape[0] * shape[1] * shape[2]

        // Copy data to Float buffer
        var rawFloats = [Float](repeating: 0, count: total)
        if proto.dataType == .float16 {
            let src = proto.dataPointer.bindMemory(to: UInt16.self, capacity: total)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: 1, width: vImagePixelCount(total), rowBytes: total * MemoryLayout<UInt16>.size)
            var dstBuf = vImage_Buffer(data: &rawFloats, height: 1, width: vImagePixelCount(total), rowBytes: total * MemoryLayout<Float>.size)
            _ = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
        } else if proto.dataType == .float32 {
            let src = proto.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(&rawFloats, src, total * MemoryLayout<Float>.size)
        } else {
            for i in 0..<total { rawFloats[i] = proto[i].floatValue }
        }

        // Reorder to [count][h][w]
        var planes = [Float](repeating: 0, count: count * planeSize)
        if cIdx == 0 {
            // Already [C,H,W]
            memcpy(&planes, rawFloats, count * planeSize * MemoryLayout<Float>.size)
        } else if cIdx == 2 {
            // [H,W,C] -> [C,H,W]
            for y in 0..<h {
                for x in 0..<w {
                    let baseHW = (y * w + x) * count
                    let dstBase = y * w + x
                    for k in 0..<count {
                        planes[k * planeSize + dstBase] = rawFloats[baseHW + k]
                    }
                }
            }
        } else {
            // Generic reorder using strides
            let strides = proto.strides.map { $0.intValue }
            for k in 0..<count {
                for y in 0..<h {
                    for x in 0..<w {
                        let idx = k * strides[0] + y * strides[1] + x * strides[2]
                        planes[k * planeSize + (y * w + x)] = rawFloats[idx]
                    }
                }
            }
        }
        return (planes, count, h, w)
    }

    private func generateCutout(
        stage1Prototypes: MLMultiArray,
        primaryBBoxes: [DetectionSmarty],
        originalImage: CVPixelBuffer,
        letterboxGain: Float,
        letterboxPadX: Float,
        letterboxPadY: Float,
        modelInput: Int = 1280
    ) {
        let functionStart = Date()
        print("\n🎭 ===== GENERATE CUTOUT START =====")
        
        let width  = CVPixelBufferGetWidth(originalImage)
        let height = CVPixelBufferGetHeight(originalImage)
        if debugMode {
            print("📐 Image dimensions: \(width) x \(height)")
            print("📦 Input params: letterboxGain=\(letterboxGain), padX=\(letterboxPadX), padY=\(letterboxPadY), modelInput=\(modelInput)")
            print("🔢 Primary bboxes count: \(primaryBBoxes.count)")
        }

        // STAGE 1: Parse prototypes
        let parseStart = Date()
        guard let protoInfo = parsePrototypes(stage1Prototypes) else {
            print("❌ STAGE 1 FAILED: Could not parse prototypes")
            DispatchQueue.main.async { self.isProcessing = false }
            return
        }
        let planes = protoInfo.planes
        let pCount = protoInfo.count
        let pH = protoInfo.height
        let pW = protoInfo.width
        guard pCount >= 32 else {
            print("❌ STAGE 1 FAILED: Proto count \(pCount) < 32")
            DispatchQueue.main.async { self.isProcessing = false }
            return
        }
        let parseEnd = Date()
        if debugMode {
            print("✅ STAGE 1 - Parse prototypes: \(String(format: "%.2f", parseEnd.timeIntervalSince(parseStart) * 1000))ms")
            print("   Proto dimensions: \(pCount) channels, \(pW) x \(pH)")
            print("   Proto plane size: \(planes.count) floats")
        }

        let planeSize = pH * pW

        // STAGE 2: Reorganize prototype planes to row-major matrix (ULTRA OPTIMIZED)
        let reorganizeStart = Date()
        var planesRM = [Float](repeating: 0, count: planeSize * 32)
        var zero: Float = 0.0  // vDSP_vsadd needs this
        
        // Use vDSP for vectorized strided copies - much faster than manual loops
        planesRM.withUnsafeMutableBufferPointer { dstPtr in
            planes.withUnsafeBufferPointer { srcPtr in
                for k in 0..<32 {
                    let srcStart = srcPtr.baseAddress!.advanced(by: k * planeSize)
                    let dstStart = dstPtr.baseAddress!.advanced(by: k)
                    
                    // Copy plane k to every 32nd position starting at k
                    // This is equivalent to: dstStart[i*32] = srcStart[i] for i in 0..<planeSize
                    vDSP_vsadd(srcStart, 1, &zero, dstStart, 32, vDSP_Length(planeSize))
                }
            }
        }
        
        let reorganizeEnd = Date()
        if debugMode {
            print("✅ STAGE 2 - Reorganize planes: \(String(format: "%.2f", reorganizeEnd.timeIntervalSince(reorganizeStart) * 1000))ms")
            print("   Created row-major matrix: \(planeSize) x 32")
        }

        // STAGE 3: Select and sort detections
        let selectStart = Date()
        let maxMasks = 100  // Reduced from 10 to 5 for better performance
        let dets = primaryBBoxes.sorted { $0.confidence > $1.confidence }
        let selected = Array(dets.prefix(maxMasks))
        let selectEnd = Date()
        if debugMode {
            print("✅ STAGE 3 - Select detections: \(String(format: "%.2f", selectEnd.timeIntervalSince(selectStart) * 1000))ms")
            print("   Selected \(selected.count)/\(primaryBBoxes.count) detections (max \(maxMasks))")
            if strongDebug {
                for (i, det) in selected.enumerated() {
                    print("   [\(i)] \(det.className) conf=\(String(format: "%.3f", det.confidence)) bbox=(\(Int(det.x)), \(Int(det.y)), \(Int(det.width))x\(Int(det.height)))")
                }
            }
        }

        // STAGE 4: Create CGContext for output
        let contextStart = Date()
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
            print("❌ STAGE 4 FAILED: Could not create CGContext")
            DispatchQueue.main.async { self.isProcessing = false }
            return
        }
        guard let outBase = ctx.data?.assumingMemoryBound(to: UInt8.self) else {
            print("❌ STAGE 4 FAILED: Could not get CGContext data pointer")
            DispatchQueue.main.async { self.isProcessing = false }
            return
        }
        // Initialize to white background for opaque cutout effect
        for i in stride(from: 0, to: width * height * 4, by: 4) {
//            outBase[i + 0] = 255  // B (Blue) = White
//            outBase[i + 1] = 255  // G (Green) = White  
//            outBase[i + 2] = 255  // R (Red) = White
            outBase[i + 3] = 0  // A (Alpha) = Opaque
        }
        let contextEnd = Date()
        
        print("✅ STAGE 4 - Create context: \(String(format: "%.2f", contextEnd.timeIntervalSince(contextStart) * 1000))ms")
        print("   Context: \(width) x \(height) RGBA, \(width * height * 4) bytes")

        if selected.isEmpty {
            print("⚠️  No detections selected - drawing empty result")
//            self.drawLabelsAndBoxes(
//                ctx: ctx,
//                stage1: primaryBBoxes,
//                stage2: [],
//                imageWidth: width,
//                imageHeight: height,
//                drawBoxes: self.debugMode,
//                letterboxGain: letterboxGain,
//                letterboxPadX: letterboxPadX,
//                letterboxPadY: letterboxPadY
//            )
            if let out = ctx.makeImage() {
                DispatchQueue.main.async {
                    self.maskImageView.image = UIImage(cgImage: out)
                    self.isProcessing = false
                }
            } else {
                DispatchQueue.main.async { self.isProcessing = false }
            }
            return
        }

        // STAGE 5: Setup work buffers and parameters
        let setupStart = Date()
        var logits = [Float](repeating: 0, count: planeSize)
        var binMaskSmall = [UInt8](repeating: 0, count: planeSize)

        // BLAS GEMV params
        let alpha: Float = 1.0
        let beta:  Float = 0.0
        let m = Int32(planeSize) // rows
        let n = Int32(32)        // cols
        let lda = Int32(32)      // row-major leading dim = cols

        // Model -> proto scaling
        let protoScaleX = Float(pW) / Float(modelInput)
        let protoScaleY = Float(pH) / Float(modelInput)

        @inline(__always)
        func modelToOrigX(_ xModel: Float) -> Float { (xModel - letterboxPadX) / letterboxGain }
        @inline(__always)
        func modelToOrigY(_ yModel: Float) -> Float { (yModel - letterboxPadY) / letterboxGain }
        
        let setupEnd = Date()
        
        print("✅ STAGE 5 - Setup buffers: \(String(format: "%.2f", setupEnd.timeIntervalSince(setupStart) * 1000))ms")
        print("   Logits buffer: \(logits.count) floats")
        print("   Binary mask buffer: \(binMaskSmall.count) bytes")
        print("   Proto scale: X=\(String(format: "%.4f", protoScaleX)), Y=\(String(format: "%.4f", protoScaleY))")
        print("   BLAS params: m=\(m), n=\(n), lda=\(lda)")

        // STAGE 6: Process each detection
        print("\n🔄 STAGE 6 - Processing detections...")
        let processStart = Date()
        var totalPixelsSet = 0
        
        // Lock original image for reading once for all detections
        CVPixelBufferLockBaseAddress(originalImage, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(originalImage, .readOnly) }
        guard let origBase = CVPixelBufferGetBaseAddress(originalImage)?.assumingMemoryBound(to: UInt8.self) else {
            print("❌ Could not access original image pixels")
            DispatchQueue.main.async { self.isProcessing = false }
            return
        }
        let origBytesPerRow = CVPixelBufferGetBytesPerRow(originalImage)
        
        for (detIndex, det) in selected.enumerated() {
            let detStart = Date()
            if strongDebug {
                print("\n  🎯 Detection \(detIndex + 1)/\(selected.count): \(det.className)")
            }
            
            // SUB-STAGE 6.1: Matrix multiplication (logits = prototypes * coeffs)
            let matmulStart = Date()
            logits.withUnsafeMutableBufferPointer { yPtr in
                det.maskCoeffs.withUnsafeBufferPointer { xPtr in
                    planesRM.withUnsafeBufferPointer { aPtr in
                        cblas_sgemv(
                            CblasRowMajor, CblasNoTrans,
                            m, n,
                            alpha,
                            aPtr.baseAddress!, lda,
                            xPtr.baseAddress!, 1,
                            beta,
                            yPtr.baseAddress!, 1
                        )
                    }
                }
            }
            let matmulEnd = Date()
            
            // Analyze logits
            var minLogit: Float = 0, maxLogit: Float = 0
            vDSP_minv(logits, 1, &minLogit, vDSP_Length(logits.count))
            vDSP_maxv(logits, 1, &maxLogit, vDSP_Length(logits.count))
            let positiveCount = logits.filter { $0 > 0 }.count
            if strongDebug {
                print("    ✅ 6.1 Matrix multiply: \(String(format: "%.2f", matmulEnd.timeIntervalSince(matmulStart) * 1000))ms")
                print("       Logits range: [\(String(format: "%.3f", minLogit)), \(String(format: "%.3f", maxLogit))], positive: \(positiveCount)/\(logits.count)")
            }

            // SUB-STAGE 6.2: Coordinate transformations
            let coordStart = Date()
            
            // bbox in model (letterboxed) coords
            let mx1 = det.x - det.width  * 0.5
            let my1 = det.y - det.height * 0.5
            let mx2 = det.x + det.width  * 0.5
            let my2 = det.y + det.height * 0.5

            // map bbox to original image pixels
            var bx1 = Int(floor(modelToOrigX(mx1)))
            var by1 = Int(floor(modelToOrigY(my1)))
            var bx2 = Int(ceil (modelToOrigX(mx2)))
            var by2 = Int(ceil (modelToOrigY(my2)))

            bx1 = max(0, min(width  - 1, bx1))
            by1 = max(0, min(height - 1, by1))
            bx2 = max(0, min(width,  bx2))
            by2 = max(0, min(height, by2))

            let bw = bx2 - bx1
            let bh = by2 - by1
            
            if bw <= 0 || bh <= 0 { 
                print("    ❌ 6.2 Invalid bbox size: \(bw) x \(bh) - skipping")
                continue 
            }

            // proto-space bbox (still model-space scaled)
            var px1 = Int(floor(mx1 * protoScaleX))
            var py1 = Int(floor(my1 * protoScaleY))
            var px2 = Int(ceil (mx2 * protoScaleX))
            var py2 = Int(ceil (my2 * protoScaleY))

            px1 = max(0, min(pW - 1, px1))
            py1 = max(0, min(pH - 1, py1))
            px2 = max(0, min(pW, px2))
            py2 = max(0, min(pH, py2))
            
            if px2 <= px1 || py2 <= py1 { 
                print("    ❌ 6.2 Invalid proto bbox: [\(px1),\(py1)] to [\(px2),\(py2)] - skipping")
                continue 
            }
            
            let coordEnd = Date()
            if strongDebug {
                print("    ✅ 6.2 Coordinates: \(String(format: "%.2f", coordEnd.timeIntervalSince(coordStart) * 1000))ms")
                print("       Model bbox: [\(String(format: "%.1f", mx1)),\(String(format: "%.1f", my1))] to [\(String(format: "%.1f", mx2)),\(String(format: "%.1f", my2))]")
                print("       Image bbox: [\(bx1),\(by1)] to [\(bx2),\(by2)] → \(bw)x\(bh)")
                print("       Proto bbox: [\(px1),\(py1)] to [\(px2),\(py2)] → \((px2-px1))x\((py2-py1))")
            }
            // SUB-STAGE 6.3: Binarize logits in proto space
            let binarizeStart = Date()
            binMaskSmall.withUnsafeMutableBytes { memset($0.baseAddress!, 0, planeSize) }

            var protoPositiveCount = 0
            for y in py1..<py2 {
                let rowBase = y * pW
                for x in px1..<px2 {
                    let idx = rowBase + x
                    if logits[idx] > 0.0 {
                        binMaskSmall[idx] = 255
                        protoPositiveCount += 1
                    }
                }
            }
            let binarizeEnd = Date()
            if strongDebug {
                print("    ✅ 6.3 Binarize: \(String(format: "%.2f", binarizeEnd.timeIntervalSince(binarizeStart) * 1000))ms")
                print("       Proto positive pixels: \(protoPositiveCount)/\((px2-px1)*(py2-py1))")
            }

            // SUB-STAGE 6.4: Upscale proto mask to image ROI
            let upscaleStart = Date()
            var binMaskROI = [UInt8](repeating: 0, count: bw * bh)
            binMaskROI.withUnsafeMutableBufferPointer { dstPtr in
                binMaskSmall.withUnsafeBufferPointer { srcPtr in
                    var s = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!),
                        height: vImagePixelCount(pH),
                        width:  vImagePixelCount(pW),
                        rowBytes: pW
                    )
                    var d = vImage_Buffer(
                        data: dstPtr.baseAddress!,
                        height: vImagePixelCount(bh),
                        width:  vImagePixelCount(bw),
                        rowBytes: bw
                    )
                    let flags: vImage_Flags = self.useBilinearUpscaling
                        ? vImage_Flags(kvImageHighQualityResampling)
                        : vImage_Flags(kvImageNoFlags)
                    _ = vImageScale_Planar8(&s, &d, nil, flags)
                }
            }
            let upscaleEnd = Date()
            
            let roiPositiveCount = binMaskROI.filter { $0 > 0 }.count
            if strongDebug {
                print("    ✅ 6.4 Upscale: \(String(format: "%.2f", upscaleEnd.timeIntervalSince(upscaleStart) * 1000))ms")
                print("       ROI positive pixels: \(roiPositiveCount)/\(bw*bh) (\(String(format: "%.1f", Float(roiPositiveCount) / Float(bw*bh) * 100))%)")
                print("       Upscale method: \(self.useBilinearUpscaling ? "bilinear" : "nearest-neighbor")")
            }
            // SUB-STAGE 6.5: Composite into output image - Copy original pixels
            let compositeStart = Date()
            var pixelsSet = 0
            
            for y in 0..<bh {
                let outRowStart = (by1 + y) * width + bx1
                let roiRowStart = y * bw
                let origRowStart = (by1 + y) * origBytesPerRow + bx1 * 4
                
                for x in 0..<bw {
                    let roiPixel = roiRowStart + x
                    if binMaskROI[roiPixel] > 0 {
                        let outIdx = outRowStart + x
                        let px = outIdx * 4
                        let origPx = origRowStart + x * 4
                        
                        // Copy original pixel colors (BGRA -> BGRA)
                        outBase[px + 0] = origBase[origPx + 0]  // B
                        outBase[px + 1] = origBase[origPx + 1]  // G  
                        outBase[px + 2] = origBase[origPx + 2]  // R
                        outBase[px + 3] = 255                   // A (fully opaque)
                        pixelsSet += 1
                    }
                }
            }
            
            totalPixelsSet += pixelsSet
            let compositeEnd = Date()
            if strongDebug {
                print("    ✅ 6.5 Composite: \(String(format: "%.2f", compositeEnd.timeIntervalSince(compositeStart) * 1000))ms")
                print("       Pixels set in output: \(pixelsSet)")
            }
            
            if strongDebug {
                let detEnd = Date()
                print("    🎯 Detection \(detIndex + 1) total: \(String(format: "%.2f", detEnd.timeIntervalSince(detStart) * 1000))ms")
            }
        }
        
        let processEnd = Date()
        if debugMode {
            print("✅ STAGE 6 - Process detections: \(String(format: "%.2f", processEnd.timeIntervalSince(processStart) * 1000))ms")
            print("   Total output pixels set: \(totalPixelsSet) (\(String(format: "%.3f", Float(totalPixelsSet) / Float(width * height) * 100))%)")
        }

        if strongDebug {
            // STAGE 7: Draw labels and boxes
            let labelsStart = Date()
            self.drawLabelsAndBoxes(
                ctx: ctx,
                stage1: selected,  // Use same selected detections instead of all primaryBBoxes
                stage2: [],
                imageWidth: width,
                imageHeight: height,
                drawBoxes: self.debugMode,
                letterboxGain: letterboxGain,
                letterboxPadX: letterboxPadX,
                letterboxPadY: letterboxPadY
            )
            
            let labelsEnd = Date()
            if debugMode {
                print("✅ STAGE 7 - Draw labels: \(String(format: "%.2f", labelsEnd.timeIntervalSince(labelsStart) * 1000))ms")
            }
        }

        // STAGE 8: Create final image and update UI
        let finalizeStart = Date()
        if let out = ctx.makeImage() {
            DispatchQueue.main.async {
                self.maskImageView.image = UIImage(cgImage: out)
                self.isProcessing = false
            }
            print("✅ STAGE 8 - Finalize: Image created and UI updated")
        } else {
            DispatchQueue.main.async { self.isProcessing = false }
            print("❌ STAGE 8 FAILED: Could not create final image")
        }
        let finalizeEnd = Date()
        
        let functionEnd = Date()
        print("🎭 ===== GENERATE CUTOUT COMPLETE =====")
        print("   Total time: \(String(format: "%.2f", functionEnd.timeIntervalSince(functionStart) * 1000))ms")
        print("   Stage breakdown:")
        print("     1. Parse: \(String(format: "%.1f", parseEnd.timeIntervalSince(parseStart) * 1000))ms")
        print("     2. Reorganize: \(String(format: "%.1f", reorganizeEnd.timeIntervalSince(reorganizeStart) * 1000))ms")
        print("     3. Select: \(String(format: "%.1f", selectEnd.timeIntervalSince(selectStart) * 1000))ms")
        print("     4. Context: \(String(format: "%.1f", contextEnd.timeIntervalSince(contextStart) * 1000))ms")
        print("     5. Setup: \(String(format: "%.1f", setupEnd.timeIntervalSince(setupStart) * 1000))ms")
        print("     6. Process: \(String(format: "%.1f", processEnd.timeIntervalSince(processStart) * 1000))ms")
//        print("     7. Labels: \(String(format: "%.1f", labelsEnd.timeIntervalSince(labelsStart) * 1000))ms")
        print("     8. Finalize: \(String(format: "%.1f", finalizeEnd.timeIntervalSince(finalizeStart) * 1000))ms")
        
        // Call finishFirstDetectionIfNeeded if this was successful
        if totalPixelsSet > 0 {
            self.finishFirstDetectionIfNeeded()
        }
    }




    // MARK: - NMS and Utility Methods
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

    
    private func resizePixelBufferToSquare(_ src: CVPixelBuffer, size: Int = 1280) -> (buffer: CVPixelBuffer, gain: Float, padX: Float, padY: Float)? {
        let t0 = Date()
        
        CVPixelBufferLockBaseAddress(src, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly) }

        let srcW = CVPixelBufferGetWidth(src)
        let srcH = CVPixelBufferGetHeight(src)

        // Calculate letterbox parameters
        let scaleW = Float(size) / Float(srcW)
        let scaleH = Float(size) / Float(srcH)
        let gain = min(scaleW, scaleH)  // Scale that fits inside square
        
        let newW = Int(Float(srcW) * gain)
        let newH = Int(Float(srcH) * gain)
        
        let padX = Float(size - newW) / 2.0
        let padY = Float(size - newH) / 2.0

        var dstOpt: CVPixelBuffer?
        let status = CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, nil, &dstOpt)
        guard status == kCVReturnSuccess, let dst = dstOpt else { return nil }

        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }

        guard let srcBase = CVPixelBufferGetBaseAddress(src),
              let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

        // Clear destination to gray (letterbox color)
        memset(dstBase, 128, size * size * 4)

        // Scale and center the image
        var srcBuffer = vImage_Buffer(data: srcBase,
                                      height: vImagePixelCount(srcH),
                                      width: vImagePixelCount(srcW),
                                      rowBytes: CVPixelBufferGetBytesPerRow(src))
        
        // Create destination buffer for scaled content
        let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)
        let dstRowBytes = CVPixelBufferGetBytesPerRow(dst)
        let offsetPtr = dstPtr.advanced(by: Int(padY) * dstRowBytes + Int(padX) * 4)
        
        var dstBuffer = vImage_Buffer(data: offsetPtr,
                                      height: vImagePixelCount(newH),
                                      width: vImagePixelCount(newW),
                                      rowBytes: dstRowBytes)

        let err = vImageScale_ARGB8888(&srcBuffer, &dstBuffer, nil, vImage_Flags(0))
        guard err == kvImageNoError else { return nil }

        if self.debugMode {
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print(String(format: "⏱ letterbox %dx%d → %dx%d: %.2f ms (gain=%.3f, pad=%.1f,%.1f)",
                         srcW, srcH, size, size, dt, gain, padX, padY))
        }

        return (buffer: dst, gain: gain, padX: padX, padY: padY)
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

    private func drawLabelsAndBoxes(
        ctx: CGContext,
        stage1: [DetectionSmarty],
        stage2: [DetectionSmarty],
        imageWidth: Int,
        imageHeight: Int,
        drawBoxes: Bool,
        letterboxGain: Float = 1.0,
        letterboxPadX: Float = 0.0,
        letterboxPadY: Float = 0.0
    ) {
        if debugMode {
            print("\n🏷️ ===== DRAW LABELS AND BOXES DEBUG =====")
            
            let allDetections = stage1 + stage2
            print("📊 Input parameters:")
            print("   stage1 count: \(stage1.count)")
            print("   stage2 count: \(stage2.count)")
            print("   total detections: \(allDetections.count)")
            print("   imageWidth: \(imageWidth), imageHeight: \(imageHeight)")
            print("   drawBoxes: \(drawBoxes)")
            print("   letterbox: gain=\(letterboxGain), padX=\(letterboxPadX), padY=\(letterboxPadY)")
        }
        
        let allDetections = stage1 + stage2
        guard !allDetections.isEmpty else { 
            if debugMode { print("⚠️  No detections to draw - returning early") }
            return 
        }
        
        // Helper functions to convert from model coordinates to image coordinates
        @inline(__always)
        func modelToImageX(_ xModel: Float) -> Float {
            return (xModel - letterboxPadX) / letterboxGain
        }
        
        @inline(__always)
        func modelToImageY(_ yModel: Float) -> Float {
            return (yModel - letterboxPadY) / letterboxGain
        }
        
        if debugMode {
            print("📐 Coordinate transformation:")
            print("   Model space: 1280x1280 (letterboxed)")
            print("   Image space: \(imageWidth)x\(imageHeight) (original)")
            print("   Transform: (modelCoord - pad) / gain")
        }
        
        // Set up text attributes
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 20, nil)
        if debugMode { print("🖋  Text setup completed - font size: 20pt") }
        
        for (index, detection) in allDetections.enumerated() {
            if debugMode && index < 10 {  // Only debug first 10 detections to avoid spam
                print("\n  🎯 Processing detection \(index + 1)/\(allDetections.count):")
                print("     Class: \(detection.className), Confidence: \(String(format: "%.3f", detection.confidence))")
                print("     Original model coords: x=\(String(format: "%.1f", detection.x)), y=\(String(format: "%.1f", detection.y)), w=\(String(format: "%.1f", detection.width)), h=\(String(format: "%.1f", detection.height))")
            }
            
            // Convert model coordinates to image coordinates using letterbox transformation
            let centerX = modelToImageX(detection.x)
            let centerY = modelToImageY(detection.y)
            let width = detection.width / letterboxGain
            let height = detection.height / letterboxGain
            
            let left = centerX - width / 2
            let top = centerY - height / 2
            let right = centerX + width / 2
            let bottom = centerY + height / 2
            
            if debugMode && index < 10 {
                print("     Transformed coords: centerX=\(String(format: "%.1f", centerX)), centerY=\(String(format: "%.1f", centerY)), w=\(String(format: "%.1f", width)), h=\(String(format: "%.1f", height))")
                print("     Final box: left=\(String(format: "%.1f", left)), top=\(String(format: "%.1f", top)), right=\(String(format: "%.1f", right)), bottom=\(String(format: "%.1f", bottom))")
                
                // Check if coordinates are reasonable
                if left < -50 || top < -50 || right > Float(imageWidth + 50) || bottom > Float(imageHeight + 50) {
                    print("     ⚠️  WARNING: Coordinates seem out of bounds!")
                    print("        Image bounds: 0x0 to \(imageWidth)x\(imageHeight)")
                    print("        Box extends: \(String(format: "%.1f", left)) to \(String(format: "%.1f", right)) (x), \(String(format: "%.1f", top)) to \(String(format: "%.1f", bottom)) (y)")
                }
            }
            
            if drawBoxes {
                if debugMode && index < 10 { print("     📦 Drawing bounding box...") }
                // Draw bounding box
                ctx.setStrokeColor(UIColor.green.cgColor)
                ctx.setLineWidth(2.0)
                let rect = CGRect(x: CGFloat(left), y: CGFloat(top),
                                width: CGFloat(width), height: CGFloat(height))
                if debugMode && index < 10 { print("        CGRect: \(rect)") }
                ctx.stroke(rect)
                if debugMode && index < 10 { print("        ✅ Bounding box drawn") }
            }
            
            // Draw label with clamped confidence
            let clampedConf = min(99, max(0, Int(detection.confidence * 100)))
            let label = "\(detection.className) \(clampedConf)%"
            if debugMode && index < 10 { print("     🏷️  Preparing label: '\(label)'") }
            
            let attributedString = NSAttributedString(string: label, attributes: [
                .font: font,
                .foregroundColor: UIColor.white
            ])
            
            let labelRect = CGRect(x: CGFloat(left), y: CGFloat(max(0, top - 25)),
                                 width: CGFloat(width), height: 25)
            if debugMode && index < 10 { print("        Label rect: \(labelRect)") }
            
            // Check label positioning
            if labelRect.minY < 0 && debugMode && index < 10 {
                print("        ⚠️  WARNING: Label rect extends above image bounds (minY: \(labelRect.minY))")
            }
            
            // Draw background for text
            if debugMode && index < 10 { print("        Drawing label background...") }
            ctx.setFillColor(UIColor.black.withAlphaComponent(0.7).cgColor)
            ctx.fill(labelRect)
            
            // Draw text
            if debugMode && index < 10 { print("        Drawing label text...") }
            let line = CTLineCreateWithAttributedString(attributedString)
            let textPosition = CGPoint(x: CGFloat(left + 5), y: CGFloat(max(20, top - 5)))
            if debugMode && index < 10 { print("        Text position: \(textPosition)") }
            ctx.textPosition = textPosition
            CTLineDraw(line, ctx)
            if debugMode && index < 10 { print("        ✅ Label drawn successfully") }
        }
        
        if debugMode {
            print("🏷️ ===== DRAW LABELS AND BOXES COMPLETE =====")
            print("   Total detections processed: \(allDetections.count)")
            print("   Letterbox transform: gain=\(letterboxGain), padX=\(letterboxPadX), padY=\(letterboxPadY)")
            print("   Target image size: \(imageWidth) x \(imageHeight)")
        }
    }
    
    // MARK: - Best Detection Visualization
    private func drawBestDetectionVisualization(
        ctx: CGContext,
        detection: DetectionSmarty,
        imageWidth: Int,
        imageHeight: Int,
        letterboxGain: Float,
        letterboxPadX: Float,
        letterboxPadY: Float
    ) {
        // Helper functions to convert from model coordinates to image coordinates
        @inline(__always)
        func modelToImageX(_ xModel: Float) -> Float {
            return (xModel - letterboxPadX) / letterboxGain
        }
        
        @inline(__always)
        func modelToImageY(_ yModel: Float) -> Float {
            return (yModel - letterboxPadY) / letterboxGain
        }
        
        // Convert model coordinates to image coordinates
        let centerX = modelToImageX(detection.x)
        let centerY = modelToImageY(detection.y)
        let width = detection.width / letterboxGain
        let height = detection.height / letterboxGain
        
        let left = centerX - width / 2
        let top = centerY - height / 2
        
        // Draw RED dashed bounding box to indicate "below threshold"
        ctx.setStrokeColor(UIColor.red.cgColor)
        ctx.setLineWidth(3.0)
        ctx.setLineDash(phase: 0, lengths: [10.0, 5.0]) // Dashed line pattern
        let rect = CGRect(x: CGFloat(left), y: CGFloat(top),
                        width: CGFloat(width), height: CGFloat(height))
        ctx.stroke(rect)
        
        // Reset line dash for text background
        ctx.setLineDash(phase: 0, lengths: [])
        
        // Draw label with special formatting
        let clampedConf = min(99, max(0, Int(detection.confidence * 100)))
        let label = "BEST: \(detection.className) \(clampedConf)% (below threshold)"
        
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 18, nil)
        let attributedString = NSAttributedString(string: label, attributes: [
            .font: font,
            .foregroundColor: UIColor.white
        ])
        
        // Calculate label size
        let textBounds = CTLineGetBoundsWithOptions(
            CTLineCreateWithAttributedString(attributedString), 
            CTLineBoundsOptions()
        )
        let labelWidth = max(200.0, textBounds.width + 20.0)
        let labelHeight: CGFloat = 30
        
        let labelRect = CGRect(
            x: CGFloat(left), 
            y: CGFloat(max(0.0, CGFloat(top) - labelHeight - 5.0)),
            width: labelWidth, 
            height: labelHeight
        )
        
        // Draw red background for "below threshold" label
        ctx.setFillColor(UIColor.red.withAlphaComponent(0.8).cgColor)
        ctx.fill(labelRect)
        
        // Draw white border around label
        ctx.setStrokeColor(UIColor.white.cgColor)
        ctx.setLineWidth(1.0)
        ctx.stroke(labelRect)
        
        // Draw text
        let line = CTLineCreateWithAttributedString(attributedString)
        let textPosition = CGPoint(x: CGFloat(left + 10), y: CGFloat(max(25, top - 10)))
        ctx.textPosition = textPosition
        CTLineDraw(line, ctx)
        
        if debugMode {
            print("📍 Drawing BEST detection visualization:")
            print("   Class: \(detection.className), Confidence: \(String(format: "%.3f", detection.confidence))")
            print("   Box: [\(String(format: "%.1f", left)),\(String(format: "%.1f", top))] size \(String(format: "%.1f", width))x\(String(format: "%.1f", height))")
            print("   Status: Below threshold (showing as red dashed box)")
        }
    }
    
    // MARK: - Remaining utility methods
}


// Add sigmoid helper function outside the class here:
fileprivate func sigmoid(_ x: inout [Float]) {
    let count = vDSP_Length(x.count)
    var zero: Float = 0
    var one: Float = 1
    // sigmoid(x) = 1 / (1 + exp(-x))
    var negX = [Float](repeating: 0, count: x.count)
    vDSP_vneg(x, 1, &negX, 1, count)
    vvexpf(&negX, negX, [Int32(x.count)])
    vDSP_vsadd(negX, 1, &one, &negX, 1, count) // 1 + exp(-x)
    vvrecf(&x, negX, [Int32(x.count)]) // 1 / (1 + exp(-x))
}


