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
    var processInterval: TimeInterval = 0.05
    var confidenceThreshold: Float = 0.01//kiss
    
    var detectAllObjects: Bool = true
    var useBilinearUpscaling: Bool = true
    var maskThreshold: Float = 0.0
    var minimumMaskPixels: Int = 10
    var debugMode: Bool = true
    var active: Bool = false
    var edgeFillMode: EdgeFillMode = .chairType

    func makeUIView(context: Context) -> SmartyPantsContainerView {
        let v = SmartyPantsContainerView()
        v.processInterval = processInterval
        v.confidenceThreshold = confidenceThreshold
//        v.detectAllObjects = detectAllObjects
        v.useBilinearUpscaling = useBilinearUpscaling
        v.maskThreshold = maskThreshold
        v.minimumMaskPixels = minimumMaskPixels
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
        uiView.minimumMaskPixels = minimumMaskPixels
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
    var processInterval: TimeInterval = 0.05
    var confidenceThreshold: Float = 0.01 //kiss
    var debugMode: Bool = true  // Enable debug prints and image saves
    var strongDebug: Bool = true  // Enable debug prints and image saves
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
    var maskThreshold: Float = 0.0 //kiss
    
    // Minimum mask pixels: filter out detections with masks smaller than this
    var minimumMaskPixels: Int = 10
    
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
    
    
    
    // MARK: - Ignored (Structure / Room / Background / Openings)
    private let clsToIgnore: [Int: String] = [
        // Walls / surfaces
        571: "wall",
        944: "city wall",
        1887: "glass wall",

        // Floor / ground
        1692: "floor",
        1758: "forest floor",
        2037: "hardwood floor",
        1881: "glass floor",

        // Ceiling / roof
        802: "ceiling",

        // Curtains / coverings
        1234: "curtain",

        // Tiles / materials
        815: "ceramic tile",

        // Rooms / interiors
        330: "bathroom",
        378: "bedroom",
        1323: "dining room",
        1518: "entrance hall",
        1080: "meeting room",
        2115: "home interior",
        2152: "hotel room",

        // Windows
        1719: "window",
        1720: "windowpane",
        1880: "glass window",
        2045: "bay window",
        2050: "skylight",

        // Doors
        531: "door",
        532: "sliding door",
        533: "glass door",
        534: "screen door",

        // Generic building / structure
        613: "building",
        615: "building facade",
        616: "building material",
        1072: "concrete",
        810: "cement",
        
        2320: "kitchen floor",
        4161: "tile",
        4162: "tile flooring",
        4163: "tile roof",
        4164: "tile wall"
    ]


    private func shouldIgnore(classId: Int) -> Bool {
        return clsToIgnore.keys.contains(classId)
    }
    
    // MARK: Furniture & Household Classes (LVIS indices)
//    private let furnitureClasses: [Int: String] = [
//        // Seating
//        132: "armchair", 276: "bar stool", 352: "beach chair", 364: "bean bag chair",
//        402: "bench", 821: "chair", 1060: "computer chair", 1602: "feeding chair",
//        1721: "folding chair", 2499: "loveseat", 2754: "music stool", 2834: "office chair",
//        2939: "park bench", 3024: "church bench", 3423: "rocking chair", 3584: "seat",
//        3888: "step stool", 3909: "stool", 4041: "swivel chair", 4473: "wheelchair",
//        4506: "window seat",
//        
//        // Beds & Bedding
//        375: "bed", 376: "bedcover", 377: "bed frame", 378: "bedsheet", 379: "bed sheet",
//        632: "bunk bed", 714: "canopy bed", 823: "daybed", 1137: "infant bed",
//        1270: "day bed", 1364: "dog bed", 2141: "hospital bed", 2599: "mattress",
//        3049: "pillow", 455: "blanket", 1047: "comforter", 1425: "duvet",
//        3625: "sheet", 3626: "sheets", 431: "bedspread", 2450: "linen",
//        
//        // Sofas & Couches
//        1141: "couch", 1816: "futon", 4331: "vanity", 2936: "ottoman", 3728: "sofa",
//        
//        // Tables
//        429: "billiard table", 1006: "cocktail table", 1061: "computer desk", 1301: "table",
//        1325: "dining table", 1503: "side table", 1885: "glass table", 2247: "island",
//        2319: "kitchen counter", 2322: "kitchen island",
//        2324: "kitchen table", 2802: "nightstand", 2836: "office desk", 3045: "picnic table",
//        3061: "table tennis table", 3145: "poker table", 3449: "round table",
//        4055: "table top", 4545: "workbench", 4564: "writing desk", 1007: "coffee table",
//        
//        // Storage
//        332: "bathroom cabinet", 517: "bookshelf", 567: "chest", 636: "bureau",
//        670: "cabinet", 977: "closet", 996: "coatrack", 1396: "drawer", 1405: "dresser",
//        1624: "file cabinet", 2318: "kitchen cabinet", 2614: "medicine cabinet",
//        3621: "shelf", 3678: "side cabinet", 3812: "spice rack", 4004: "supermarket shelf",
//        4294: "tv cabinet", 4513: "wine cabinet", 4516: "wine rack", 4433: "wardrobe",
//        
//        // Lighting
//        382: "bedside lamp", 1302: "table lamp", 1619: "floor lamp", 2383: "lamp",
//        2384: "lampshade", 732: "candle", 898: "chandelier",
//        2449: "light bulb", 2451: "light fixture", 4210: "torch", 3862: "stand",
//        
//        // Mirrors & Decor
//        334: "bathroom mirror", 2654: "mirror", 1214: "curtain", 3485: "rug",
//        3046: "picture frame", 4056: "tablecloth", 4358: "vase", 3081: "plant",
//        1750: "footrest", 749: "carpet", 1402: "drape", 1403: "drapery",
//        
//        // Electronics
//        4161: "television", 4162: "tv", 1058: "computer monitor", 1059: "computer",
//        3365: "remote control", 3802: "speaker",
//        
//        // Bathroom
//        4179: "toilet seat", 4178: "toilet", 4213: "towel bar", 4212: "towel",
//        386: "bathtub", 3635: "shower", 3636: "shower curtain", 387: "bath mat",
//        
//        // Kitchen
//        3357: "refrigerator", 2914: "oven", 2637: "microwave", 3675: "sink",
//        1350: "dishwasher", 3915: "stovetop", 1780: "freezer",
//        
//        // Misc
//        213: "baby seat", 733: "car seat", 834: "changing table", 679: "cake stand",
//        1143: "counter", 1144: "counter top", 1303: "desktop", 1733: "food stand",
//        1801: "fruit stand", 2193: "ice shelf", 2219: "information desk",
//        1099: "cot", 1183: "cradle", 3088: "playpen", 1234: "furniture"  // ← Add mystery class
//    ]
    
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
//        if debugMode { print("📵 App entered background – stopping camera & delegate") }
        // Stop delivering frames
        videoOutput.setSampleBufferDelegate(nil, queue: nil)
        stopCamera()
    }

    @objc private func handleAppDidBecomeActive() {
//        if debugMode { print("📲 App became active – restarting camera if needed") }
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
    

    
    // MARK: - Mask Pixel Count Validation
    private func validateMaskPixelCount(
        coeffs: [Float],
        A: [Float],                // precomputed [planeSize x 32] row-major
        planeSize: Int,
        bbox: (x: Float, y: Float, w: Float, h: Float),
        pW: Int, pH: Int,
        minimumPixels: Int
    ) -> Int {
        // Fast proto-space bbox gate to avoid GEMV when clearly too small
        let protoScaleX = Float(pW) / 1280.0
        let protoScaleY = Float(pH) / 1280.0
        let px1i = max(0, min(pW - 1, Int((bbox.x - bbox.w * 0.5) * protoScaleX)))
        let py1i = max(0, min(pH - 1, Int((bbox.y - bbox.h * 0.5) * protoScaleY)))
        let px2i = max(0, min(pW,     Int((bbox.x + bbox.w * 0.5) * protoScaleX)))
        let py2i = max(0, min(pH,     Int((bbox.y + bbox.h * 0.5) * protoScaleY)))
        let area = max(0, px2i - px1i) * max(0, py2i - py1i)
        if area < minimumPixels { return 0 }

        // 1) Compute mask logits using matrix multiplication (A * coeffs)
        var logits = [Float](repeating: 0, count: planeSize)
        let m = Int32(planeSize)
        let n = Int32(32)
        let lda = Int32(32)
        let alpha: Float = 1.0
        let beta: Float = 0.0

        logits.withUnsafeMutableBufferPointer { yPtr in
            coeffs.withUnsafeBufferPointer { xPtr in
                A.withUnsafeBufferPointer { aPtr in
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

        // 2) Count positive logits within proto-space bbox
        var positivePixels = 0
        for y in py1i..<py2i {
            let rowBase = y * pW
            for x in px1i..<px2i {
                let idx = rowBase + x
                if logits[idx] > 0.0 { positivePixels += 1 }
            }
        }
        return positivePixels
    }

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
        var t0 = CFAbsoluteTimeGetCurrent()
        var tLast = t0
        
        func logTime(_ label: String) {
            let now = CFAbsoluteTimeGetCurrent()
            let delta = (now - tLast) * 1000.0
            let total = (now - t0) * 1000.0
            print(String(format: "%@: %.1f ms (%.1f ms total)", label, delta, total))
            tLast = now
        }

        guard let model = mlModel else { return }
        let now = Date()
        guard now.timeIntervalSince(lastProcessTime) >= processInterval, !isProcessing else { return }
        lastProcessTime = now
        isProcessing = true

        // 1) Letterbox + to MLMultiArray
        logTime("Letterbox start")
        setProgress(0.25, text: "Preprocessing…")
        guard let lb = resizePixelBufferToSquare(pixelBuffer, size: 1280) else {
            isProcessing = false
            return
        }
        logTime("Letterbox done")
        
        // 🔑 Bind letterbox parameters to local variables for scope access
        let letterboxGain: Float = lb.gain
        let letterboxPadX: Float = lb.padX
        let letterboxPadY: Float = lb.padY
        
        logTime("MLMultiArray start")
        guard let inputArray = pixelBufferToMLMultiArray(lb.buffer) else {
            isProcessing = false
            return
        }
        logTime("MLMultiArray done")

        // 2) Inference
        logTime("Model prediction start")
        setProgress(0.45, text: "Running model…")
        guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
            isProcessing = false
            return
        }

        guard let output = try? model.prediction(from: inputProvider) else {
            isProcessing = false
            return
        }
        logTime("Model prediction done")

        // 3) Grab detections + prototypes
        logTime("Extract tensors start")
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
        logTime("Extract tensors done")

        // 4) Read tensor dims
        let numFeatures = detArray.shape[1].intValue
        let numAnchors  = detArray.shape[2].intValue
        let numClasses  = numFeatures - 4 - 32
        guard numFeatures >= 36, numAnchors > 0, numClasses > 0 else {
            isProcessing = false
            return
        }

        // 5) Copy detArray -> detBuf (Float32)
        logTime("Copy detections start")
        let totalCount = detArray.count
        let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
        defer { detBuf.deallocate() }

        if detArray.dataType == .float32 {
            let src = detArray.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(detBuf, src, totalCount * MemoryLayout<Float>.size)
        }
        logTime("Copy detections done")

        // 6) Parse prototypes -> planes + dims
        logTime("Parse prototypes start")
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
        logTime("Parse prototypes done")

        let planeSize = pH * pW

        // 7) Reorg prototypes to A: [planeSize x 32] row-major
        logTime("Matrix reorganization start")
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
        logTime("Matrix reorganization done")

        // 8) Select anchors class-agnostically (ANY object, not just furniture)
        logTime("Anchor selection start")
        let stride = numAnchors
        let coeffOffset = 4 + numClasses

        var selectedDets: [UnionDet] = []
        selectedDets.reserveCapacity(256)

        // Track best detection for debug visualization
        var bestOverallConf: Float = 0
        var bestOverallClass = -1
        var bestOverallAnchor = -1

        // First pass: collect candidate anchors by best class confidence (no GEMV yet)
        var tempScores = [Float](repeating: 0, count: numClasses)
        var candidates: [(anchor: Int, conf: Float)] = []
        candidates.reserveCapacity(1024)

        var totalAnchorsScanned = 0
        var ignoredClasses = 0
        for anchor in 0..<numAnchors {
            let x = detBuf[0 * stride + anchor]
            let y = detBuf[1 * stride + anchor]
            let w = detBuf[2 * stride + anchor]
            let h = detBuf[3 * stride + anchor]

            // Skip anchors with invalid geometry
            if !(x.isFinite && y.isFinite && w.isFinite && h.isFinite && w > 0 && h > 0) {
                continue
            }

            totalAnchorsScanned += 1

            // Copy all class scores for this anchor
            let basePtr = detBuf.advanced(by: 4 * stride + anchor)
            cblas_scopy(Int32(numClasses), basePtr, Int32(stride), &tempScores, 1)

            // Find max confidence across all classes
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(tempScores, 1, &maxVal, &maxIdx, vDSP_Length(numClasses))

            // Track best overall for debug
            if maxVal > bestOverallConf {
                bestOverallConf = maxVal
                bestOverallClass = Int(maxIdx)
                bestOverallAnchor = anchor
            }

            // Skip ignored classes entirely
            if shouldIgnore(classId: Int(maxIdx)) {
                ignoredClasses += 1
                continue
            }

            // Keep as candidate if above threshold
            if maxVal > confidenceThreshold && maxVal.isFinite {
                candidates.append((anchor: anchor, conf: maxVal))
            }
        }

        // Sort by confidence and validate top-K only
        candidates.sort { $0.conf > $1.conf }
        let maxValidate = 256  // validate at most this many with GEMV
        let toValidate = min(maxValidate, candidates.count)

        var validatedCount = 0
        var maskPixelRejected = 0
        for i in 0..<toValidate {
            let anchor = candidates[i].anchor
            let x = detBuf[0 * stride + anchor]
            let y = detBuf[1 * stride + anchor]
            let w = detBuf[2 * stride + anchor]
            let h = detBuf[3 * stride + anchor]

            // Prepare coeffs lazily only for validated anchors
            var coeffs = [Float](repeating: 0, count: 32)
            let coeffBase = detBuf.advanced(by: coeffOffset * stride + anchor)
            cblas_scopy(32, coeffBase, Int32(stride), &coeffs, 1)

            let maskPixelCount = validateMaskPixelCount(
                coeffs: coeffs,
                A: A,
                planeSize: planeSize,
                bbox: (x: x, y: y, w: w, h: h),
                pW: pW, pH: pH,
                minimumPixels: minimumMaskPixels
            )

            if maskPixelCount >= minimumMaskPixels {
                selectedDets.append(UnionDet(x: x, y: y, w: w, h: h, coeffs: coeffs))
                validatedCount += 1
            } else {
                maskPixelRejected += 1
            }
        }
        logTime("Anchor selection done")

        print("Detection stats: \(selectedDets.count) selected from \(totalAnchorsScanned) anchors (candidates=\(candidates.count), validated=\(validatedCount), ignored=\(ignoredClasses))")

        // Debug: show best detection found even if below threshold
        var bestDetectionForVisualization: DetectionSmarty?
        if self.debugMode && selectedDets.isEmpty && bestOverallAnchor >= 0 && bestOverallConf > 0.01 {
            let anchor = bestOverallAnchor
            let x = detBuf[0 * stride + anchor]
            let y = detBuf[1 * stride + anchor]
            let w = detBuf[2 * stride + anchor]
            let h = detBuf[3 * stride + anchor]
            
            if x.isFinite && y.isFinite && w.isFinite && h.isFinite && w > 0 && h > 0 {
                let className = "object_\(bestOverallClass)"
                
                if !shouldIgnore(classId: bestOverallClass) {
                    bestDetectionForVisualization = DetectionSmarty(
                        x: x, y: y, width: w, height: h,
                        confidence: bestOverallConf,
                        classIdx: -1,
                        className: className,
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
        
        // Compute UNION BBOX
        logTime("Union bbox start")
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
        
        // Map to image space
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

        bx1 = max(0, min(origW - 1, bx1))
        by1 = max(0, min(origH - 1, by1))
        bx2 = max(0, min(origW, bx2))
        by2 = max(0, min(origH, by2))
        logTime("Union bbox done")

        // Union mask computation
        logTime("Union mask start")
        let maxUnionDetections = 128
        let limitedDets: [UnionDet]
        if selectedDets.count > maxUnionDetections {
            limitedDets = selectedDets.sorted { ($0.w * $0.h) > ($1.w * $1.h) }.prefix(maxUnionDetections).map { $0 }
        } else {
            limitedDets = selectedDets
        }

        var maxLogits = [Float](repeating: -Float.greatestFiniteMagnitude, count: planeSize)
        var logitsTmp = [Float](repeating: 0, count: planeSize)
        let m = Int32(planeSize)
        let n = Int32(32)
        let lda = Int32(32)
        let alphaBLAS: Float = 1.0
        let betaBLAS: Float = 0.0

        for det in limitedDets {
            logitsTmp.withUnsafeMutableBufferPointer { yPtr in
                det.coeffs.withUnsafeBufferPointer { xPtr in
                    A.withUnsafeBufferPointer { aPtr in
                        cblas_sgemv(
                            CblasRowMajor, CblasNoTrans,
                            m, n,
                            alphaBLAS,
                            aPtr.baseAddress!, lda,
                            xPtr.baseAddress!, 1,
                            betaBLAS,
                            yPtr.baseAddress!, 1
                        )
                    }
                }
            }
            maxLogits.withUnsafeMutableBufferPointer { dst in
                logitsTmp.withUnsafeBufferPointer { src in
                    vDSP_vmax(dst.baseAddress!, 1, src.baseAddress!, 1, dst.baseAddress!, 1, vDSP_Length(planeSize))
                }
            }
        }
        logTime("Union mask done")

        // Threshold to binary mask
        logTime("Threshold mask start")
        var maskSmall = [UInt8](repeating: 0, count: planeSize)
        for i in 0..<planeSize {
            maskSmall[i] = (maxLogits[i] > 0.0) ? 255 : 0
        }
        logTime("Threshold mask done")

        // Upscale mask
        logTime("Upscale mask start")
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
        logTime("Upscale mask done")

        // Composite image
        logTime("Composite start")
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
                
                if x < bx1 || x >= bx2 || y < by1 || y >= by2 {
                    outBase[outPx + 3] = 0
                    continue
                }
                
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
        logTime("Composite done")

        // Draw bbox overlay
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

        // UI update
        logTime("UI update start")
        if let out = ctx.makeImage() {
            DispatchQueue.main.async {
                self.maskImageView.image = UIImage(cgImage: out)
                self.isProcessing = false
            }
        } else {
            DispatchQueue.main.async { self.isProcessing = false }
        }
        logTime("UI update done")

        // Final stats
        let ms = Date().timeIntervalSince(frameStart) * 1000
        print(String(format: "Frame total: %.1f ms, detections: %d, pixels: %d", ms, selectedDets.count, totalSet))

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
    
    // Add sigmoid helper function outside the class here:

    private func drawBestDetectionVisualization(
        ctx: CGContext,
        detection: DetectionSmarty,
        imageWidth: Int,
        imageHeight: Int,
        letterboxGain: Float,
        letterboxPadX: Float,
        letterboxPadY: Float
    ) {
        // Map bbox from model space (1280x1280) to original image space
        @inline(__always)
        func modelToOrigX(_ x: Float) -> Int { Int(round((x - letterboxPadX) / letterboxGain)) }
        @inline(__always)
        func modelToOrigY(_ y: Float) -> Int { Int(round((y - letterboxPadY) / letterboxGain)) }

        let mx1 = detection.x - detection.width * 0.5
        let my1 = detection.y - detection.height * 0.5
        let mx2 = detection.x + detection.width * 0.5
        let my2 = detection.y + detection.height * 0.5

        var bx1 = modelToOrigX(mx1)
        var by1 = modelToOrigY(my1)
        var bx2 = modelToOrigX(mx2)
        var by2 = modelToOrigY(my2)

        bx1 = max(0, min(imageWidth - 1, bx1))
        by1 = max(0, min(imageHeight - 1, by1))
        bx2 = max(0, min(imageWidth, bx2))
        by2 = max(0, min(imageHeight, by2))

        let boxW = max(0, bx2 - bx1)
        let boxH = max(0, by2 - by1)
        let flippedY = imageHeight - by1 - boxH

        ctx.setStrokeColor(UIColor.red.cgColor)
        ctx.setLineWidth(4.0)
        ctx.stroke(CGRect(x: bx1, y: flippedY, width: boxW, height: boxH))

        // Draw label background + text
        let confText: String
        if detection.confidence.isFinite {
            confText = String(format: "%.3f", detection.confidence)
        } else {
            confText = ""
        }
        let labelText = confText.isEmpty ? detection.className : "\(detection.className) (\(confText))"
        let attr = NSAttributedString(string: labelText, attributes: self.bboxAttributes)
        let textSize = attr.size()
        let pad: CGFloat = 6
        let bgRect = CGRect(x: CGFloat(bx1), y: CGFloat(flippedY) - textSize.height - pad * 2, width: textSize.width + pad * 2, height: textSize.height + pad * 2)

        ctx.setFillColor(UIColor.black.withAlphaComponent(0.6).cgColor)
        let path = UIBezierPath(roundedRect: bgRect, cornerRadius: 8)
        ctx.addPath(path.cgPath)
        ctx.fillPath()

        attr.draw(at: CGPoint(x: bgRect.minX + pad, y: bgRect.minY + pad))
    }
}
fileprivate func resizePixelBufferToSquare(_ pixelBuffer: CVPixelBuffer, size: Int) -> (buffer: CVPixelBuffer, gain: Float, padX: Float, padY: Float)? {
    let srcW = CVPixelBufferGetWidth(pixelBuffer)
    let srcH = CVPixelBufferGetHeight(pixelBuffer)
    guard srcW > 0, srcH > 0 else { return nil }

    let fSize = Float(size)
    let gain = min(fSize / Float(srcW), fSize / Float(srcH))
    let dstW = max(1, Int(round(Float(srcW) * gain)))
    let dstH = max(1, Int(round(Float(srcH) * gain)))
    let padX = (fSize - Float(dstW)) * 0.5
    let padY = (fSize - Float(dstH)) * 0.5

    var dstBuf: CVPixelBuffer?
    var tmpBuf: CVPixelBuffer?

    let attrs: CFDictionary = [
        kCVPixelBufferCGImageCompatibilityKey: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey: true
    ] as CFDictionary

    guard CVPixelBufferCreate(kCFAllocatorDefault, size, size, kCVPixelFormatType_32BGRA, attrs, &dstBuf) == kCVReturnSuccess,
          CVPixelBufferCreate(kCFAllocatorDefault, dstW, dstH, kCVPixelFormatType_32BGRA, attrs, &tmpBuf) == kCVReturnSuccess,
          let dst = dstBuf, let tmp = tmpBuf else {
        return nil
    }

    // Clear destination to black
    CVPixelBufferLockBaseAddress(dst, [])
    if let base = CVPixelBufferGetBaseAddress(dst) {
        memset(base, 0, CVPixelBufferGetBytesPerRow(dst) * CVPixelBufferGetHeight(dst))
    }
    CVPixelBufferUnlockBaseAddress(dst, [])

    // Scale source -> tmp
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    CVPixelBufferLockBaseAddress(tmp, [])
    defer {
        CVPixelBufferUnlockBaseAddress(tmp, [])
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
    }

    guard let sBase = CVPixelBufferGetBaseAddress(pixelBuffer),
          let dBase = CVPixelBufferGetBaseAddress(tmp) else {
        return nil
    }

    var s = vImage_Buffer(
        data: sBase,
        height: vImagePixelCount(srcH),
        width: vImagePixelCount(srcW),
        rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer)
    )
    var d = vImage_Buffer(
        data: dBase,
        height: vImagePixelCount(dstH),
        width: vImagePixelCount(dstW),
        rowBytes: CVPixelBufferGetBytesPerRow(tmp)
    )

    _ = vImageScale_ARGB8888(&s, &d, nil, vImage_Flags(kvImageHighQualityResampling))

    // Copy tmp into dst at offset (padX, padY)
    let offX = max(0, Int(round(padX)))
    let offY = max(0, Int(round(padY)))

    CVPixelBufferLockBaseAddress(dst, [])
    defer { CVPixelBufferUnlockBaseAddress(dst, []) }

    guard let dstBase = CVPixelBufferGetBaseAddress(dst) else { return nil }

    let tmpRowBytes = CVPixelBufferGetBytesPerRow(tmp)
    let dstRowBytes = CVPixelBufferGetBytesPerRow(dst)

    let srcPtr = dBase.assumingMemoryBound(to: UInt8.self)
    let dstPtr = dstBase.assumingMemoryBound(to: UInt8.self)

    for row in 0..<dstH {
        let srcOffset = row * tmpRowBytes
        let dstOffset = (offY + row) * dstRowBytes + offX * 4
        memcpy(dstPtr.advanced(by: dstOffset), srcPtr.advanced(by: srcOffset), dstW * 4)
    }

    return (buffer: dst, gain: gain, padX: Float(offX), padY: Float(offY))
}

fileprivate func parsePrototypes(_ protoArray: MLMultiArray) -> (planes: [Float], count: Int, height: Int, width: Int)? {
    guard protoArray.dataType == .float32 else { return nil }
    let shape = protoArray.shape.map { $0.intValue }

    var c = 0, h = 0, w = 0
    var baseOffset = 0

    if shape.count == 4, shape[0] == 1 {
        // [1, C, H, W]
        c = shape[1]; h = shape[2]; w = shape[3]
        baseOffset = 0
    } else if shape.count == 3 {
        // [C, H, W]
        c = shape[0]; h = shape[1]; w = shape[2]
        baseOffset = 0
    } else {
        return nil
    }

    let planeSize = h * w
    let expectedCount = c * planeSize
    var out = [Float](repeating: 0, count: expectedCount)

    // Fast path: assume row-major contiguous in [C,H,W]
    protoArray.dataPointer.withMemoryRebound(to: Float.self, capacity: protoArray.count) { ptr in
        if shape.count == 4 {
            // [1,C,H,W] laid out with planes contiguous per C
            for k in 0..<c {
                let src = ptr.advanced(by: baseOffset + k * planeSize)
                let dst = out.withUnsafeMutableBufferPointer { $0.baseAddress! }.advanced(by: k * planeSize)
                memcpy(dst, src, planeSize * MemoryLayout<Float>.size)
            }
        } else {
            // [C,H,W]
            memcpy(out.withUnsafeMutableBytes { $0.baseAddress! }, ptr, expectedCount * MemoryLayout<Float>.size)
        }
    }

    return (planes: out, count: c, height: h, width: w)
}

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

