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
    
    var detectAllObjects: Bool = false
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

        if self.debugMode {
            print("\n🕒 ===== NEW FRAME @ \(now.timeIntervalSince1970) =====")
            print("🔬 ========== STAGE 1: FULL FRAME ==========")
        }
        setProgress(0.2, text: "Preprocessing frame…")

        // STAGE 1: Preprocess
        let stage1PreStart = Date()
        guard let resized = resizePixelBufferToSquare(pixelBuffer, size: 960) else {
            isProcessing = false
            return
        }
        guard let inputArray = pixelBufferToMLMultiArray(resized) else {
            isProcessing = false
            return
        }
        let stage1PreEnd = Date()
        if self.debugMode {
            print(String(format: "⏱ Stage1 preprocess (letterbox+toMultiArray): %.2f ms", stage1PreEnd.timeIntervalSince(stage1PreStart) * 1000.0))
        }

        setProgress(0.35, text: "Running detection…")

        // STAGE 1: Inference
        let stage1InfStart = Date()
        guard let inputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": inputArray]) else {
            isProcessing = false
            return
        }
        guard let output = try? model.prediction(from: inputProvider) else {
            isProcessing = false
            return
        }
        let stage1InfEnd = Date()
        if self.debugMode {
            print(String(format: "⏱ Stage1 model.prediction: %.2f ms", stage1InfEnd.timeIntervalSince(stage1InfStart) * 1000.0))
        }

        if self.debugMode {
            let names = output.featureNames.joined(separator: ", ")
            print("📤 Model outputs: \(names)")
        }

        var detectionsArray: MLMultiArray?
        if let arr = output.featureValue(for: "var_1432")?.multiArrayValue {
            detectionsArray = arr
//        } else if let arr = output.featureValue(for: "var_2421")?.multiArrayValue {
        } else if let arr = output.featureValue(for: "var_2497")?.multiArrayValue {
            detectionsArray = arr
        } else {
            for name in output.featureNames {
                if let arr = output.featureValue(for: name)?.multiArrayValue {
                    let shape = arr.shape.map { $0.intValue }
                    if shape.count == 3 && shape[0] == 1 && shape[1] > 100 {
                        detectionsArray = arr
                        if self.debugMode { print("   → Using '\(name)' as detections: \(shape)") }
                        break
                    }
                }
            }
        }

        guard let detArray = detectionsArray else {
            isProcessing = false
            return
        }

        guard let prototypesArray = output.featureValue(for: "p")?.multiArrayValue else {
            isProcessing = false
            return
        }

        let decodeStart = Date()
        let stage1DetectionsFull = extractDetections(from: detArray, confThreshold: 0.3)
        let decodeEnd = Date()
        if self.debugMode {
            print("📊 Stage 1: \(stage1DetectionsFull.count) detections")
            print(String(format: "⏱ Stage1 detection decode: %.2f ms", decodeEnd.timeIntervalSince(decodeStart) * 1000.0))
        }

//        let sorted = stage1DetectionsFull.sorted { $0.confidence > $1.confidence }
        
        let sorted = stage1DetectionsFull.sorted {
            let area0 = ($0.width * $0.height)
            let area1 = ($1.width * $1.height)
            let score0 = area0 + $0.confidence
            let score1 = area1 + $1.confidence
            return score0 > score1
        }
        
        guard let primary = sorted.first else {
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }

        if self.debugMode {
            print("🎯 Primary: \(primary.className) @ \(Int(primary.confidence * 100))%")
            print("   BBox: center(\(Int(primary.x)), \(Int(primary.y))) size(\(Int(primary.width))x\(Int(primary.height)))")
        }

        setProgress(0.55, text: "Refining crop…")

        // STAGE 2
        if self.debugMode { print("\n🔬 ========== STAGE 2: CROPPED ==========") }

        var stage2Detections: [DetectionSmarty] = []
        var stage2Prototypes: MLMultiArray? = nil

        let stage2Start = Date()
        // Kishore
        if let croppedBuffer = cropPixelBuffer(pixelBuffer, toBBox: primary, padding: 0.0),
           let resizedCrop = resizePixelBufferToSquare(croppedBuffer, size: 960),
           let cropInputArray = pixelBufferToMLMultiArray(resizedCrop),
           let cropInputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": cropInputArray]) {

            let stage2InfStart = Date()
            if let cropOutput = try? model.prediction(from: cropInputProvider) {
                let stage2InfEnd = Date()
                if self.debugMode {
                    print(String(format: "⏱ Stage2 model.prediction: %.2f ms", stage2InfEnd.timeIntervalSince(stage2InfStart) * 1000.0))
                }
                
                var cropDetArray: MLMultiArray?
                //Kishore
//                if let arr = cropOutput.featureValue(for: "var_2421")?.multiArrayValue {
                if let arr = cropOutput.featureValue(for: "var_2497")?.multiArrayValue {
                    cropDetArray = arr
                } else {
                    for name in cropOutput.featureNames {
                        if let arr = cropOutput.featureValue(for: name)?.multiArrayValue {
                            let shape = arr.shape.map { $0.intValue }
                            if shape.count == 3 && shape[0] == 1 && shape[1] > 100 {
                                cropDetArray = arr
                                break
                            }
                        }
                    }
                }

                if let detArray = cropDetArray,
                   let protoArray = cropOutput.featureValue(for: "p")?.multiArrayValue {
                    let s2DecodeStart = Date()
                    stage2Detections = extractDetections(from: detArray, confThreshold: 0.01)
                    let s2DecodeEnd = Date()
                    stage2Prototypes = protoArray
                    if self.debugMode {
                        print("📊 Stage 2: \(stage2Detections.count) detections")
                        print(String(format: "⏱ Stage2 detection decode: %.2f ms", s2DecodeEnd.timeIntervalSince(s2DecodeStart) * 1000.0))
                    }
                }
            }
        } else {
            if self.debugMode { print("⚠️ Stage 2: Failed to crop/process") }
        }
        let stage2End = Date()
        if self.debugMode {
            print(String(format: "⏱ Stage2 total (crop+preprocess+infer+decode): %.2f ms",
                         stage2End.timeIntervalSince(stage2Start) * 1000.0))
        }
        
        let rawDetections = extractDetections(from: detArray, confThreshold: 0.3)
//        let nmsStart = Date()
//        let uniqueDetections = applyNMS(rawDetections, iouThreshold: 0.99)
//        let stage1Kept = keepOverlappingDetections(uniqueDetections)
//        let stage2Kept = stage2Prototypes != nil
//            ? applyNMS(stage2Detections, iouThreshold: 0.99)
//            : []
//        let stage2Kept = stage2Prototypes != nil ? applyNMS(stage2Detections, iouThreshold: 0.99) : []
//        let stage2Kept = stage2Prototypes != nil
//            ? keepOverlappingDetections(applyNMS(stage2Detections, iouThreshold: 0.99))
//            : []
//        let nmsEnd = Date()
//        if self.debugMode {
//            print(String(format: "⏱ NMS + keepOverlapping: %.2f ms", nmsEnd.timeIntervalSince(nmsStart) * 1000.0))
//        }

//        if self.debugMode {
//            print("\n📊 UNION SUMMARY:")
//            print("   Stage 1: keeping \(uniqueDetections.count) overlapping detections")
//            print("   Stage 2: keeping \(stage2Kept.count) overlapping detections")
//        }
        if self.debugMode {
            print("\n📊 UNION SUMMARY:")
            print("   Stage 1: keeping \(rawDetections.count) overlapping detections")
            print("   Stage 2: keeping \(stage2Detections.count) overlapping detections")
        }

//        if uniqueDetections.isEmpty && stage2Kept.isEmpty {
//            DispatchQueue.main.async {
//                self.maskImageView.image = nil
//                self.isProcessing = false
//            }
//            return
//        }
        if rawDetections.isEmpty && stage2Detections.isEmpty {
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }

        // MARK: Apply tracking to assign persistent IDs
//        let trackStart = Date()
//        let allDetections = rawDetections + stage2Detections
//        let trackedDetections = tracker.update(detections: allDetections)

        // Split back into stage1 and stage2 with track IDs
//        let trackedStage1 = Array(trackedDetections.prefix(rawDetections.count))
//        let trackedStage2 = Array(trackedDetections.suffix(stage2Detections.count))
//
//        if self.debugMode {
//            let trackEnd = Date()
//            print(String(format: "⏱ Tracker update: %.2f ms", trackEnd.timeIntervalSince(trackStart) * 1000.0))
//            let confirmedTracks = tracker.getConfirmedTracks()
//            print("🔢 Active tracks: \(confirmedTracks.count) confirmed")
//            for det in trackedDetections where det.trackId != nil {
//                print("   → ID\(det.trackId!) \(det.className)")
//            }
//        }

        setProgress(0.8, text: "Building mask…")

        let cutoutStart = Date()
        generateCutoutTwoStage(
            stage1Detections: rawDetections,
            stage1Prototypes: prototypesArray,
            stage2Detections: stage2Detections,
            stage2Prototypes: stage2Prototypes,
            primaryBBox: primary,
            originalImage: pixelBuffer
        )
        let cutoutEnd = Date()
        if self.debugMode {
            print(String(format: "⏱ generateCutoutTwoStage call: %.2f ms", cutoutEnd.timeIntervalSince(cutoutStart) * 1000.0))
            print(String(format: "🕒 Frame total (processFrame): %.2f ms", cutoutEnd.timeIntervalSince(frameStart) * 1000.0))
        }
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

    
    private func resizePixelBufferToSquare(_ src: CVPixelBuffer, size: Int = 960) -> CVPixelBuffer? {
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
        
        let scale = Float(width) / 960.0
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
    
    // MARK: - TWO-STAGE CUTOUT (with Accelerate prototype build & binary mask)
    private func generateCutoutTwoStage(
        stage1Detections: [DetectionSmarty],
        stage1Prototypes: MLMultiArray,
        stage2Detections: [DetectionSmarty],
        stage2Prototypes: MLMultiArray?,
        primaryBBox: DetectionSmarty,
        originalImage: CVPixelBuffer
    ) {
        let funcStart = Date()
        
        let shape = stage1Prototypes.shape.map { $0.intValue }
        let C = shape[1]
        let Hp = shape[2]
        let Wp = shape[3]
        let spatial = Hp * Wp

        if self.debugMode {
            print("\n🎨 Generating TWO-STAGE UNION cutout")
            print("   Stage 1: \(stage1Detections.count) detections")
            print("   Stage 2: \(stage2Detections.count) detections")
            print("📐 Prototype shape: C=\(C), H=\(Hp), W=\(Wp)")
        }

        // Stage 1 prototype buffer (Accelerate)
        let protoStage1Start = Date()
        let protoMatrix1 = makePrototypeBuffer(from: stage1Prototypes, C: C, Hp: Hp, Wp: Wp)
        let protoStage1End = Date()
        if self.debugMode {
            print(String(format: "⏱ Stage1 prototype buffer build (Accelerate): %.2f ms",
                         protoStage1End.timeIntervalSince(protoStage1Start) * 1000.0))
        }

        var globalMask = [Float](repeating: 0, count: spatial)

        if self.debugMode { print("\n🔵 Processing Stage 1 masks (full frame)...") }

        var primaryRawMask: [Float]? = nil
        var primaryDet: DetectionSmarty? = nil
        var stage1PixelCount = 0

        let s1MaskStart = Date()
        for (detIndex, det) in stage1Detections.enumerated() {
            var rawMask = [Float](repeating: 0, count: spatial)
            let mmulStart = Date()
            vDSP_mmul(det.maskCoeffs, 1,
                      protoMatrix1, 1,
                      &rawMask, 1,
                      1, vDSP_Length(spatial), vDSP_Length(C))
            let mmulEnd = Date()
            if self.debugMode {
                print(String(format: "   ⏱ vDSP_mmul Stage1 det[%d]: %.2f ms", detIndex,
                             mmulEnd.timeIntervalSince(mmulStart) * 1000.0))
            }

            if detIndex == 0 {
                primaryRawMask = rawMask
                primaryDet = det

                var minVal: Float = 0, maxVal: Float = 0
                vDSP_minv(rawMask, 1, &minVal, vDSP_Length(spatial))
                vDSP_maxv(rawMask, 1, &maxVal, vDSP_Length(spatial))
                var mean: Float = 0
                vDSP_meanv(rawMask, 1, &mean, vDSP_Length(spatial))

                print("\n📊 PRIMARY MASK RAW VALUES (\(det.className) @ \(Int(det.confidence*100))%):")
                print("   Range: min=\(minVal), max=\(maxVal), mean=\(mean)")

                var posCount = 0, negCount = 0, zeroCount = 0
                for v in rawMask {
                    if v > 0 { posCount += 1 }
                    else if v < 0 { negCount += 1 }
                    else { zeroCount += 1 }
                }
                print("   Distribution: \(posCount) positive, \(negCount) negative, \(zeroCount) zero")

                print("   Mask coefficients (32): [\(det.maskCoeffs.map { String(format: "%.6f", $0) }.joined(separator: ", "))]")

                let scale = Float(Wp) / 960.0
                let mx1 = max(0, Int((det.x - det.width / 2) * scale))
                let my1 = max(0, Int((det.y - det.height / 2) * scale))
                let mx2 = min(Wp, Int((det.x + det.width / 2) * scale))
                let my2 = min(Hp, Int((det.y + det.height / 2) * scale))

                print("   BBox in mask coords: (\(mx1),\(my1)) → (\(mx2),\(my2))")
            }

            let scale = Float(Wp) / 960.0
            let mx1 = max(0, Int((det.x - det.width / 2) * scale))
            let my1 = max(0, Int((det.y - det.height / 2) * scale))
            let mx2 = min(Wp, Int((det.x + det.width / 2) * scale))
            let my2 = min(Hp, Int((det.y + det.height / 2) * scale))

            var addedPixels = 0
            rawMask.withUnsafeBufferPointer { rPtr in
                globalMask.withUnsafeMutableBufferPointer { gPtr in
                    if mx2 > mx1 && my2 > my1 {
                        for py in my1..<my2 {
                            let rowStart = py * Wp + mx1
                            let rowLen = mx2 - mx1
                            let base = rowStart
                            for i in 0..<rowLen {
                                let idx = base + i
                                if rPtr[idx] > maskThreshold && gPtr[idx] == 0 {
                                    gPtr[idx] = 1.0
                                    addedPixels += 1
                                }
                            }
                        }
                    }
                }
            }

            if self.debugMode && detIndex < 5 {
                print("   ✅ S1 \(det.className) @ \(Int(det.confidence*100))%: bbox(\(mx1),\(my1))→(\(mx2),\(my2)), +\(addedPixels)px")
            }
        }
        let s1MaskEnd = Date()

        for i in 0..<spatial { if globalMask[i] > 0 { stage1PixelCount += 1 } }
        if self.debugMode {
            print("   ⚙️ Mask threshold: \(maskThreshold)")
            print("   📊 After Stage 1: \(stage1PixelCount)/\(spatial) pixels (\(String(format: "%.1f", Float(stage1PixelCount)/Float(spatial)*100))%)")
            print(String(format: "⏱ Stage1 mask build+apply: %.2f ms", s1MaskEnd.timeIntervalSince(s1MaskStart) * 1000.0))
        }

        if self.debugMode, let rawMask = primaryRawMask, let det = primaryDet {
            saveMaskToFile(rawMask: rawMask, width: Wp, height: Hp, detection: det)
        }

        // Stage 2 masks
        if let proto2 = stage2Prototypes, !stage2Detections.isEmpty {
            let s2ProtoStart = Date()
            if self.debugMode { print("\n🟢 Processing Stage 2 masks (cropped → full frame)...") }

            let protoMatrix2 = makePrototypeBuffer(from: proto2, C: C, Hp: Hp, Wp: Wp)
            let s2ProtoEnd = Date()
            if self.debugMode {
                print(String(format: "⏱ Stage2 prototype buffer build (Accelerate): %.2f ms",
                             s2ProtoEnd.timeIntervalSince(s2ProtoStart) * 1000.0))
            }

            let padding: Float = 0.1
            let cropX1 = max(0, primaryBBox.x - primaryBBox.width / 2 * (1 + padding))
            let cropY1 = max(0, primaryBBox.y - primaryBBox.height / 2 * (1 + padding))
            let cropX2 = min(960, primaryBBox.x + primaryBBox.width / 2 * (1 + padding))
            let cropY2 = min(960, primaryBBox.y + primaryBBox.height / 2 * (1 + padding))
            let cropW = cropX2 - cropX1
            let cropH = cropY2 - cropY1

            if self.debugMode {
                print("   Crop region (model): (\(Int(cropX1)),\(Int(cropY1)))→(\(Int(cropX2)),\(Int(cropY2))) = \(Int(cropW))x\(Int(cropH))")
            }

            let scale = Float(Wp) / 960.0

            let s2MaskStart = Date()
            for det in stage2Detections {
                var rawMask = [Float](repeating: 0, count: spatial)
                let mmulStart = Date()
                vDSP_mmul(det.maskCoeffs, 1,
                          protoMatrix2, 1,
                          &rawMask, 1,
                          1, vDSP_Length(spatial), vDSP_Length(C))
                let mmulEnd = Date()
                if self.debugMode {
                    print(String(format: "   ⏱ vDSP_mmul Stage2: %.2f ms",
                                 mmulEnd.timeIntervalSince(mmulStart) * 1000.0))
                }

                let mx1_crop = max(0, Int((det.x - det.width / 2) * scale))
                let my1_crop = max(0, Int((det.y - det.height / 2) * scale))
                let mx2_crop = min(Wp, Int((det.x + det.width / 2) * scale))
                let my2_crop = min(Hp, Int((det.y + det.height / 2) * scale))

                var addedPixels = 0

                rawMask.withUnsafeBufferPointer { rPtr in
                    globalMask.withUnsafeMutableBufferPointer { gPtr in
                        if mx2_crop > mx1_crop && my2_crop > my1_crop {
                            for py_crop in my1_crop..<my2_crop {
                                let base = py_crop * Wp
                                for px_crop in mx1_crop..<mx2_crop {
                                    let cropIdx = base + px_crop
                                    if rPtr[cropIdx] > maskThreshold {
                                        let fracX = Float(px_crop) / Float(Wp)
                                        let fracY = Float(py_crop) / Float(Hp)
                                        let fullX = cropX1 + fracX * cropW
                                        let fullY = cropY1 + fracY * cropH
                                        let mx_full = Int(fullX * scale)
                                        let my_full = Int(fullY * scale)
                                        if mx_full >= 0 && mx_full < Wp && my_full >= 0 && my_full < Hp {
                                            let fullIdx = my_full * Wp + mx_full
                                            if gPtr[fullIdx] == 0 {
                                                addedPixels += 1
                                            }
                                            gPtr[fullIdx] = 1.0
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if self.debugMode {
                    print("   ✅ S2 \(det.className) @ \(Int(det.confidence*100))%: bbox(\(mx1_crop),\(my1_crop))→(\(mx2_crop),\(my2_crop)), +\(addedPixels)px NEW")
                }
            }
            let s2MaskEnd = Date()
            if self.debugMode {
                print(String(format: "⏱ Stage2 mask build+apply: %.2f ms",
                             s2MaskEnd.timeIntervalSince(s2MaskStart) * 1000.0))
            }
        }

        var finalPixelCount = 0
        for i in 0..<spatial { if globalMask[i] > 0 { finalPixelCount += 1 } }
        let addedByStage2 = finalPixelCount - stage1PixelCount

        if self.debugMode {
            print("\n📊 MERGED MASK: \(finalPixelCount)/\(spatial) pixels (\(String(format: "%.1f", Float(finalPixelCount)/Float(spatial)*100))%)")
            print("   Stage 1 contributed: \(stage1PixelCount) pixels")
            print("   Stage 2 added: \(addedByStage2) NEW pixels")
        }

        // Accelerate: convert globalMask float (0/1) to 0/255 UInt8
        let binaryMask = makeBinaryMaskFromGlobalMask(globalMask, count: spatial)
        
        
        
        

        if self.debugMode { print20x20BinaryGrid("MERGED STAGE1+STAGE2", mask: binaryMask, width: Wp, height: Hp) }

        var minX = Wp
        var maxX = -1
        var minY = Hp
        var maxY = -1
        for y in 0..<Hp {
            let rowBase = y * Wp
            for x in 0..<Wp {
                if globalMask[rowBase + x] > 0 {
                    if x < minX { minX = x }
                    if x > maxX { maxX = x }
                    if y < minY { minY = y }
                    if y > maxY { maxY = y }
                }
            }
        }

        let maskBuildEnd = Date()
        if self.debugMode {
            print(String(format: "⏱ Mask building (Stage1+Stage2+tight bbox): %.2f ms",
                         maskBuildEnd.timeIntervalSince(funcStart) * 1000.0))
        }

        autoreleasepool {
            let renderStart = Date()
            let ciImage = CIImage(cvPixelBuffer: originalImage)
            let width = CVPixelBufferGetWidth(originalImage)
            let height = CVPixelBufferGetHeight(originalImage)

            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
                if self.debugMode { print("❌ Failed to create CGImage") }
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            guard let ctx = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                if self.debugMode { print("❌ Failed to create CGContext") }
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

            guard let data = ctx.data else {
                if self.debugMode { print("❌ CGContext has no data") }
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }

            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

            let scaleX = Float(Wp) / Float(width)
            let scaleY = Float(Hp) / Float(height)

            if self.debugMode {
                print("🖼️ Upscaling \(Wp)×\(Hp) → \(width)×\(height)")
            }

            var opaqueCount = 0

            var xMap = [Int](repeating: 0, count: width)
            for px in 0..<width { xMap[px] = min(max(Int(Float(px) * scaleX), 0), Wp - 1) }

            let keptDetections: [DetectionSmarty] = (stage1Detections + stage2Detections)

            if keptDetections.isEmpty {
                memset(data, 0, width * height * 4)
                if self.debugMode { print("📊 Output: 0/\(width * height) opaque (0.0%)") }
            } else {
                let modelSize: Float = 960.0
                var imageRects = [(x0: Int, y0: Int, x1: Int, y1: Int)]()
                imageRects.reserveCapacity(keptDetections.count)
                
                for det in keptDetections {
                    let left = det.x - det.width / 2.0
                    let right = det.x + det.width / 2.0
                    let top = det.y - det.height / 2.0
                    let bottom = det.y + det.height / 2.0
                    
                    let sx = Float(width) / modelSize
                    let sy = Float(height) / modelSize
                    
                    var ix0 = Int(floor(left * sx))
                    var ix1 = Int(ceil(right * sx))
                    var iy0 = Int(floor(top * sy))
                    var iy1 = Int(ceil(bottom * sy))
                    
                    ix0 = max(0, min(ix0, width))
                    ix1 = max(0, min(ix1, width))
                    iy0 = max(0, min(iy0, height))
                    iy1 = max(0, min(iy1, height))
                    
                    if ix0 < ix1 && iy0 < iy1 {
                        imageRects.append((x0: ix0, y0: iy0, x1: ix1, y1: iy1))
                    }
                }
                
                var rowIntervals = Array(repeating: [(start:Int,end:Int)](), count: height)
                for r in imageRects {
                    for y in r.y0..<r.y1 {
                        rowIntervals[y].append((start: r.x0, end: r.x1))
                    }
                }
                
                for y in 0..<height {
                    if rowIntervals[y].isEmpty { continue }
                    var intervals = rowIntervals[y]
                    intervals.sort { $0.start < $1.start }
                    var merged: [(Int,Int)] = []
                    var cur = intervals[0]
                    for i in 1..<intervals.count {
                        let it = intervals[i]
                        if it.start <= cur.end { cur.end = max(cur.end, it.end) } else { merged.append(cur); cur = it }
                    }
                    merged.append(cur)
                    rowIntervals[y] = merged
                }
                
                for py in 0..<height {
                    let my = min(max(Int(Float(py) * scaleY), 0), Hp - 1)
                    let maskRowOffset = my * Wp
                    let rowBase = pixels.advanced(by: py * width * 4)
                    
                    let intervals = rowIntervals[py]
                    if intervals.isEmpty {
                        memset(rowBase, 0, width * 4)
                        continue
                    }
                    
                    var x = 0
                    var intervalIndex = 0
                    
                    while x < width {
                        let nextInterval = intervalIndex < intervals.count ? intervals[intervalIndex] : (start: width, end: width)
                        if x < nextInterval.start {
                            let len = min(nextInterval.start, width) - x
                            let byteOffset = x * 4
                            memset(rowBase.advanced(by: byteOffset), 0, len * 4)
                            x += len
                            continue
                        }
                        
                        let runEnd = min(nextInterval.end, width)
                        var pxIdx = x
                        while pxIdx < runEnd {
                            let maskIdx = maskRowOffset + xMap[pxIdx]
                            let pixelPtr = rowBase.advanced(by: pxIdx * 4)
                            if globalMask[maskIdx] > 0 {
                                pixelPtr[3] = 255
                                opaqueCount += 1
                            } else {
                                pixelPtr[0] = 0; pixelPtr[1] = 0; pixelPtr[2] = 0; pixelPtr[3] = 0
                            }
                            pxIdx += 1
                        }
                        x = runEnd
                        intervalIndex += 1
                    }
                }
                
                let edgeStart = Date()
                let edges = computeMaskEdges(mask: globalMask, width: Wp, height: Hp)
                
                switch edgeFillMode {
                case .furniMaterial:
                    // Preserve fine edges using fast scanline fill between boundary crossings
                    if !edges.isEmpty {
                        let sxOutline = Float(width) / Float(Wp)
                        let syOutline = Float(height) / Float(Hp)

                        // Collect edge x-positions per full-res row (fy)
                        var rowXs = Array(repeating: [Int](), count: height)
                        for (ex, ey) in edges {
                            let fx = Int(Float(ex) * sxOutline)
                            let fy = Int(Float(ey) * syOutline)
                            guard fx >= 0, fx < width, fy >= 0, fy < height else { continue }
                            rowXs[fy].append(fx)
                        }

                        // Draw edge dots for visibility (only where alpha != 0)
                        let r: UInt8 = 255, g: UInt8 = 0, b: UInt8 = 200, a: UInt8 = 255
                        for (ex, ey) in edges {
                            let fx = Int(Float(ex) * sxOutline)
                            let fy = Int(Float(ey) * syOutline)
                            let y0 = max(0, fy - 1), y1 = min(height - 1, fy + 1)
                            let x0 = max(0, fx - 1), x1 = min(width - 1, fx + 1)
                            for yy in y0...y1 {
                                let row = pixels.advanced(by: yy * width * 4)
                                for xx in x0...x1 {
                                    let p = row.advanced(by: xx * 4)
                                    if p[3] != 0 {
                                        p[0] = b; p[1] = g; p[2] = r; p[3] = a
                                    }
                                }
                            }
                        }

                        // Fill interior between edge crossings on each scanline (even-odd rule)
                        for fy in 0..<height {
                            var xs = rowXs[fy]
                            if xs.count < 2 { continue }
                            xs.sort()
                            let row = pixels.advanced(by: fy * width * 4)
                            var i = 0
                            while i + 1 < xs.count {
                                let start = max(0, min(xs[i], width - 1))
                                let end   = max(0, min(xs[i + 1], width - 1))
                                if end > start {
                                    var xx = start
                                    while xx < end {
                                        let p = row.advanced(by: xx * 4)
                                        p[3] = 255
                                        xx += 1
                                    }
                                }
                                i += 2
                            }
                        }
                    }

                case .clothBased:
                    // Solid fill using exterior flood-fill to close interior holes
                    if !edges.isEmpty {
                        let sxOutline = Float(width) / Float(Wp)
                        let syOutline = Float(height) / Float(Hp)

                        // Find bounding box of the mask in full-res coordinates
                        var minX = width, maxX = 0, minY = height, maxY = 0
                        for (ex, ey) in edges {
                            let fx = Int(Float(ex) * sxOutline)
                            let fy = Int(Float(ey) * syOutline)
                            if fx >= 0 && fx < width && fy >= 0 && fy < height {
                                minX = min(minX, fx)
                                maxX = max(maxX, fx)
                                minY = min(minY, fy)
                                maxY = max(maxY, fy)
                            }
                        }

                        // Add padding for flood fill boundary
                        let pad = 2
                        minX = max(0, minX - pad)
                        maxX = min(width - 1, maxX + pad)
                        minY = max(0, minY - pad)
                        maxY = min(height - 1, maxY + pad)

                        let boxW = maxX - minX + 1
                        let boxH = maxY - minY + 1

                        if boxW > 4 && boxH > 4 {
                            // Create a mask of opaque pixels (from current alpha)
                            // 0 = transparent (unknown), 1 = opaque (definitely mask), 2 = exterior (flood-filled)
                            var fillMask = [UInt8](repeating: 0, count: boxW * boxH)

                            // Mark opaque pixels
                            for ly in 0..<boxH {
                                let fy = minY + ly
                                let srcRow = pixels.advanced(by: fy * width * 4)
                                for lx in 0..<boxW {
                                    let fx = minX + lx
                                    if srcRow[fx * 4 + 3] > 0 {
                                        fillMask[ly * boxW + lx] = 1
                                    }
                                }
                            }

                            // Flood fill from edges to mark exterior pixels
                            var queue = [(x: Int, y: Int)]()
                            queue.reserveCapacity(2 * (boxW + boxH))

                            // Add border pixels to queue
                            for lx in 0..<boxW {
                                if fillMask[lx] == 0 { queue.append((lx, 0)); fillMask[lx] = 2 }
                                let bottomIdx = (boxH - 1) * boxW + lx
                                if fillMask[bottomIdx] == 0 { queue.append((lx, boxH - 1)); fillMask[bottomIdx] = 2 }
                            }
                            for ly in 1..<(boxH - 1) {
                                let leftIdx = ly * boxW
                                if fillMask[leftIdx] == 0 { queue.append((0, ly)); fillMask[leftIdx] = 2 }
                                let rightIdx = ly * boxW + boxW - 1
                                if fillMask[rightIdx] == 0 { queue.append((boxW - 1, ly)); fillMask[rightIdx] = 2 }
                            }

                            // BFS flood fill
                            var qIdx = 0
                            while qIdx < queue.count {
                                let (cx, cy) = queue[qIdx]
                                qIdx += 1

                                let neighbors = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
                                for (nx, ny) in neighbors {
                                    guard nx >= 0 && nx < boxW && ny >= 0 && ny < boxH else { continue }
                                    let nIdx = ny * boxW + nx
                                    if fillMask[nIdx] == 0 {
                                        fillMask[nIdx] = 2
                                        queue.append((nx, ny))
                                    }
                                }
                            }

                            // Fill interior holes (still 0)
                            var holesFilled = 0
                            for ly in 0..<boxH {
                                let fy = minY + ly
                                let srcRow = pixels.advanced(by: fy * width * 4)
                                for lx in 0..<boxW {
                                    let idx = ly * boxW + lx
                                    if fillMask[idx] == 0 {
                                        let fx = minX + lx
                                        srcRow[fx * 4 + 3] = 255
                                        holesFilled += 1
                                    }
                                }
                            }
                            if self.debugMode {
                                print("🕳️ Hole fill: filled \(holesFilled) interior hole pixels")
                            }
                        }

                        // Draw original edge outline in magenta
                        let r: UInt8 = 255, g: UInt8 = 0, b: UInt8 = 200, a: UInt8 = 255
                        for (ex, ey) in edges {
                            let fx = Int(Float(ex) * sxOutline)
                            let fy = Int(Float(ey) * syOutline)
                            guard fx >= 1 && fx < width - 1 && fy >= 1 && fy < height - 1 else { continue }
                            for yy in (fy - 1)...(fy + 1) {
                                let row = pixels.advanced(by: yy * width * 4)
                                for xx in (fx - 1)...(fx + 1) {
                                    let p = row.advanced(by: xx * 4)
                                    if p[3] != 0 { p[0] = b; p[1] = g; p[2] = r; p[3] = a }
                                }
                            }
                        }
                    }

                case .chairType:
                    // Morphological close on alpha inside tight region to fill small gaps, preserve shape
                    if !edges.isEmpty {
                        let sxOutline = Float(width) / Float(Wp)
                        let syOutline = Float(height) / Float(Hp)

                        // Tight bounding box in full-res
                        var minX = width, maxX = 0, minY = height, maxY = 0
                        for (ex, ey) in edges {
                            let fx = Int(Float(ex) * sxOutline)
                            let fy = Int(Float(ey) * syOutline)
                            if fx >= 0 && fx < width && fy >= 0 && fy < height {
                                minX = min(minX, fx)
                                maxX = max(maxX, fx)
                                minY = min(minY, fy)
                                maxY = max(maxY, fy)
                            }
                        }
                        let pad = 0
                        minX = max(0, minX - pad)
                        maxX = min(width - 1, maxX + pad)
                        minY = max(0, minY - pad)
                        maxY = min(height - 1, maxY + pad)

                        let boxW = max(0, maxX - minX + 1)
                        let boxH = max(0, maxY - minY + 1)
                        if boxW > 2 && boxH > 2 {
                            // Extract alpha plane
                            var alpha = [UInt8](repeating: 0, count: boxW * boxH)
                            for ly in 0..<boxH {
                                let fy = minY + ly
                                let srcRow = pixels.advanced(by: fy * width * 4)
                                for lx in 0..<boxW {
                                    let fx = minX + lx
                                    alpha[ly * boxW + lx] = srcRow[fx * 4 + 3]
                                }
                            }

                            // vImage morphological closing (3x3 kernel)
                            var k: [UInt8] = [1,1,1, 1,1,1, 1,1,1]
                            alpha.withUnsafeMutableBytes { aPtr in
                                var src = vImage_Buffer(data: aPtr.baseAddress!, height: vImagePixelCount(boxH), width: vImagePixelCount(boxW), rowBytes: boxW)
                                var tmp = vImage_Buffer(data: malloc(boxW * boxH), height: vImagePixelCount(boxH), width: vImagePixelCount(boxW), rowBytes: boxW)
                                var dst = vImage_Buffer(data: malloc(boxW * boxH), height: vImagePixelCount(boxH), width: vImagePixelCount(boxW), rowBytes: boxW)
                                defer { free(tmp.data); free(dst.data) }
                                k.withUnsafeMutableBufferPointer { kPtr in
                                    vImageDilate_Planar8(&src, &tmp, 0, 0, kPtr.baseAddress!, 3, 3, vImage_Flags(kvImageNoFlags))
                                    vImageErode_Planar8(&tmp, &dst, 0, 0, kPtr.baseAddress!, 3, 3, vImage_Flags(kvImageNoFlags))
                                }
                                // Copy back to alpha
                                memcpy(aPtr.baseAddress!, dst.data, boxW * boxH)
                            }

                            // Write back alpha to pixels
                            for ly in 0..<boxH {
                                let fy = minY + ly
                                let dstRow = pixels.advanced(by: fy * width * 4)
                                for lx in 0..<boxW {
                                    let fx = minX + lx
                                    dstRow[fx * 4 + 3] = alpha[ly * boxW + lx]
                                }
                            }
                        }

                        // Draw original edge outline in magenta for visibility
                        let r: UInt8 = 255, g: UInt8 = 0, b: UInt8 = 200, a: UInt8 = 255
                        for (ex, ey) in edges {
                            let fx = Int(Float(ex) * sxOutline)
                            let fy = Int(Float(ey) * syOutline)
                            guard fx >= 1 && fx < width - 1 && fy >= 1 && fy < height - 1 else { continue }
                            for yy in (fy - 1)...(fy + 1) {
                                let row = pixels.advanced(by: yy * width * 4)
                                for xx in (fx - 1)...(fx + 1) {
                                    let p = row.advanced(by: xx * 4)
                                    if p[3] != 0 { p[0] = b; p[1] = g; p[2] = r; p[3] = a }
                                }
                            }
                        }
                    }
                }
                
            }
            // ✅ Labels (always), boxes only in debugMode
            self.drawLabelsAndBoxes(
                ctx: ctx,
                stage1: [primaryBBox],
                stage2: stage2Detections,
                imageWidth: width,
                imageHeight: height,
                drawBoxes: self.debugMode
            )

            let renderEnd = Date()
            if self.debugMode {
                print(String(format: "⏱ Rendering + upscaling + cutout: %.2f ms",
                             renderEnd.timeIntervalSince(renderStart) * 1000.0))
                print(String(format: "⏱ generateCutoutTwoStage total: %.2f ms",
                             renderEnd.timeIntervalSince(funcStart) * 1000.0))
            }

            if let outImage = ctx.makeImage() {
                DispatchQueue.main.async {
                    self.maskImageView.image = UIImage(cgImage: outImage, scale: 1.0, orientation: .up)
                    self.isProcessing = false
                    if self.debugMode { print("✅ ==================== FRAME COMPLETE ====================\n") }
                }
            } else {
                if self.debugMode { print("❌ Failed to make output image") }
                DispatchQueue.main.async { self.isProcessing = false }
            }
        }
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
        let modelSize: CGFloat = 960.0
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

        // YOLOE outputs are scaled to 960×960 model input
        let sx = CGFloat(imageWidth) / 960.0
        let sy = CGFloat(imageHeight) / 960.0

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
        let modelSize: CGFloat = 960.0
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


    // MARK: - Extract Detections (with timing)
    // Kishore
    private func extractDetections(from detections: MLMultiArray, confThreshold: Float) -> [DetectionSmarty] {
        let t0 = Date()
        var all: [DetectionSmarty] = []

        let numFeatures = detections.shape[1].intValue
        let numAnchors = detections.shape[2].intValue
        let numClasses = numFeatures - 4 - 32

        // Validate tensor dimensions
        guard numFeatures >= 36 && numAnchors > 0 && numClasses > 0 else {
            if self.debugMode {
                print("⚠️ extractDetections: Invalid tensor dimensions - features:\(numFeatures), anchors:\(numAnchors), classes:\(numClasses)")
            }
            return []
        }

        if self.debugMode {
            print("🔍 Tensor shape: [1, \(numFeatures), \(numAnchors)]")
            print("   → \(numClasses) classes, \(numAnchors) predictions")
            print("   → Mode: \(detectAllObjects ? "ALL OBJECTS" : "FURNITURE ONLY")")
            if numClasses == 4585 {
                print("   → Model: YOLOE (LVIS open-vocabulary)")
            } else if numClasses == 80 {
                print("   → Model: YOLO11-seg (COCO)")
            }
        }

        let totalCount = detections.count
        
        // Validate total count matches expected dimensions
        let expectedCount = numFeatures * numAnchors
        guard totalCount >= expectedCount else {
            if self.debugMode {
                print("⚠️ extractDetections: Array size mismatch - expected:\(expectedCount), got:\(totalCount)")
            }
            return []
        }

        let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
        defer { detBuf.deallocate() }

        let copyStart = Date()
        if detections.dataType == .float16 {
            let src = detections.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src),
                                       height: 1, width: vImagePixelCount(totalCount),
                                       rowBytes: totalCount * MemoryLayout<UInt16>.size)
            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(detBuf),
                                       height: 1, width: vImagePixelCount(totalCount),
                                       rowBytes: totalCount * MemoryLayout<Float>.size)
            let result = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
            if result != kvImageNoError && self.debugMode {
                print("⚠️ extractDetections: vImage conversion failed: \(result)")
            }
        } else if detections.dataType == .float32 {
            let src = detections.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(detBuf, src, totalCount * MemoryLayout<Float>.size)
        } else {
            for i in 0..<totalCount {
                detBuf[i] = detections[i].floatValue
            }
        }
        let copyEnd = Date()
        if self.debugMode {
            print(String(format: "⏱ extractDetections copy/convert: %.2f ms",
                         copyEnd.timeIntervalSince(copyStart) * 1000.0))
        }

        let coeffOffset = 4 + numClasses
        let stride = numAnchors

        let decodeStart = Date()
        if detectAllObjects {
            for anchor in 0..<numAnchors {
                // Bounds checking for coordinate access
                guard anchor < stride,
                      1 * stride + anchor < totalCount,
                      2 * stride + anchor < totalCount,
                      3 * stride + anchor < totalCount else {
                    if self.debugMode { print("⚠️ Coordinate bounds check failed for anchor \(anchor)") }
                    continue
                }
                
                let x = detBuf[0 * stride + anchor]
                let y = detBuf[1 * stride + anchor]
                let w = detBuf[2 * stride + anchor]
                let h = detBuf[3 * stride + anchor]
                
                // Validate coordinate values
                guard x.isFinite && y.isFinite && w.isFinite && h.isFinite && w > 0 && h > 0 else {
                    continue
                }

                var bestConf: Float = 0
                var bestClassIdx = -1

                let baseConfIdx = 4 * stride + anchor
                for classIdx in 0..<numClasses {
                    let confIndex = baseConfIdx + classIdx * stride
                    guard confIndex < totalCount else {
                        if self.debugMode { print("⚠️ Confidence bounds check failed for class \(classIdx), anchor \(anchor)") }
                        break
                    }
                    
                    let conf = detBuf[confIndex]
                    if conf > bestConf && conf.isFinite {
                        bestConf = conf
                        bestClassIdx = classIdx
                    }
                }

                if bestConf > confidenceThreshold && bestClassIdx >= 0 {
                    var coeffs = [Float](repeating: 0, count: 32)
                    let coeffStart = coeffOffset * stride + anchor
                    
                    // Bounds checking for mask coefficients
                    var validCoeffs = true
                    for k in 0..<32 {
                        let coeffIndex = coeffStart + k * stride
                        if coeffIndex < totalCount {
                            coeffs[k] = detBuf[coeffIndex]
                        } else {
                            if self.debugMode { print("⚠️ Coefficient bounds check failed for k=\(k), anchor=\(anchor)") }
                            validCoeffs = false
                            break
                        }
                    }
                    
                    if validCoeffs {
                        let className = furnitureClasses[bestClassIdx] ?? "object_\(bestClassIdx)"
                        all.append(DetectionSmarty(
                            x: x, y: y, width: w, height: h,
                            confidence: bestConf, classIdx: bestClassIdx, className: className,
                            maskCoeffs: coeffs
                        ))
                    }
                }
            }
        } else {
            let furnitureList = furnitureClasses.filter { $0.key < numClasses }

            // Debug: track best furniture confidence found
            var bestFurnitureConf: Float = 0
            var bestFurnitureClass = ""
            var bestFurnitureAnchor = -1

            for anchor in 0..<numAnchors {
                // Bounds checking for coordinate access
                guard anchor < stride,
                      1 * stride + anchor < totalCount,
                      2 * stride + anchor < totalCount,
                      3 * stride + anchor < totalCount else {
                    if self.debugMode { print("⚠️ Coordinate bounds check failed for anchor \(anchor)") }
                    continue
                }

                let x = detBuf[0 * stride + anchor]
                let y = detBuf[1 * stride + anchor]
                let w = detBuf[2 * stride + anchor]
                let h = detBuf[3 * stride + anchor]

                // Validate coordinate values
                guard x.isFinite && y.isFinite && w.isFinite && h.isFinite && w > 0 && h > 0 else {
                    continue
                }

                for (classIdx, className) in furnitureList {
                    let confIdx = (4 + classIdx) * stride + anchor
                    guard confIdx < totalCount else {
                        if self.debugMode { print("⚠️ Furniture confidence bounds check failed for class \(classIdx), anchor \(anchor)") }
                        continue
                    }

                    let conf = detBuf[confIdx]

                    // Track best furniture confidence for debug
                    if conf > bestFurnitureConf && conf.isFinite {
                        bestFurnitureConf = conf
                        bestFurnitureClass = className
                        bestFurnitureAnchor = anchor
                    }

                    if conf > confidenceThreshold && conf.isFinite {
                        var coeffs = [Float](repeating: 0, count: 32)
                        let coeffStart = coeffOffset * stride + anchor

                        // Bounds checking for mask coefficients
                        var validCoeffs = true
                        for k in 0..<32 {
                            let coeffIndex = coeffStart + k * stride
                            if coeffIndex < totalCount {
                                coeffs[k] = detBuf[coeffIndex]
                            } else {
                                if self.debugMode { print("⚠️ Coefficient bounds check failed for k=\(k), anchor=\(anchor)") }
                                validCoeffs = false
                                break
                            }
                        }

                        if validCoeffs {
                            all.append(DetectionSmarty(
                                x: x, y: y, width: w, height: h,
                                confidence: conf, classIdx: classIdx, className: className,
                                maskCoeffs: coeffs
                            ))
                        }
                    }
                }
            }

            // Debug: show best furniture confidence found even if below threshold
            if self.debugMode && all.isEmpty {
                print("🔍 FURNITURE DEBUG: Best conf=\(String(format: "%.3f", bestFurnitureConf)) for '\(bestFurnitureClass)' at anchor \(bestFurnitureAnchor)")
                print("   Threshold was: \(confidenceThreshold), furniture classes checked: \(furnitureList.count)")

                // Also find best confidence across ALL classes for comparison
                var bestOverallConf: Float = 0
                var bestOverallClass = -1
                for anchor in 0..<min(100, numAnchors) {  // Check first 100 anchors
                    for classIdx in 0..<numClasses {
                        let confIdx = (4 + classIdx) * stride + anchor
                        if confIdx < totalCount {
                            let conf = detBuf[confIdx]
                            if conf > bestOverallConf && conf.isFinite {
                                bestOverallConf = conf
                                bestOverallClass = classIdx
                            }
                        }
                    }
                }
                print("   Best OVERALL conf=\(String(format: "%.3f", bestOverallConf)) for class \(bestOverallClass)")
            }
        }
        let decodeEnd = Date()

        if self.debugMode {
            print(String(format: "⏱ extractDetections decode loop: %.2f ms",
                         decodeEnd.timeIntervalSince(decodeStart) * 1000.0))

            let grouped = Dictionary(grouping: all) { $0.className }
            print("\n📊 DETECTION SUMMARY: \(all.count) total")
            for (className, dets) in grouped.sorted(by: { $0.value.count > $1.value.count }).prefix(20) {
                let confidences = dets.map { Int($0.confidence * 100) }
                print("  - \(className): \(dets.count)x, conf: \(confidences)%")
            }
            if grouped.count > 20 {
                print("  ... and \(grouped.count - 20) more classes")
            }
            let tEnd = Date()
            print(String(format: "⏱ extractDetections total: %.2f ms",
                         tEnd.timeIntervalSince(t0) * 1000.0))
        }

        return all
    }


    // MARK: - Pixel Buffer to MLMultiArray (Accelerate) — with timing
    private func pixelBufferToMLMultiArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
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
