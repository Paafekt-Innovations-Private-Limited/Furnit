import SwiftUI
import UIKit
import AVFoundation
import CoreML
import CoreImage
import Accelerate

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

    func makeUIView(context: Context) -> SmartyPantsContainerView {
        let v = SmartyPantsContainerView()
        v.processInterval = processInterval
        v.confidenceThreshold = confidenceThreshold
        v.detectAllObjects = detectAllObjects
        v.useBilinearUpscaling = useBilinearUpscaling
        v.maskThreshold = maskThreshold
        v.debugMode = debugMode
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
        if active { uiView.startIfNeeded() } else { uiView.stop() }
    }

    static func dismantleUIView(_ uiView: SmartyPantsContainerView, coordinator: ()) {
        uiView.stop()
    }
}

// MARK: - Detection Model

struct DetectionSmarty {
    var x: Float
    var y: Float
    var width: Float
    var height: Float
    var confidence: Float
    var classIdx: Int
    var className: String
    var maskCoeffs: [Float]
}

// MARK: - Container View

final class SmartyPantsContainerView: UIView,
                                      UIGestureRecognizerDelegate,
                                      AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // MARK: - Public knobs (set from SwiftUI wrapper)
    var processInterval: TimeInterval = 0.1
    var confidenceThreshold: Float = 0.5
    var detectAllObjects: Bool = false
    var useBilinearUpscaling: Bool = true
    var maskThreshold: Float = 0.0
    var debugMode: Bool = true
    var active: Bool = false
    
    // MARK: - CoreML
    private var mlModel: MLModel?
    private let ciContext = CIContext(options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
    
    // MARK: Brightness gate (prevent processing when phone is lying down / frame is dark)
    private var lumaThreshold: Float = 0.08          // 0.0 .. 1.0
    private var brightStreak: Int = 0
    private var requiredBrightStreak: Int = 3         // require a few bright frames before resuming
    private var isDarkGateActive: Bool = false
    
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
    
    // MARK: - Camera
    private let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sampleQueue = DispatchQueue(label: "com.furnit.smarty.sample", qos: .userInitiated)
    private let previewLayer = AVCaptureVideoPreviewLayer()
    
    // MARK: - Detection / processing
    private let detectionQueue = DispatchQueue(label: "com.furnit.smarty.detection", qos: .userInitiated)
    private var lastProcessTime: Date = .distantPast
    private var isProcessing: Bool = false
    
    // MARK: - Frame counting (for perimeter system)
    private var frameIndex: Int = 0
    private var maskMemoryAge: Int = 0
    private let maskMemoryMaxAge: Int = 5  // Reduced from 10 to be more responsive
    
    // MARK: - Perimeter "lock" of the best union mask we've seen so far
    // We only store the perimeter (edge) as a thin binary mask, not the full area.
    private var bestPerimeterMask: [UInt8]? = nil
    private var bestPerimeterArea: Int = 0

    
    // MARK: - App state
    private var isAppActive: Bool = true
    
    // MARK: - UI
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
    
    // simple bbox label style
    private let bboxFont: CTFont = CTFontCreateWithName("Helvetica-Bold" as CFString, 28, nil)
    private lazy var bboxAttributes: [NSAttributedString.Key: Any] = [
        .font: bboxFont,
        .foregroundColor: UIColor.white
    ]
    
    // MARK: - Init
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
        
        // Gestures
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
        installAppStateObservers()
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
    
    @objc private func appDidBecomeActive() {
        isAppActive = true
    }

    @objc private func appWillResignActive() {
        isAppActive = false
        // Also stop any "in-flight" processing quickly.
        detectionQueue.async { [weak self] in
            self?.isProcessing = false
        }
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
    
    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
    }
    
    // MARK: - Perimeter helpers

    /// Count number of 1s in a binary mask.
    private func area(of mask: [UInt8]) -> Int {
        var count = 0
        for v in mask where v != 0 {
            count += 1
        }
        return count
    }

    /// Overlap = intersection(previous, candidate) / area(previous)
    /// (i.e. “how much of the old edge is still present in the new mask?”)
    private func overlapFraction(previous: [UInt8], candidate: [UInt8]) -> Float {
        guard previous.count == candidate.count else { return 0 }

        var intersection = 0
        var prevArea = 0

        for i in 0..<previous.count {
            if previous[i] != 0 {
                prevArea += 1
                if candidate[i] != 0 {
                    intersection += 1
                }
            }
        }

        if prevArea == 0 { return 0 }
        return Float(intersection) / Float(prevArea)
    }

    /// Extract a 1-pixel border from a binary mask (4-connected perimeter).
    /// Input mask is width×height, flattened row-major.
    private func extractPerimeter(from mask: [UInt8], width: Int, height: Int) -> [UInt8] {
        let count = mask.count
        guard width > 0, height > 0, count == width * height else {
            return [UInt8](repeating: 0, count: count)
        }

        var edge = [UInt8](repeating: 0, count: count)

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                if mask[idx] == 0 { continue }

                var isBorder = false

                // Image boundary always counts as border
                if x == 0 || x == width - 1 || y == 0 || y == height - 1 {
                    isBorder = true
                } else {
                    let leftIdx  = idx - 1
                    let rightIdx = idx + 1
                    let upIdx    = idx - width
                    let downIdx  = idx + width

                    if mask[leftIdx] == 0 ||
                       mask[rightIdx] == 0 ||
                       mask[upIdx] == 0 ||
                       mask[downIdx] == 0 {
                        isBorder = true
                    }
                }

                if isBorder {
                    edge[idx] = 1
                }
            }
        }

        return edge
    }

    
    override func point(inside point: CGPoint, with event: UIEvent?) -> Bool {
        if point.y < 100 { return false }
        return true
    }
    
    // MARK: - Gesture Handlers
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
    
    // MARK: - Progress + first-detection hooks
    func startIfNeeded() {
        hasFirstDetection = false
        isDarkGateActive = false
        brightStreak = 0
        setProgress(0.05, text: "Starting camera…")
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
                self.progressView.alpha = 1
            }
        }
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
        
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: .back) else {
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
    
    // MARK: - Crop Pixel Buffer to BBox (vImage copy)
    private func cropPixelBuffer(_ pixelBuffer: CVPixelBuffer, toBBox det: DetectionSmarty, padding: Float = -0.05) -> CVPixelBuffer? {
        let cropStart = Date()
        
        let fullWf = Float(CVPixelBufferGetWidth(pixelBuffer))
        let fullHf = Float(CVPixelBufferGetHeight(pixelBuffer))
        
        let scaleX = fullWf / 640.0
        let scaleY = fullHf / 640.0
        
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
        guard let resized = letterbox(pixelBuffer, size: 640) else {
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
        } else if let arr = output.featureValue(for: "var_2421")?.multiArrayValue {
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
        let stage1DetectionsFull = extractDetections(from: detArray)
        let decodeEnd = Date()
        if self.debugMode {
            print("📊 Stage 1: \(stage1DetectionsFull.count) detections")
            print(String(format: "⏱ Stage1 detection decode: %.2f ms", decodeEnd.timeIntervalSince(decodeStart) * 1000.0))
        }
        
        let sorted = stage1DetectionsFull.sorted { $0.confidence > $1.confidence }
        
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
        if let croppedBuffer = cropPixelBuffer(pixelBuffer, toBBox: primary, padding: 0.1),
           let resizedCrop = letterbox(croppedBuffer, size: 640),
           let cropInputArray = pixelBufferToMLMultiArray(resizedCrop),
           let cropInputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": cropInputArray]) {
            
            let stage2InfStart = Date()
            if let cropOutput = try? model.prediction(from: cropInputProvider) {
                let stage2InfEnd = Date()
                if self.debugMode {
                    print(String(format: "⏱ Stage2 model.prediction: %.2f ms", stage2InfEnd.timeIntervalSince(stage2InfStart) * 1000.0))
                }
                
                var cropDetArray: MLMultiArray?
                if let arr = cropOutput.featureValue(for: "var_2421")?.multiArrayValue {
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
                    stage2Detections = extractDetections(from: detArray)
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
        
        let rawDetections = extractDetections(from: detArray)
        let uniqueDetections = applyNMS(rawDetections, iouThreshold: 1.0)
        let stage2KeptStage2 = applyNMS(uniqueDetections, iouThreshold: 1.0)
        
        if rawDetections.isEmpty {
            DispatchQueue.main.async {
                self.maskImageView.image = nil
                self.isProcessing = false
            }
            return
        }
        
        setProgress(0.8, text: "Building mask…")
        
        let cutoutStart = Date()
        generateCutoutTwoStage(
            stage1Detections: uniqueDetections,
            stage1Prototypes: prototypesArray,
            stage2Detections: stage2KeptStage2,
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
    
    private func letterbox(_ src: CVPixelBuffer, size: Int = 640) -> CVPixelBuffer? {
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
    
    // MARK: - Print Perimeter Debug Grid
    private func printPerimeterDebugGrid(_ title: String, mask: [UInt8], width: Int, height: Int) {
        guard self.debugMode else { return }
        
        print("\n🔴 [\(title)] (20x20 perimeter, # = edge, . = empty):")
        for gy in 0..<20 {
            var rowSymbols = ""
            for gx in 0..<20 {
                let y = min(gy * height / 20, height - 1)
                let x = min(gx * width / 20, width - 1)
                let idx = y * width + x
                rowSymbols += mask[idx] > 0 ? "#" : "."
            }
            print("   \(rowSymbols)")
        }
    }
    
    // MARK: - Check if pixel is within any bounding box
    private func isPixelWithinAnyBBox(x: Int, y: Int, 
                                     imageWidth: Int, imageHeight: Int,
                                     stage1Detections: [DetectionSmarty], 
                                     stage2Detections: [DetectionSmarty]) -> Bool {
        let allDetections = stage1Detections + stage2Detections
        
        // Convert pixel coordinates to model space (640x640)
        let modelSize: Float = 640.0
        let scaleX = modelSize / Float(imageWidth)
        let scaleY = modelSize / Float(imageHeight)
        
        let modelX = Float(x) * scaleX
        let modelY = Float(y) * scaleY
        
        for detection in allDetections {
            // Calculate bbox bounds in model space
            let left = detection.x - detection.width / 2
            let right = detection.x + detection.width / 2
            let top = detection.y - detection.height / 2
            let bottom = detection.y + detection.height / 2
            
            // Check if pixel is within this bbox
            if modelX >= left && modelX <= right && modelY >= top && modelY <= bottom {
                return true
            }
        }
        
        return false
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
    

    
    // MARK: - TWO-STAGE CUTOUT (with perimeter lock, NO mask memory)
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
        
        // Clear perimeter after some frames without any detections
        let hasDetections = !stage1Detections.isEmpty || !stage2Detections.isEmpty
        
        if !hasDetections {
            maskMemoryAge += 1
            if maskMemoryAge > maskMemoryMaxAge {
                if self.debugMode && bestPerimeterMask != nil {
                    print("🔴 [PERIMETER DEBUG] Clearing perimeter lock")
                    print("🔴 [PERIMETER DEBUG] Previous perimeter area: \(bestPerimeterArea)")
                    print("🔴 [PERIMETER DEBUG] Mask memory age exceeded: \(maskMemoryAge) > \(maskMemoryMaxAge)")
                }
                bestPerimeterMask = nil
                bestPerimeterArea = 0
                maskMemoryAge = 0
                if self.debugMode {
                    print("🧷 Perimeter cleared (no detections for \(maskMemoryMaxAge) frames)")
                    print("🔴 [PERIMETER DEBUG] Perimeter lock reset complete")
                }
            }
        } else {
            maskMemoryAge = 0
        }
        
        if self.debugMode {
            print("\n🎨 Generating TWO-STAGE UNION cutout")
            print("   Stage 1: \(stage1Detections.count) detections")
            print("   Stage 2: \(stage2Detections.count) detections")
            print("📐 Prototype shape: C=\(C), H=\(Hp), W=\(Wp)")
            print("🔴 [PERIMETER DEBUG] Current perimeter status:")
            if let _ = bestPerimeterMask {
                print("🔴 [PERIMETER DEBUG] - Active perimeter area: \(bestPerimeterArea) pixels")
                print("🔴 [PERIMETER DEBUG] - Mask memory age: \(maskMemoryAge)")
            } else {
                print("🔴 [PERIMETER DEBUG] - No active perimeter")
            }
            print("🔴 [PERIMETER DEBUG] - Has detections: \(hasDetections ? "YES" : "NO")")
        }
        
        var mappedStage2Detections: [DetectionSmarty] = []
        
        // Stage 1 prototype buffer
        let protoMatrix1 = makePrototypeBuffer(from: stage1Prototypes, C: C, Hp: Hp, Wp: Wp)
        
        // Global mask in proto-res space (Hp x Wp), Float 0/1
        var globalMask = [Float](repeating: 0, count: spatial)
        
        var primaryDet: DetectionSmarty? = nil
        var stage1PixelCount = 0
        
        // === STAGE 1 MASKS ===
        for (detIndex, det) in stage1Detections.enumerated() {
            var rawMask = [Float](repeating: 0, count: spatial)
            
            guard det.maskCoeffs.count == C else { continue }
            let hasInvalidCoeffs = det.maskCoeffs.contains { !$0.isFinite }
            guard !hasInvalidCoeffs else { continue }
            guard protoMatrix1.count == C * spatial else { continue }
            
            vDSP_mmul(det.maskCoeffs, 1, protoMatrix1, 1, &rawMask, 1,
                      1, vDSP_Length(spatial), vDSP_Length(C))
            
            if detIndex == 0 { primaryDet = det }
            
            let scale = Float(Wp) / 640.0
            let mx1 = max(0, Int((det.x - det.width / 2) * scale))
            let my1 = max(0, Int((det.y - det.height / 2) * scale))
            let mx2 = min(Wp, Int((det.x + det.width / 2) * scale))
            let my2 = min(Hp, Int((det.y + det.height / 2) * scale))
            
            if mx2 > mx1 && my2 > my1 {
                for py in my1..<my2 {
                    for px in mx1..<mx2 {
                        let idx = py * Wp + px
                        if rawMask[idx] > 0 {
                            globalMask[idx] = 1.0
                        }
                    }
                }
            }
        }
        
        for i in 0..<spatial { if globalMask[i] > 0 { stage1PixelCount += 1 } }
        
        // === STAGE 2 MASKS ===
        if let proto2 = stage2Prototypes, !stage2Detections.isEmpty {
            let protoMatrix2 = makePrototypeBuffer(from: proto2, C: C, Hp: Hp, Wp: Wp)
            
            let padding: Float = 0.1
            let cropX1 = max(0, primaryBBox.x - primaryBBox.width / 2 * (1 + padding))
            let cropY1 = max(0, primaryBBox.y - primaryBBox.height / 2 * (1 + padding))
            let cropX2 = min(640, primaryBBox.x + primaryBBox.width / 2 * (1 + padding))
            let cropY2 = min(640, primaryBBox.y + primaryBBox.height / 2 * (1 + padding))
            let cropW = cropX2 - cropX1
            let cropH = cropY2 - cropY1
            
            let s2ToS1ScaleX = cropW / 640.0
            let s2ToS1ScaleY = cropH / 640.0
            let scaleMask = Float(Wp) / 640.0
            
            for det in stage2Detections {
                var rawMask = [Float](repeating: 0, count: spatial)
                
                guard det.maskCoeffs.count == C else { continue }
                let hasInvalidCoeffs = det.maskCoeffs.contains { !$0.isFinite }
                guard !hasInvalidCoeffs else { continue }
                guard protoMatrix2.count == C * spatial else { continue }
                
                vDSP_mmul(det.maskCoeffs, 1, protoMatrix2, 1, &rawMask, 1,
                          1, vDSP_Length(spatial), vDSP_Length(C))
                
                let mx1_crop = max(0, Int((det.x - det.width / 2) * scaleMask))
                let my1_crop = max(0, Int((det.y - det.height / 2) * scaleMask))
                let mx2_crop = min(Wp, Int((det.x + det.width / 2) * scaleMask))
                let my2_crop = min(Hp, Int((det.y + det.height / 2) * scaleMask))
                
                if mx2_crop > mx1_crop && my2_crop > my1_crop {
                    for py_crop in my1_crop..<my2_crop {
                        for px_crop in mx1_crop..<mx2_crop {
                            let cropIdx = py_crop * Wp + px_crop
                            if rawMask[cropIdx] > 0 {
                                let fracX = Float(px_crop) / Float(Wp)
                                let fracY = Float(py_crop) / Float(Hp)
                                let fullX = cropX1 + fracX * 640.0 * s2ToS1ScaleX
                                let fullY = cropY1 + fracY * 640.0 * s2ToS1ScaleY
                                let mx_full = Int(fullX * scaleMask)
                                let my_full = Int(fullY * scaleMask)
                                
                                if mx_full >= 0 && mx_full < Wp && my_full >= 0 && my_full < Hp {
                                    globalMask[my_full * Wp + mx_full] = 1.0
                                }
                            }
                        }
                    }
                }
                
                let mapped = DetectionSmarty(
                    x: cropX1 + det.x * s2ToS1ScaleX,
                    y: cropY1 + det.y * s2ToS1ScaleY,
                    width: det.width * s2ToS1ScaleX,
                    height: det.height * s2ToS1ScaleY,
                    confidence: det.confidence,
                    classIdx: det.classIdx,
                    className: det.className,
                    maskCoeffs: det.maskCoeffs
                )
                mappedStage2Detections.append(mapped)
            }
        }
        
        // ============================
        //  🧷 PERIMETER LOCK + FILL
        // ============================
        
        // Build binary candidate from current frame
        var candidateMask = [UInt8](repeating: 0, count: spatial)
        var candidateArea = 0
        for i in 0..<spatial {
            if globalMask[i] > 0 {
                candidateMask[i] = 1
                candidateArea += 1
            }
        }
        
        // Update perimeter if bigger & overlaps enough OR if it's significantly different
        if let prevEdge = bestPerimeterMask, prevEdge.count == spatial {
            let prevArea = bestPerimeterArea
            let overlap = overlapFraction(previous: prevEdge, candidate: candidateMask)
            
            // More aggressive update conditions to prevent ghosting
            let shouldUpdate = candidateArea > prevArea && overlap >= 0.10 || // bigger with some overlap
                              overlap < 0.30 || // low overlap means detections moved significantly 
                              candidateArea > prevArea * 3 / 2 // significantly larger area
            
            if shouldUpdate {
                // Store the entire filled candidate area as "perimeter", not just the edge
                bestPerimeterMask = candidateMask // Store full mask instead of just edge
                bestPerimeterArea = candidateArea
                if self.debugMode {
                    print("🧷 [PERIMETER] Updated → area=\(candidateArea), overlap=\(String(format: "%.1f", overlap*100))%")
                    print("🔴 [PERIMETER DEBUG] Previous perimeter area: \(prevArea) pixels")
                    print("🔴 [PERIMETER DEBUG] New perimeter area: \(candidateArea) pixels")
                    print("🔴 [PERIMETER DEBUG] Overlap fraction: \(String(format: "%.3f", overlap)) (\(String(format: "%.1f", overlap*100))%)")
                    print("🔴 [PERIMETER DEBUG] Update reason: \(candidateArea > prevArea ? "larger area" : "low overlap")")
                    print("🔴 [PERIMETER DEBUG] Perimeter dimensions: \(Wp)×\(Hp)")
                    if let currentEdge = bestPerimeterMask {
                        let edgePixelCount = area(of: currentEdge)
                        print("🔴 [PERIMETER DEBUG] Edge pixels count: \(edgePixelCount)")
                        printPerimeterDebugGrid("UPDATED PERIMETER", mask: currentEdge, width: Wp, height: Hp)
                    }
                }
            } else if self.debugMode {
                print("🧷 [PERIMETER] Kept previous (\(prevArea) px)")
                print("🔴 [PERIMETER DEBUG] Candidate area: \(candidateArea), overlap: \(String(format: "%.3f", overlap))")
                print("🔴 [PERIMETER DEBUG] Rejection reason: area=\(candidateArea <= prevArea ? "not larger" : "adequate"), overlap=\(overlap >= 0.30 ? "sufficient" : "low")")
            }
        } else if candidateArea > 0 {
            // Store the entire filled candidate area as "perimeter", not just the edge
            bestPerimeterMask = candidateMask // Store full mask instead of just edge
            bestPerimeterArea = candidateArea
            if self.debugMode {
                print("🧷 [PERIMETER] Initialized with area=\(candidateArea)")
                print("🔴 [PERIMETER DEBUG] First perimeter creation")
                print("🔴 [PERIMETER DEBUG] Candidate mask area: \(candidateArea) pixels")
                print("🔴 [PERIMETER DEBUG] Perimeter dimensions: \(Wp)×\(Hp)")
                if let newEdge = bestPerimeterMask {
                    let edgePixelCount = area(of: newEdge)
                    print("🔴 [PERIMETER DEBUG] Initial edge pixels count: \(edgePixelCount)")
                    printPerimeterDebugGrid("INITIAL PERIMETER", mask: newEdge, width: Wp, height: Hp)
                }
            }
        }
        
        // Seal perimeter onto globalMask and fill it
        if let edgeMask = bestPerimeterMask, edgeMask.count == spatial {
            var sealedPixels = 0
            
            // First, seal the perimeter edges
            for i in 0..<spatial {
                if edgeMask[i] != 0 {
                    if globalMask[i] == 0 {
                        sealedPixels += 1
                    }
                    globalMask[i] = 1.0
                }
            }
            
            // Then flood fill the perimeter area to make it solid
            var tempMask = globalMask
            for y in 0..<Hp {
                let rowBase = y * Wp
                var first = -1
                var last = -1
                for x in 0..<Wp {
                    if edgeMask[rowBase + x] != 0 {
                        if first < 0 { first = x }
                        last = x
                    }
                }
                if first >= 0 && last > first {
                    for x in first...last {
                        tempMask[rowBase + x] = 1.0
                    }
                }
            }
            globalMask = tempMask
            if self.debugMode {
                let totalEdgePixels = area(of: edgeMask)
                print("🔴 [PERIMETER DEBUG] Sealing perimeter onto global mask")
                print("🔴 [PERIMETER DEBUG] Total edge pixels: \(totalEdgePixels)")
                print("🔴 [PERIMETER DEBUG] New pixels sealed: \(sealedPixels)")
                print("🔴 [PERIMETER DEBUG] Already covered pixels: \(totalEdgePixels - sealedPixels)")
            }
        }
        
        // FLOOD FILL inside perimeter (scanline per row)
        var floodFilledPixels = 0
        var floodFilledRows = 0
        for y in 0..<Hp {
            let rowBase = y * Wp
            var first = -1
            var last = -1
            for x in 0..<Wp {
                if globalMask[rowBase + x] > 0 {
                    if first < 0 { first = x }
                    last = x
                }
            }
            if first >= 0 && last > first {
                var rowFilled = 0
                for x in first...last {
                    if globalMask[rowBase + x] == 0 {
                        globalMask[rowBase + x] = 1.0
                        floodFilledPixels += 1
                        rowFilled += 1
                    }
                }
                if rowFilled > 0 {
                    floodFilledRows += 1
                }
            }
        }
        
        if self.debugMode {
            print("🔴 [PERIMETER DEBUG] Flood fill completed")
            print("🔴 [PERIMETER DEBUG] Flood filled pixels: \(floodFilledPixels)")
            print("🔴 [PERIMETER DEBUG] Rows with flood fill: \(floodFilledRows)/\(Hp)")
        }
        
        var finalPixelCount = 0
        for i in 0..<spatial { if globalMask[i] > 0 { finalPixelCount += 1 } }
        
        if self.debugMode {
            print("📊 Final mask: \(finalPixelCount)/\(spatial) (\(String(format: "%.1f", Float(finalPixelCount)/Float(spatial)*100))%)")
            print("🔴 [PERIMETER DEBUG] Final statistics:")
            print("🔴 [PERIMETER DEBUG] - Stage 1 pixels: \(stage1PixelCount)")
            print("🔴 [PERIMETER DEBUG] - Candidate area: \(candidateArea)")
            print("🔴 [PERIMETER DEBUG] - Final area: \(finalPixelCount)")
            print("🔴 [PERIMETER DEBUG] - Growth from stage 1: \(finalPixelCount - stage1PixelCount) pixels")
            if bestPerimeterMask != nil {
                print("🔴 [PERIMETER DEBUG] - Perimeter lock contributed to final result")
            }
        }
        
        // === RENDER ===
        autoreleasepool {
            let ciImage = CIImage(cvPixelBuffer: originalImage)
            let width = CVPixelBufferGetWidth(originalImage)
            let height = CVPixelBufferGetHeight(originalImage)
            
            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            
            guard let ctx = CGContext(
                data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            
            ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            guard let data = ctx.data else {
                DispatchQueue.main.async { self.isProcessing = false }
                return
            }
            
            let pixels = data.bindMemory(to: UInt8.self, capacity: width * height * 4)
            
            // Upscale mask
            var scaledMask = [Float](repeating: 0, count: width * height)
            globalMask.withUnsafeBufferPointer { srcPtr in
                scaledMask.withUnsafeMutableBufferPointer { dstPtr in
                    guard let sBase = srcPtr.baseAddress, let dBase = dstPtr.baseAddress else { return }
                    var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: sBase),
                                               height: vImagePixelCount(Hp), width: vImagePixelCount(Wp),
                                               rowBytes: Wp * MemoryLayout<Float>.size)
                    var dstBuf = vImage_Buffer(data: dBase,
                                               height: vImagePixelCount(height), width: vImagePixelCount(width),
                                               rowBytes: width * MemoryLayout<Float>.size)
                    vImageScale_PlanarF(&srcBuf, &dstBuf, nil, vImage_Flags(0))
                }
            }
            
            // Apply mask
            for y in 0..<height {
                let rowBase = pixels.advanced(by: y * width * 4)
                for x in 0..<width {
                    let idx = y * width + x
                    let pixelPtr = rowBase.advanced(by: x * 4)
                    if scaledMask[idx] > 0.5 {
                        pixelPtr[3] = 255
                    } else {
                        memset(pixelPtr, 0, 4)
                    }
                }
            }
            
            // Green edge + Red perimeter debug
            var totalPerimeterPixels = 0
            var drawnPerimeterPixels = 0
            for y in 0..<height {
                let rowBase = pixels.advanced(by: y * width * 4)
                for x in 0..<width {
                    let idx = y * width + x
                    if scaledMask[idx] <= 0.5 { continue }
                    
                    let up    = (y > 0)          ? scaledMask[idx - width] : 0
                    let down  = (y < height - 1) ? scaledMask[idx + width] : 0
                    let left  = (x > 0)          ? scaledMask[idx - 1]     : 0
                    let right = (x < width - 1)  ? scaledMask[idx + 1]     : 0
                    
                    // Check if this is a green edge (standard boundary)
                    let isEdge = up <= 0.5 || down <= 0.5 || left <= 0.5 || right <= 0.5
                    
                    // Check if this is part of the perimeter lock (red line) AND within a bbox
                    var isPerimeter = false
                    if self.debugMode, let edgeMask = self.bestPerimeterMask, edgeMask.count == spatial {
                        let maskY = y * Hp / height
                        let maskX = x * Wp / width
                        if maskY < Hp && maskX < Wp {
                            let maskIdx = maskY * Wp + maskX
                            if edgeMask[maskIdx] != 0 {
                                totalPerimeterPixels += 1
                                // Check if this pixel is within any detected bounding box
                                if self.isPixelWithinAnyBBox(x: x, y: y, 
                                                           imageWidth: width, imageHeight: height,
                                                           stage1Detections: stage1Detections,
                                                           stage2Detections: mappedStage2Detections) {
                                    isPerimeter = true
                                    drawnPerimeterPixels += 1
                                }
                            }
                        }
                    }
                    
                    let p = rowBase.advanced(by: x * 4)
                    if isPerimeter && self.debugMode {
                        // RED perimeter line (takes priority in debug mode, only within bboxes)
                        p[0] = 0; p[1] = 0; p[2] = 255; p[3] = 255
                    } else if isEdge {
                        // GREEN standard edge
                        p[0] = 0; p[1] = 255; p[2] = 0; p[3] = 255
                    }
                }
            }
            
            if self.debugMode && totalPerimeterPixels > 0 {
                let restrictedPixels = totalPerimeterPixels - drawnPerimeterPixels
                print("🔴 [PERIMETER DEBUG] Red line restriction:")
                print("🔴 [PERIMETER DEBUG] - Total perimeter pixels: \(totalPerimeterPixels)")
                print("🔴 [PERIMETER DEBUG] - Drawn within bboxes: \(drawnPerimeterPixels)")
                print("🔴 [PERIMETER DEBUG] - Restricted (outside bboxes): \(restrictedPixels)")
                print("🔴 [PERIMETER DEBUG] - Restriction ratio: \(String(format: "%.1f", Float(restrictedPixels)/Float(totalPerimeterPixels)*100))%")
            }
            
            // Update frame counter
            self.frameIndex += 1
            
            if self.debugMode {
                self.drawLabelsAndBoxes(ctx: ctx, stage1: stage1Detections, stage2: mappedStage2Detections,
                                        imageWidth: width, imageHeight: height, drawBoxes: true)
            }
            
            if let outCG = ctx.makeImage() {
                let finalCG = self.renderLabelsOnFinalImage(baseCGImage: outCG, width: width, height: height,
                                                            stage1: stage1Detections, stage2: mappedStage2Detections)
                DispatchQueue.main.async {
                    self.finishFirstDetectionIfNeeded()
                    self.maskImageView.image = UIImage(cgImage: finalCG)
                    self.isProcessing = false
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
        drawBoxes: Bool
    ) {
        let all = stage1 + stage2
        guard !all.isEmpty else { return }
        
        let W = CGFloat(imageWidth)
        let H = CGFloat(imageHeight)
        let modelSize: CGFloat = 640
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
        
        // YOLOE outputs are scaled to 640×640 model input
        let sx = CGFloat(imageWidth) / 640.0
        let sy = CGFloat(imageHeight) / 640.0
        
        // ---------------------------------------
        // Compute top-left of bbox in output image
        // ---------------------------------------
        var rectX = (cx - w/2.0) * sx
        var rectY = (cy - h/2.0) * sy
        
        // Clamp inside final image
        rectX = max(0, min(rectX, CGFloat(imageWidth - 1)))
        rectY = max(0, min(rectY, CGFloat(imageHeight - 1)))
        
        // ---------------------------------------
        // Prepare label text
        // ---------------------------------------
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
        let modelSize: CGFloat = 640.0
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
        
        // label text
        let confidence = Int(detection.confidence * 100)
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
    private func extractDetections(from detections: MLMultiArray) -> [DetectionSmarty] {
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
        guard let array = try? MLMultiArray(shape: [1, 3, 640, 640], dataType: .float32) else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let width = 640
        let height = 640
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
