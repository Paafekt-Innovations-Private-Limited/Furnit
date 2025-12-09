// SmartyPantsView.swift
// Two-Stage Detection: Full frame -> Crop to primary bbox -> Re-detect -> UNION BOTH
// With timing logs at crucial stages + real progress bar until first detection
// + Post-processing: seal perimeter, fill holes, keep largest connected mask

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
    var confidenceThreshold: Float = 0.2
    
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
    var confidenceThreshold: Float = 0.2
    var debugMode: Bool = true  // Enable debug prints and image saves
    
    // Detection mode: true = detect ALL objects, false = furniture classes only
    var detectAllObjects: Bool = false
    
    // MARK: Brightness gate (prevent processing when phone is lying down / frame is dark)
    private var lumaThreshold: Float = 0.08          // 0.0 .. 1.0
    private var brightStreak: Int = 0
    private var requiredBrightStreak: Int = 3         // require a few bright frames before resuming
    private var isDarkGateActive: Bool = false
    
    private let sessionQueue = DispatchQueue(label: "com.furnit.smarty.session")
    
    // Mask upscaling: true = bilinear (smooth edges), false = nearest-neighbor (faster)
    var useBilinearUpscaling: Bool = true
    
    // Mask threshold: values above this are considered "object"
    var maskThreshold: Float = 0.0
    
    // Add class property
    private var accumulatedPerimeter: [Float]? = nil
    
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

//    // MARK: Furniture & Household Classes (LVIS indices)
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
//        1099: "cot", 1183: "cradle", 3088: "playpen"
//    ]
    
    private var isAppActive: Bool = true

    // MARK: - Perimeter Lock Properties
    private var bestPerimeterMask: [UInt8]? = nil
    private var bestPerimeterArea: Int = 0
    private var maskMemoryAge: Int = 0
    private var maskMemoryMaxAge: Int = 3
    private var frameIndex: Int = 0

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

            // Brightness validation: if the frame is too dark, pause detection until bright again
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
        // Also stop any "in-flight" processing quickly.
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

        // Observe app going to background / foreground
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

        // Clear UI safely
        if Thread.isMainThread {
            self.maskImageView.image = nil
            self.layer.removeAllAnimations()
        } else {
            DispatchQueue.main.sync {
                self.maskImageView.image = nil
                self.layer.removeAllAnimations()
            }
        }

        // Perform ordered shutdown (blocks until everything is stopped)
        shutdownPipelinesSynchronously()
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
        sessionQueue.sync {
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
                if captureSession.canAddInput(input) { captureSession.addInput(input) }

                videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
                videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                videoOutput.alwaysDiscardsLateVideoFrames = true
                if captureSession.canAddOutput(videoOutput) { captureSession.addOutput(videoOutput) }
                if let conn = videoOutput.connection(with: .video) {
                    conn.videoRotationAngle = 90
                }
                captureSession.commitConfiguration()
                captureSession.startRunning()
            } catch {
                captureSession.commitConfiguration()
            }
        }
    }
    
    private func shutdownPipelinesSynchronously() {
        // 1) Stop frame delivery first (no more callbacks)
        videoOutput.setSampleBufferDelegate(nil, queue: nil)

        // 2) Stop the camera synchronously
        stopCamera()

        // 3) Quiesce detection queue:
        //    - prevent new work
        //    - wait for in-flight work to finish
        detectionQueue.sync {
            self.isProcessing = false
            self.mlModel = nil
        }
    }

    private func stopCamera() {
        sessionQueue.sync {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
            // Remove I/O to break retain cycles and ensure no more callbacks
            self.captureSession.beginConfiguration()
            self.captureSession.inputs.forEach { self.captureSession.removeInput($0) }
            self.captureSession.outputs.forEach { self.captureSession.removeOutput($0) }
            self.captureSession.commitConfiguration()
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

    // MARK: - Crop Pixel Buffer to BBox (vImage copy)
    private func cropPixelBuffer(_ pixelBuffer: CVPixelBuffer, toBBox det: DetectionSmarty, padding: Float = -0.15) -> CVPixelBuffer? {
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
            print("\n🚒 ===== NEW FRAME @ \(now.timeIntervalSince1970) =====")
            print("📌 ========== STAGE 1: FULL FRAME ==========")
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
        if self.debugMode { print("\n📌 ========== STAGE 2: CROPPED ==========") }

        var stage2Detections: [DetectionSmarty] = []
        var stage2Prototypes: MLMultiArray? = nil

        let stage2Start = Date()
        if let croppedBuffer = cropPixelBuffer(pixelBuffer, toBBox: primary, padding: 0),
           let resizedCrop = letterbox(croppedBuffer, size: 640),
           let cropInputArray = pixelBufferToMLMultiArray(resizedCrop),
           
           let cropInputProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": cropInputArray]) {
            guard cropInputArray.count == 1 * 3 * 640 * 640 else {
                if self.debugMode { print("⚠️ Stage2: bad input count:", cropInputArray.count) }
                self.isProcessing = false
                return
            }
            let options = MLPredictionOptions()
            // Flip this to true to see if the crash disappears on CPU:
            options.usesCPUOnly = false  // set true to test
            
            autoreleasepool{
                let stage2InfStart = Date()
                if let cropOutput = try? model.prediction(from: cropInputProvider, options: options) {
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
        let uniqueDetections = applyNMS(rawDetections, iouThreshold: 0.6)
        let stage2KeptStage2 = applyNMS(uniqueDetections, iouThreshold: 0.6)
        if rawDetections.isEmpty && stage2Detections.isEmpty {
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
            print(String(format: "🚒 Frame total (processFrame): %.2f ms", cutoutEnd.timeIntervalSince(frameStart) * 1000.0))
        }
    }

    private func buildGlobalMaskWithOverlapFilter(
        globalMask: inout [Float],
        allDetections: [DetectionSmarty],
        protoMatrix: [Float],
        primaryBBox: DetectionSmarty,
        C: Int, Wp: Int, Hp: Int,
        minOverlap: Float = 0.5
    ) {
        let spatial = Wp * Hp
        let scale = Float(Wp) / 640.0
        
        // Bbox bounds
        let bboxX1 = max(0, Int((primaryBBox.x - primaryBBox.width / 2) * scale))
        let bboxY1 = max(0, Int((primaryBBox.y - primaryBBox.height / 2) * scale))
        let bboxX2 = min(Wp, Int((primaryBBox.x + primaryBBox.width / 2) * scale))
        let bboxY2 = min(Hp, Int((primaryBBox.y + primaryBBox.height / 2) * scale))
        
        // Compute all masks
        var detMasks = [[Float]]()
        var detAreas = [Int]()
        
        for det in allDetections {
            var rawMask = [Float](repeating: 0, count: spatial)
            guard det.maskCoeffs.count == C else {
                detMasks.append(rawMask)
                detAreas.append(0)
                continue
            }
            guard !det.maskCoeffs.contains(where: { !$0.isFinite }) else {
                detMasks.append(rawMask)
                detAreas.append(0)
                continue
            }
            
            det.maskCoeffs.withUnsafeBufferPointer { coeffPtr in
                protoMatrix.withUnsafeBufferPointer { protoPtr in
                    guard let coeffBase = coeffPtr.baseAddress,
                          let protoBase = protoPtr.baseAddress else { return }
                    vDSP_mmul(coeffBase, 1, protoBase, 1, &rawMask, 1,
                              1, vDSP_Length(spatial), vDSP_Length(C))
                }
            }
            
            var area = 0
            for y in 0..<Hp {
                for x in 0..<Wp {
                    let idx = y * Wp + x
                    if rawMask[idx] > maskThreshold &&
                       x >= bboxX1 && x < bboxX2 &&
                       y >= bboxY1 && y < bboxY2 {
                        rawMask[idx] = 1.0
                        area += 1
                    } else {
                        rawMask[idx] = 0.0
                    }
                }
            }
            detMasks.append(rawMask)
            detAreas.append(area)
        }
        
        // START WITH LARGEST AREA, NOT HIGHEST CONFIDENCE
        guard let primaryIdx = detAreas.indices.max(by: { detAreas[$0] < detAreas[$1] }) else { return }
        guard detAreas[primaryIdx] > 0 else {
            if self.debugMode { print("⚠️ No valid masks with area > 0") }
            return
        }
        
        for i in 0..<spatial {
            globalMask[i] = detMasks[primaryIdx][i]
        }
        
        var used = [Bool](repeating: false, count: allDetections.count)
        used[primaryIdx] = true
        
        if self.debugMode {
            print("🔷 Primary (largest): \(allDetections[primaryIdx].className) @ \(Int(allDetections[primaryIdx].confidence * 100))%, area=\(detAreas[primaryIdx])px")
        }
        
        // Iteratively add masks with >= 50% overlap
        var changed = true
        while changed {
            changed = false
            
            for (idx, detMask) in detMasks.enumerated() {
                if used[idx] || detAreas[idx] == 0 { continue }
                
                var overlapCount = 0
                for i in 0..<spatial {
                    if detMask[i] > 0 && globalMask[i] > 0 {
                        overlapCount += 1
                    }
                }
                
                let overlapRatio = Float(overlapCount) / Float(detAreas[idx])
                
                if overlapRatio >= minOverlap {
                    var added = 0
                    for i in 0..<spatial {
                        if detMask[i] > 0 && globalMask[i] == 0 {
                            globalMask[i] = 1.0
                            added += 1
                        }
                    }
                    used[idx] = true
                    changed = true
                    
                    if self.debugMode {
                        print("🔗 Merged \(allDetections[idx].className) @ \(Int(allDetections[idx].confidence * 100))%: overlap=\(Int(overlapRatio * 100))%, +\(added)px")
                    }
                }
            }
        }
        
        if self.debugMode {
            let mergedCount = used.filter { $0 }.count
            var totalArea = 0
            for i in 0..<spatial { if globalMask[i] > 0 { totalArea += 1 } }
            print("🔷 buildGlobalMask: \(mergedCount)/\(allDetections.count) merged, total=\(totalArea)px")
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
    
    // MARK: - Draw Perimeter Outline (debug)
    private func drawPerimeterOutline(on pixels: UnsafeMutablePointer<UInt8>,
                                       mask: [Float],
                                       maskWidth: Int,
                                       maskHeight: Int,
                                       imageWidth: Int,
                                       imageHeight: Int) {
        let scaleX = Float(imageWidth) / Float(maskWidth)
        let scaleY = Float(imageHeight) / Float(maskHeight)
        
        // Find edge pixels in mask (pixels where mask=1 and at least one neighbor=0)
        for my in 0..<maskHeight {
            for mx in 0..<maskWidth {
                let idx = my * maskWidth + mx
                if mask[idx] > 0 {
                    // Check if edge (any 4-neighbor is 0 or out of bounds)
                    var isEdge = false
                    if mx == 0 || mx == maskWidth - 1 || my == 0 || my == maskHeight - 1 {
                        isEdge = true
                    } else {
                        if mask[idx - 1] == 0 || mask[idx + 1] == 0 ||
                           mask[idx - maskWidth] == 0 || mask[idx + maskWidth] == 0 {
                            isEdge = true
                        }
                    }
                    
                    if isEdge {
                        // Map to image coordinates and draw thick line (3x3)
                        let imgX = Int(Float(mx) * scaleX)
                        let imgY = Int(Float(my) * scaleY)
                        
                        // Draw 3x3 block for visibility
                        for dy in -1...1 {
                            for dx in -1...1 {
                                let px = imgX + dx
                                let py = imgY + dy
                                if px >= 0 && px < imageWidth && py >= 0 && py < imageHeight {
                                    let pixelIdx = (py * imageWidth + px) * 4
                                    // Cyan color: R=0, G=255, B=255, A=255
                                    // BGRA format: [B, G, R, A]
                                    pixels[pixelIdx + 0] = 255   // B
                                    pixels[pixelIdx + 1] = 255   // G
                                    pixels[pixelIdx + 2] = 0     // R
                                    pixels[pixelIdx + 3] = 255   // A
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // MARK: - Morphological Operations

    /// Dilate mask by 1 pixel (expand edges)
    private func dilateMask(_ mask: inout [UInt8], width: Int, height: Int) {
        var result = [UInt8](repeating: 0, count: width * height)
        
        for y in 0..<height {
            for x in 0..<width {
                if mask[y * width + x] == 1 {
                    // Set this pixel and all 8 neighbors
                    for dy in -1...1 {
                        for dx in -1...1 {
                            let nx = x + dx
                            let ny = y + dy
                            if nx >= 0 && nx < width && ny >= 0 && ny < height {
                                result[ny * width + nx] = 1
                            }
                        }
                    }
                }
            }
        }
        mask = result
    }

    /// Erode mask by 1 pixel (shrink edges)
    private func erodeMask(_ mask: inout [UInt8], width: Int, height: Int) {
        var result = [UInt8](repeating: 0, count: width * height)
        
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                let idx = y * width + x
                if mask[idx] == 1 {
                    // Keep only if all 8 neighbors are also 1
                    var keep = true
                    outer: for dy in -1...1 {
                        for dx in -1...1 {
                            if mask[(y + dy) * width + (x + dx)] == 0 {
                                keep = false
                                break outer
                            }
                        }
                    }
                    if keep { result[idx] = 1 }
                }
            }
        }
        mask = result
    }

    /// Morphological closing - bridges small gaps between touching edges
    private func closeMask(_ mask: inout [UInt8], width: Int, height: Int, radius: Int = 2) {
        for _ in 0..<radius {
            dilateMask(&mask, width: width, height: height)
        }
        for _ in 0..<radius {
            erodeMask(&mask, width: width, height: height)
        }
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
        
        print("\n🔲 [\(title)] (20x20 binary, * = object, . = background):")
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
    
    // MARK: - Perimeter Lock Helper Functions
    
    private func overlapFraction(previous: [UInt8], candidate: [UInt8]) -> Float {
        guard previous.count == candidate.count else { return 0.0 }
        
        var intersection = 0
        var union = 0
        
        for i in 0..<previous.count {
            if previous[i] != 0 || candidate[i] != 0 {
                union += 1
                if previous[i] != 0 && candidate[i] != 0 {
                    intersection += 1
                }
            }
        }
        
        return union > 0 ? Float(intersection) / Float(union) : 0.0
    }
    
    private func area(of mask: [UInt8]) -> Int {
        return mask.reduce(0) { count, value in
            count + (value != 0 ? 1 : 0)
        }
    }
    
    private func printPerimeterDebugGrid(_ label: String, mask: [UInt8], width: Int, height: Int) {
        guard debugMode else { return }
        print("🔴 [\(label)] Grid (\(width)x\(height)):")
        for y in 0..<min(height, 10) {
            var row = "🔴 "
            for x in 0..<min(width, 20) {
                let idx = y * width + x
                row += mask[idx] != 0 ? "█" : "·"
            }
            if width > 20 { row += "..." }
            print(row)
        }
        if height > 10 { print("🔴 ...") }
    }
    
    private func buildGlobalMaskWithOverlap(
        globalMask: inout [Float],
        allDetections: [DetectionSmarty],
        protoMatrix: [Float],
        primaryBBox: DetectionSmarty,
        C: Int, Wp: Int, Hp: Int,
        minOverlap: Float = 0.5
    ) {
        let spatial = Wp * Hp
        let scale = Float(Wp) / 640.0
        
        // Primary bbox in mask coords
        let pX1 = Int((primaryBBox.x - primaryBBox.width / 2) * scale)
        let pY1 = Int((primaryBBox.y - primaryBBox.height / 2) * scale)
        let pX2 = Int((primaryBBox.x + primaryBBox.width / 2) * scale)
        let pY2 = Int((primaryBBox.y + primaryBBox.height / 2) * scale)
        let primaryBBoxArea = (pX2 - pX1) * (pY2 - pY1)
        
        // Compute all masks and areas
        var detMasks = [[Float]]()
        var detAreas = [Int]()
        var detBBoxOverlaps = [Float]()  // Overlap with primaryBBox
        
        for det in allDetections {
            var rawMask = [Float](repeating: 0, count: spatial)
            guard det.maskCoeffs.count == C else {
                detMasks.append(rawMask)
                detAreas.append(0)
                detBBoxOverlaps.append(0)
                continue
            }
            guard !det.maskCoeffs.contains(where: { !$0.isFinite }) else {
                detMasks.append(rawMask)
                detAreas.append(0)
                detBBoxOverlaps.append(0)
                continue
            }
            
            det.maskCoeffs.withUnsafeBufferPointer { coeffPtr in
                protoMatrix.withUnsafeBufferPointer { protoPtr in
                    guard let coeffBase = coeffPtr.baseAddress,
                          let protoBase = protoPtr.baseAddress else { return }
                    vDSP_mmul(coeffBase, 1, protoBase, 1, &rawMask, 1,
                              1, vDSP_Length(spatial), vDSP_Length(C))
                }
            }
            
            // Threshold and count area + overlap with primaryBBox
            var area = 0
            var overlapWithPrimary = 0
            for y in 0..<Hp {
                for x in 0..<Wp {
                    let idx = y * Wp + x
                    if rawMask[idx] > maskThreshold {
                        rawMask[idx] = 1.0
                        area += 1
                        // Check if inside primaryBBox
                        if x >= pX1 && x < pX2 && y >= pY1 && y < pY2 {
                            overlapWithPrimary += 1
                        }
                    } else {
                        rawMask[idx] = 0.0
                    }
                }
            }
            
            detMasks.append(rawMask)
            detAreas.append(area)
            let bboxOverlapRatio = area > 0 ? Float(overlapWithPrimary) / Float(area) : 0
            detBBoxOverlaps.append(bboxOverlapRatio)
        }
        
        // Find primary: largest area AMONG those that overlap >= 50% with primaryBBox
        var primaryIdx: Int? = nil
        var maxArea = 0
        for i in 0..<allDetections.count {
            if detBBoxOverlaps[i] >= 0.5 && detAreas[i] > maxArea {
                maxArea = detAreas[i]
                primaryIdx = i
            }
        }
        
        // Fallback: if none overlap 50%, take largest that overlaps at all
        if primaryIdx == nil {
            for i in 0..<allDetections.count {
                if detBBoxOverlaps[i] > 0 && detAreas[i] > maxArea {
                    maxArea = detAreas[i]
                    primaryIdx = i
                }
            }
        }
        
        guard let pIdx = primaryIdx, detAreas[pIdx] > 0 else {
            if self.debugMode { print("⚠️ No valid masks within primaryBBox") }
            return
        }
        
        for i in 0..<spatial {
            globalMask[i] = detMasks[pIdx][i]
        }
        
        var used = [Bool](repeating: false, count: allDetections.count)
        used[pIdx] = true
        
        if self.debugMode {
            print("🔷 Primary (in bbox): \(allDetections[pIdx].className) @ \(Int(allDetections[pIdx].confidence * 100))%, area=\(detAreas[pIdx])px, bboxOverlap=\(Int(detBBoxOverlaps[pIdx] * 100))%")
        }
        
        // Iteratively merge masks with >= 50% overlap with globalMask
        var changed = true
        while changed {
            changed = false
            
            for (idx, detMask) in detMasks.enumerated() {
                if used[idx] || detAreas[idx] == 0 { continue }
                
                var overlapCount = 0
                for i in 0..<spatial {
                    if detMask[i] > 0 && globalMask[i] > 0 {
                        overlapCount += 1
                    }
                }
                
                let overlapRatio = Float(overlapCount) / Float(detAreas[idx])
                
                if overlapRatio >= minOverlap {
                    var added = 0
                    for i in 0..<spatial {
                        if detMask[i] > 0 && globalMask[i] == 0 {
                            globalMask[i] = 1.0
                            added += 1
                        }
                    }
                    used[idx] = true
                    changed = true
                    
                    if self.debugMode {
                        print("🔗 Merged \(allDetections[idx].className) @ \(Int(allDetections[idx].confidence * 100))%: overlap=\(Int(overlapRatio * 100))%, +\(added)px")
                    }
                } else if self.debugMode && overlapRatio > 0 {
                    print("⏭️ Skipped \(allDetections[idx].className) @ \(Int(allDetections[idx].confidence * 100))%: overlap=\(Int(overlapRatio * 100))% < 50%")
                }
            }
        }
        
        if self.debugMode {
            let mergedCount = used.filter { $0 }.count
            var totalArea = 0
            for i in 0..<spatial { if globalMask[i] > 0 { totalArea += 1 } }
            print("🔷 buildGlobalMask: \(mergedCount)/\(allDetections.count) merged, total=\(totalArea)px")
        }
    }
    
    private func buildGlobalMaskUnionAll(
        globalMask: inout [Float],
        allDetections: [DetectionSmarty],
        protoMatrix: [Float],
        primaryBBox: DetectionSmarty,
        C: Int, Wp: Int, Hp: Int
    ) {
        let spatial = Wp * Hp
        let scale = Float(Wp) / 640.0
        
        // Bbox bounds
        let bboxX1 = max(0, Int((primaryBBox.x - primaryBBox.width / 2) * scale))
        let bboxY1 = max(0, Int((primaryBBox.y - primaryBBox.height / 2) * scale))
        let bboxX2 = min(Wp, Int((primaryBBox.x + primaryBBox.width / 2) * scale))
        let bboxY2 = min(Hp, Int((primaryBBox.y + primaryBBox.height / 2) * scale))
        
        var totalAdded = 0
        
        for det in allDetections {
            var rawMask = [Float](repeating: 0, count: spatial)
            guard det.maskCoeffs.count == C else { continue }
            guard !det.maskCoeffs.contains(where: { !$0.isFinite }) else { continue }
            
            det.maskCoeffs.withUnsafeBufferPointer { coeffPtr in
                protoMatrix.withUnsafeBufferPointer { protoPtr in
                    guard let coeffBase = coeffPtr.baseAddress,
                          let protoBase = protoPtr.baseAddress else { return }
                    vDSP_mmul(coeffBase, 1, protoBase, 1, &rawMask, 1,
                              1, vDSP_Length(spatial), vDSP_Length(C))
                }
            }
            
            // Union into globalMask (within bbox)
            var added = 0
            for y in bboxY1..<bboxY2 {
                for x in bboxX1..<bboxX2 {
                    let idx = y * Wp + x
                    if rawMask[idx] > maskThreshold && globalMask[idx] == 0 {
                        globalMask[idx] = 1.0
                        added += 1
                    }
                }
            }
            totalAdded += added
        }
        
        if self.debugMode {
            var totalArea = 0
            for i in 0..<spatial { if globalMask[i] > 0 { totalArea += 1 } }
            print("🔷 buildGlobalMaskUnionAll: \(allDetections.count) detections → \(totalArea)px")
        }
    }

    private func postProcessMaskWithMemory(_ mask: inout [Float], width: Int, height: Int) {
        let count = width * height
        
        // Init or reset if size changed
        if accumulatedPerimeter == nil || accumulatedPerimeter!.count != count {
            accumulatedPerimeter = [Float](repeating: 0, count: count)
        }
        
        // Union current frame into accumulated
        for i in 0..<count {
            if mask[i] > 0 {
                accumulatedPerimeter![i] = 1.0
            }
        }
        
        // Create outer ring from accumulated
        var ring = accumulatedPerimeter!
        
        // Horizontal fill
        for y in 0..<height {
            var minX = -1, maxX = -1
            for x in 0..<width {
                if ring[y * width + x] > 0 {
                    if minX < 0 { minX = x }
                    maxX = x
                }
            }
            if minX >= 0 {
                for x in minX...maxX { ring[y * width + x] = 1.0 }
            }
        }
        
        // Vertical fill
        for x in 0..<width {
            var minY = -1, maxY = -1
            for y in 0..<height {
                if ring[y * width + x] > 0 {
                    if minY < 0 { minY = y }
                    maxY = y
                }
            }
            if minY >= 0 {
                for y in minY...maxY { ring[y * width + x] = 1.0 }
            }
        }
        
        mask = ring
    }

    // Call this when user resets or object changes significantly
    func resetAccumulatedPerimeter() {
        accumulatedPerimeter = nil
    }
    
    // MARK: - Mask Post-Processing
    
    // MARK: - Outer Ring Post-Processing

    private func postProcessMask(_ mask: inout [Float], width: Int, height: Int) {
        let t0 = Date()
        let count = width * height
        guard count > 0 else { return }
        
        // Step 1: Mark exterior by flood fill from edges (4-connectivity)
        var exterior = [Bool](repeating: false, count: count)
        
        // Simple queue-based flood from all edge background pixels
        var queue = [Int]()
        queue.reserveCapacity(count / 4)
        
        // Seed top & bottom edges
        for x in 0..<width {
            if mask[x] == 0 { queue.append(x) }
            let bottomIdx = (height - 1) * width + x
            if mask[bottomIdx] == 0 { queue.append(bottomIdx) }
        }
        // Seed left & right edges
        for y in 0..<height {
            let leftIdx = y * width
            let rightIdx = y * width + width - 1
            if mask[leftIdx] == 0 { queue.append(leftIdx) }
            if mask[rightIdx] == 0 { queue.append(rightIdx) }
        }
        
        // Flood fill exterior
        var head = 0
        while head < queue.count {
            let idx = queue[head]
            head += 1
            
            if exterior[idx] { continue }
            if mask[idx] > 0 { continue }
            
            exterior[idx] = true
            
            let x = idx % width
            let y = idx / width
            
            if x > 0 && !exterior[idx - 1] && mask[idx - 1] == 0 { queue.append(idx - 1) }
            if x < width - 1 && !exterior[idx + 1] && mask[idx + 1] == 0 { queue.append(idx + 1) }
            if y > 0 && !exterior[idx - width] && mask[idx - width] == 0 { queue.append(idx - width) }
            if y < height - 1 && !exterior[idx + width] && mask[idx + width] == 0 { queue.append(idx + width) }
        }
        
        // Step 2: Everything NOT exterior = fill it (object + holes inside)
        for i in 0..<count {
            mask[i] = exterior[i] ? 0.0 : 1.0
        }
        
        if self.debugMode {
            var finalCount = 0
            for i in 0..<count { if mask[i] > 0 { finalCount += 1 } }
            let dt = Date().timeIntervalSince(t0) * 1000.0
            print("🔷 postProcessMask: filled interior = \(finalCount)px (\(String(format: "%.1f", Float(finalCount)/Float(count)*100))%)")
            print(String(format: "⏱ postProcessMask: %.2f ms", dt))
        }
    }

    // Fast flood fill using pre-allocated stack
    private func floodFillLabelFast(_ labels: inout [Int], binary: [UInt8], width: Int, height: Int, startX: Int, startY: Int, label: Int) -> Int {
        var stack = ContiguousArray<Int32>(repeating: 0, count: 8192)
        var stackTop = 0
        stack[stackTop] = Int32(startX)
        stack[stackTop + 1] = Int32(startY)
        stackTop += 2
        
        var area = 0
        
        while stackTop > 0 {
            stackTop -= 2
            let x = Int(stack[stackTop])
            let y = Int(stack[stackTop + 1])
            let idx = y * width + x
            
            if x < 0 || x >= width || y < 0 || y >= height { continue }
            if binary[idx] == 0 || labels[idx] != 0 { continue }
            
            labels[idx] = label
            area += 1
            
            // 8-connectivity
            if stackTop + 16 < stack.count {
                if x > 0 { stack[stackTop] = Int32(x-1); stack[stackTop+1] = Int32(y); stackTop += 2 }
                if x < width-1 { stack[stackTop] = Int32(x+1); stack[stackTop+1] = Int32(y); stackTop += 2 }
                if y > 0 { stack[stackTop] = Int32(x); stack[stackTop+1] = Int32(y-1); stackTop += 2 }
                if y < height-1 { stack[stackTop] = Int32(x); stack[stackTop+1] = Int32(y+1); stackTop += 2 }
                if x > 0 && y > 0 { stack[stackTop] = Int32(x-1); stack[stackTop+1] = Int32(y-1); stackTop += 2 }
                if x < width-1 && y > 0 { stack[stackTop] = Int32(x+1); stack[stackTop+1] = Int32(y-1); stackTop += 2 }
                if x > 0 && y < height-1 { stack[stackTop] = Int32(x-1); stack[stackTop+1] = Int32(y+1); stackTop += 2 }
                if x < width-1 && y < height-1 { stack[stackTop] = Int32(x+1); stack[stackTop+1] = Int32(y+1); stackTop += 2 }
            }
        }
        return area
    }

    private func fillHolesFast(_ mask: inout [UInt8], width: Int, height: Int) {
        let count = width * height
        var visited = [Bool](repeating: false, count: count)
        var stack = ContiguousArray<Int32>(repeating: 0, count: 4096)
        var stackTop = 0
        
        // Seed from edges
        for x in 0..<width {
            if mask[x] == 0 { stack[stackTop] = Int32(x); stack[stackTop+1] = 0; stackTop += 2 }
            if mask[(height-1)*width + x] == 0 { stack[stackTop] = Int32(x); stack[stackTop+1] = Int32(height-1); stackTop += 2 }
        }
        for y in 0..<height {
            if mask[y*width] == 0 { stack[stackTop] = 0; stack[stackTop+1] = Int32(y); stackTop += 2 }
            if mask[y*width + width-1] == 0 { stack[stackTop] = Int32(width-1); stack[stackTop+1] = Int32(y); stackTop += 2 }
        }
        
        while stackTop > 0 {
            stackTop -= 2
            let x = Int(stack[stackTop])
            let y = Int(stack[stackTop + 1])
            let idx = y * width + x
            
            if x < 0 || x >= width || y < 0 || y >= height { continue }
            if mask[idx] != 0 || visited[idx] { continue }
            
            visited[idx] = true
            
            if stackTop + 8 < stack.count {
                if x > 0 { stack[stackTop] = Int32(x-1); stack[stackTop+1] = Int32(y); stackTop += 2 }
                if x < width-1 { stack[stackTop] = Int32(x+1); stack[stackTop+1] = Int32(y); stackTop += 2 }
                if y > 0 { stack[stackTop] = Int32(x); stack[stackTop+1] = Int32(y-1); stackTop += 2 }
                if y < height-1 { stack[stackTop] = Int32(x); stack[stackTop+1] = Int32(y+1); stackTop += 2 }
            }
        }
        
        // Fill holes (not reachable from edge)
        for i in 0..<count {
            if !visited[i] { mask[i] = 1 }
        }
    }

    /// Flood fill to label a connected component, returns area
    private func floodFillLabel(_ labels: inout [Int], binary: [UInt8], width: Int, height: Int, startX: Int, startY: Int, label: Int) -> Int {
        var stack = [(x: Int, y: Int)]()
        stack.reserveCapacity(1024)
        stack.append((startX, startY))
        
        var area = 0
        let dx = [-1, 1, 0, 0, -1, -1, 1, 1] // 8-connectivity
        let dy = [0, 0, -1, 1, -1, 1, -1, 1]
        
        while !stack.isEmpty {
            let (x, y) = stack.removeLast()
            let idx = y * width + x
            
            guard x >= 0 && x < width && y >= 0 && y < height else { continue }
            guard binary[idx] == 1 && labels[idx] == 0 else { continue }
            
            labels[idx] = label
            area += 1
            
            for d in 0..<8 {
                let nx = x + dx[d]
                let ny = y + dy[d]
                if nx >= 0 && nx < width && ny >= 0 && ny < height {
                    let nidx = ny * width + nx
                    if binary[nidx] == 1 && labels[nidx] == 0 {
                        stack.append((nx, ny))
                    }
                }
            }
        }
        return area
    }

    /// Fill holes by flood-filling background from edges, then inverting
    private func fillHoles(_ mask: inout [UInt8], width: Int, height: Int) {
        let count = width * height
        var visited = [Bool](repeating: false, count: count)
        var stack = [(x: Int, y: Int)]()
        stack.reserveCapacity(1024)
        
        // Seed from all edge pixels that are background
        for x in 0..<width {
            if mask[x] == 0 { stack.append((x, 0)) }
            if mask[(height-1) * width + x] == 0 { stack.append((x, height-1)) }
        }
        for y in 0..<height {
            if mask[y * width] == 0 { stack.append((0, y)) }
            if mask[y * width + width - 1] == 0 { stack.append((width-1, y)) }
        }
        
        let dx = [-1, 1, 0, 0]
        let dy = [0, 0, -1, 1]
        
        // Flood fill background from edges
        while !stack.isEmpty {
            let (x, y) = stack.removeLast()
            let idx = y * width + x
            
            guard x >= 0 && x < width && y >= 0 && y < height else { continue }
            guard mask[idx] == 0 && !visited[idx] else { continue }
            
            visited[idx] = true
            
            for d in 0..<4 {
                let nx = x + dx[d]
                let ny = y + dy[d]
                if nx >= 0 && nx < width && ny >= 0 && ny < height {
                    let nidx = ny * width + nx
                    if mask[nidx] == 0 && !visited[nidx] {
                        stack.append((nx, ny))
                    }
                }
            }
        }
        
        // Everything NOT visited from edges = inside the perimeter = fill it
        for i in 0..<count {
            if !visited[i] {
                mask[i] = 1
            }
        }
    }
    
    private func fillMaskInterior(_ mask: inout [Float], width: Int, height: Int) {
        let count = width * height
        guard count > 0 else { return }
        
        // Step 1: Dilate mask to close small gaps (1-2 pixel gaps)
        var dilated = [Float](repeating: 0, count: count)
        for y in 0..<height {
            for x in 0..<width {
                if mask[y * width + x] > 0 {
                    // Set 3x3 neighborhood
                    for dy in -1...1 {
                        for dx in -1...1 {
                            let nx = x + dx
                            let ny = y + dy
                            if nx >= 0 && nx < width && ny >= 0 && ny < height {
                                dilated[ny * width + nx] = 1.0
                            }
                        }
                    }
                }
            }
        }
        
        // Step 2: Flood fill exterior on dilated mask
        var exterior = [Bool](repeating: false, count: count)
        var queue = [Int](repeating: 0, count: count)
        var head = 0
        var tail = 0
        
        for x in 0..<width {
            let topIdx = x
            let botIdx = (height - 1) * width + x
            if dilated[topIdx] == 0 && !exterior[topIdx] {
                exterior[topIdx] = true
                queue[tail] = topIdx; tail += 1
            }
            if dilated[botIdx] == 0 && !exterior[botIdx] {
                exterior[botIdx] = true
                queue[tail] = botIdx; tail += 1
            }
        }
        for y in 0..<height {
            let leftIdx = y * width
            let rightIdx = y * width + width - 1
            if dilated[leftIdx] == 0 && !exterior[leftIdx] {
                exterior[leftIdx] = true
                queue[tail] = leftIdx; tail += 1
            }
            if dilated[rightIdx] == 0 && !exterior[rightIdx] {
                exterior[rightIdx] = true
                queue[tail] = rightIdx; tail += 1
            }
        }
        
        while head < tail {
            let idx = queue[head]
            head += 1
            let x = idx % width
            let y = idx / width
            
            if x > 0 {
                let n = idx - 1
                if dilated[n] == 0 && !exterior[n] { exterior[n] = true; queue[tail] = n; tail += 1 }
            }
            if x < width - 1 {
                let n = idx + 1
                if dilated[n] == 0 && !exterior[n] { exterior[n] = true; queue[tail] = n; tail += 1 }
            }
            if y > 0 {
                let n = idx - width
                if dilated[n] == 0 && !exterior[n] { exterior[n] = true; queue[tail] = n; tail += 1 }
            }
            if y < height - 1 {
                let n = idx + width
                if dilated[n] == 0 && !exterior[n] { exterior[n] = true; queue[tail] = n; tail += 1 }
            }
        }
        
        // Step 3: Fill - but use original mask edges, not dilated
        // Interior = not exterior
        for i in 0..<count {
            if !exterior[i] {
                mask[i] = 1.0
            }
            // Keep original mask pixels even if in exterior zone
            // (this prevents shrinking the original detection)
        }
        
        if self.debugMode {
            var finalCount = 0
            for i in 0..<count { if mask[i] > 0 { finalCount += 1 } }
            print("🕳️ fillMaskInterior: final=\(finalCount)px")
        }
    }
    
    // MARK: - TWO-STAGE CUTOUT (CLEAN - NO GHOST)
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
            print("\n🌈 Generating TWO-STAGE UNION cutout")
            print("   Stage 1: \(stage1Detections.count) detections")
            print("   Stage 2: \(stage2Detections.count) detections (Stage2 coords)")
            print("📐 Prototype shape: C=\(C), H=\(Hp), W=\(Wp)")
        }

        var mappedStage2Detections: [DetectionSmarty] = []

        // Stage 1 prototype buffer
        let protoStage1Start = Date()
        let protoMatrix1 = makePrototypeBuffer(from: stage1Prototypes, C: C, Hp: Hp, Wp: Wp)
        let protoStage1End = Date()
        if self.debugMode {
            print(String(format: "⏱ Stage1 prototype buffer build (Accelerate): %.2f ms",
                         protoStage1End.timeIntervalSince(protoStage1Start) * 1000.0))
        }

        var globalMask = [Float](repeating: 0, count: spatial)

        // Map Stage 2 detections to Stage 1 coords
        if let proto2 = stage2Prototypes, !stage2Detections.isEmpty {
            let padding: Float = 0.1
            let cropX1 = max(0, primaryBBox.x - primaryBBox.width / 2 * (1 + padding))
            let cropY1 = max(0, primaryBBox.y - primaryBBox.height / 2 * (1 + padding))
            let cropX2 = min(640, primaryBBox.x + primaryBBox.width / 2 * (1 + padding))
            let cropY2 = min(640, primaryBBox.y + primaryBBox.height / 2 * (1 + padding))
            let cropW = cropX2 - cropX1
            let cropH = cropY2 - cropY1
            let s2ToS1ScaleX = cropW / 640.0
            let s2ToS1ScaleY = cropH / 640.0

            for det in stage2Detections {
                let newX = cropX1 + det.x * s2ToS1ScaleX
                let newY = cropY1 + det.y * s2ToS1ScaleY
                let newW = det.width * s2ToS1ScaleX
                let newH = det.height * s2ToS1ScaleY

                let mapped = DetectionSmarty(
                    x: newX, y: newY, width: newW, height: newH,
                    confidence: det.confidence, classIdx: det.classIdx,
                    className: det.className, maskCoeffs: det.maskCoeffs
                )
                mappedStage2Detections.append(mapped)
            }
        }

        // Combine all detections
        let allDetections = stage1Detections + mappedStage2Detections

        // Build globalMask with 50% overlap filter
        let buildStart = Date()
        // Find this call and add primaryBBox:
        buildGlobalMaskWithOverlap(
            globalMask: &globalMask,
            allDetections: allDetections,
            protoMatrix: protoMatrix1,
            primaryBBox: primaryBBox,  // ← Add this
            C: C, Wp: Wp, Hp: Hp,
            minOverlap: 0.5
        )
        let buildEnd = Date()
        
        if self.debugMode {
            var rawCount = 0
            for i in 0..<spatial { if globalMask[i] > 0 { rawCount += 1 } }
            print(String(format: "⏱ buildGlobalMaskWithOverlapFilter: %.2f ms", buildEnd.timeIntervalSince(buildStart) * 1000.0))
            print("📊 After overlap filter: \(rawCount)/\(spatial) pixels (\(String(format: "%.1f", Float(rawCount)/Float(spatial)*100))%)")
        }

        if self.debugMode {
            let maskCopy = globalMask  // Copy to avoid race
            DispatchQueue.main.async {
                self.saveMaskToPhotos(maskCopy, width: Wp, height: Hp, label: "globalMask_raw")
            }
        }

        // Fill inside perimeter
        let ppStart = Date()
        fillInsidePerimeter(&globalMask, width: Wp, height: Hp)
        let ppEnd = Date()
        
        var finalPixelCount = 0
        for i in 0..<spatial { if globalMask[i] > 0 { finalPixelCount += 1 } }

        if self.debugMode {
            saveMaskToPhotos(globalMask, width: Wp, height: Hp, label: "globalMask_filled")
            print(String(format: "⏱ fillInsidePerimeter: %.2f ms", ppEnd.timeIntervalSince(ppStart) * 1000.0))
            print("📊 FINAL MASK: \(finalPixelCount)/\(spatial) pixels (\(String(format: "%.1f", Float(finalPixelCount)/Float(spatial)*100))%)")
        }

        // =============================================
        // RENDER TO IMAGE
        // =============================================
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

            // Apply mask as alpha
            for py in 0..<height {
                let my = min(max(Int(Float(py) * scaleY), 0), Hp - 1)
                let maskRowOffset = my * Wp
                let rowBase = pixels.advanced(by: py * width * 4)

                for px in 0..<width {
                    let mx = min(max(Int(Float(px) * scaleX), 0), Wp - 1)
                    let maskIdx = maskRowOffset + mx
                    let pixelPtr = rowBase.advanced(by: px * 4)
                    
                    if globalMask[maskIdx] > 0 {
                        pixelPtr[3] = 255  // opaque
                    } else {
                        pixelPtr[0] = 0
                        pixelPtr[1] = 0
                        pixelPtr[2] = 0
                        pixelPtr[3] = 0  // transparent
                    }
                }
            }
            
            // Draw perimeter outline in debug mode
            if self.debugMode {
                drawPerimeterOutline(
                    on: pixels,
                    mask: globalMask,
                    maskWidth: Wp,
                    maskHeight: Hp,
                    imageWidth: width,
                    imageHeight: height
                )
            }

            // Draw labels
            self.drawLabelsAndBoxes(
                ctx: ctx,
                stage1: stage1Detections,
                stage2: mappedStage2Detections,
                imageWidth: width,
                imageHeight: height,
                drawBoxes: self.debugMode
            )

            let renderEnd = Date()
            if self.debugMode {
                print(String(format: "⏱ Rendering: %.2f ms", renderEnd.timeIntervalSince(renderStart) * 1000.0))
                print(String(format: "⏱ generateCutoutTwoStage total: %.2f ms", renderEnd.timeIntervalSince(funcStart) * 1000.0))
                print("✅ ==================== FRAME COMPLETE ====================\n")
            }

            if let outCG = ctx.makeImage() {
                DispatchQueue.main.async {
                    self.finishFirstDetectionIfNeeded()
                    self.maskImageView.image = UIImage(cgImage: outCG)
                    self.isProcessing = false
                }
            } else {
                DispatchQueue.main.async { self.isProcessing = false }
            }
        }
    }

//    // MARK: - Build Global Mask with 50% Overlap Filter
//    private func buildGlobalMaskWithOverlapFilter(
//        globalMask: inout [Float],
//        allDetections: [DetectionSmarty],
//        protoMatrix: [Float],
//        primaryBBox: DetectionSmarty,
//        C: Int, Wp: Int, Hp: Int,
//        minOverlap: Float = 0.5
//    ) {
//        let spatial = Wp * Hp
//        let scale = Float(Wp) / 640.0
//        
//        // Bbox bounds
//        let bboxX1 = max(0, Int((primaryBBox.x - primaryBBox.width / 2) * scale))
//        let bboxY1 = max(0, Int((primaryBBox.y - primaryBBox.height / 2) * scale))
//        let bboxX2 = min(Wp, Int((primaryBBox.x + primaryBBox.width / 2) * scale))
//        let bboxY2 = min(Hp, Int((primaryBBox.y + primaryBBox.height / 2) * scale))
//        
//        // Compute all masks
//        var detMasks = [[Float]]()
//        var detAreas = [Int]()
//        
//        for det in allDetections {
//            var rawMask = [Float](repeating: 0, count: spatial)
//            guard det.maskCoeffs.count == C else {
//                detMasks.append(rawMask)
//                detAreas.append(0)
//                continue
//            }
//            guard !det.maskCoeffs.contains(where: { !$0.isFinite }) else {
//                detMasks.append(rawMask)
//                detAreas.append(0)
//                continue
//            }
//            
//            det.maskCoeffs.withUnsafeBufferPointer { coeffPtr in
//                protoMatrix.withUnsafeBufferPointer { protoPtr in
//                    guard let coeffBase = coeffPtr.baseAddress,
//                          let protoBase = protoPtr.baseAddress else { return }
//                    vDSP_mmul(coeffBase, 1, protoBase, 1, &rawMask, 1,
//                              1, vDSP_Length(spatial), vDSP_Length(C))
//                }
//            }
//            
//            var area = 0
//            for y in 0..<Hp {
//                for x in 0..<Wp {
//                    let idx = y * Wp + x
//                    if rawMask[idx] > maskThreshold &&
//                       x >= bboxX1 && x < bboxX2 &&
//                       y >= bboxY1 && y < bboxY2 {
//                        rawMask[idx] = 1.0
//                        area += 1
//                    } else {
//                        rawMask[idx] = 0.0
//                    }
//                }
//            }
//            detMasks.append(rawMask)
//            detAreas.append(area)
//        }
//        
//        // Start with highest confidence detection
//        guard let primaryIdx = allDetections.indices.max(by: { allDetections[$0].confidence < allDetections[$1].confidence }) else { return }
//        
//        for i in 0..<spatial {
//            globalMask[i] = detMasks[primaryIdx][i]
//        }
//        
//        var used = [Bool](repeating: false, count: allDetections.count)
//        used[primaryIdx] = true
//        
//        if self.debugMode {
//            print("🔷 Primary: \(allDetections[primaryIdx].className) @ \(Int(allDetections[primaryIdx].confidence * 100))%, area=\(detAreas[primaryIdx])px")
//        }
//        
//        // Iteratively add masks with >= 50% overlap
//        var changed = true
//        while changed {
//            changed = false
//            
//            for (idx, detMask) in detMasks.enumerated() {
//                if used[idx] || detAreas[idx] == 0 { continue }
//                
//                // Calculate overlap with globalMask
//                var overlapCount = 0
//                for i in 0..<spatial {
//                    if detMask[i] > 0 && globalMask[i] > 0 {
//                        overlapCount += 1
//                    }
//                }
//                
//                let overlapRatio = Float(overlapCount) / Float(detAreas[idx])
//                
//                if overlapRatio >= minOverlap {
//                    // Merge
//                    var added = 0
//                    for i in 0..<spatial {
//                        if detMask[i] > 0 && globalMask[i] == 0 {
//                            globalMask[i] = 1.0
//                            added += 1
//                        }
//                    }
//                    used[idx] = true
//                    changed = true
//                    
//                    if self.debugMode {
//                        print("🔗 Merged \(allDetections[idx].className) @ \(Int(allDetections[idx].confidence * 100))%: overlap=\(Int(overlapRatio * 100))%, +\(added)px")
//                    }
//                }
//            }
//        }
//        
//        if self.debugMode {
//            let mergedCount = used.filter { $0 }.count
//            var totalArea = 0
//            for i in 0..<spatial { if globalMask[i] > 0 { totalArea += 1 } }
//            print("🔷 buildGlobalMask: \(mergedCount)/\(allDetections.count) merged, total=\(totalArea)px")
//        }
//    }

    // MARK: - Fill Inside Perimeter
    private func fillInsidePerimeter(_ mask: inout [Float], width: Int, height: Int) {
        let count = width * height
        
        // Step 1: Dilate to seal perimeter gaps
        var sealed = [Float](repeating: 0, count: count)
        let radius = 3
        
        for y in 0..<height {
            for x in 0..<width {
                if mask[y * width + x] > 0 {
                    for dy in -radius...radius {
                        for dx in -radius...radius {
                            let nx = x + dx, ny = y + dy
                            if nx >= 0 && nx < width && ny >= 0 && ny < height {
                                sealed[ny * width + nx] = 1.0
                            }
                        }
                    }
                }
            }
        }
        
        // Step 2: Flood fill from edges to mark EXTERIOR
        var exterior = [Bool](repeating: false, count: count)
        var queue = [Int]()
        queue.reserveCapacity(count / 4)
        
        for x in 0..<width {
            if sealed[x] == 0 { queue.append(x); exterior[x] = true }
            let b = (height - 1) * width + x
            if sealed[b] == 0 && !exterior[b] { queue.append(b); exterior[b] = true }
        }
        for y in 0..<height {
            let l = y * width
            let r = y * width + width - 1
            if sealed[l] == 0 && !exterior[l] { queue.append(l); exterior[l] = true }
            if sealed[r] == 0 && !exterior[r] { queue.append(r); exterior[r] = true }
        }
        
        var head = 0
        while head < queue.count {
            let idx = queue[head]; head += 1
            let x = idx % width, y = idx / width
            
            let neighbors = [idx - 1, idx + 1, idx - width, idx + width]
            let valid = [x > 0, x < width - 1, y > 0, y < height - 1]
            
            for i in 0..<4 {
                if valid[i] {
                    let n = neighbors[i]
                    if sealed[n] == 0 && !exterior[n] {
                        exterior[n] = true
                        queue.append(n)
                    }
                }
            }
        }
        
        // Step 3: NOT exterior = interior = fill with 1
        var filled = 0
        for i in 0..<count {
            if !exterior[i] {
                if mask[i] == 0 { filled += 1 }
                mask[i] = 1.0
            }
        }
        
        if self.debugMode {
            print("🔷 fillInsidePerimeter: \(filled)px holes filled")
        }
    }

    // MARK: - Debug: Save Mask to Photos
    private func saveMaskToPhotos(_ mask: [Float], width: Int, height: Int, label: String = "mask") {
        let count = width * height
        
        var pixels = [UInt8](repeating: 0, count: count * 4)
        
        for i in 0..<count {
            let val: UInt8 = mask[i] > 0 ? 255 : 0
            pixels[i * 4 + 0] = val  // R
            pixels[i * 4 + 1] = val  // G
            pixels[i * 4 + 2] = val  // B
            pixels[i * 4 + 3] = 255  // A
        }
        
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else {
            print("❌ Failed to create data provider")
            return
        }
        
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            print("❌ Failed to create CGImage")
            return
        }
        
        let uiImage = UIImage(cgImage: cgImage)
        UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil)
        print("📸 Saved \(label) (\(width)x\(height)) to Photos")
    }

    // MARK: - Draw Perimeter Outline (debug)
//    private func drawPerimeterOutline(on pixels: UnsafeMutablePointer<UInt8>,
//                                       mask: [Float],
//                                       maskWidth: Int,
//                                       maskHeight: Int,
//                                       imageWidth: Int,
//                                       imageHeight: Int) {
//        let scaleX = Float(imageWidth) / Float(maskWidth)
//        let scaleY = Float(imageHeight) / Float(maskHeight)
//        
//        for my in 0..<maskHeight {
//            for mx in 0..<maskWidth {
//                let idx = my * maskWidth + mx
//                if mask[idx] > 0 {
//                    var isEdge = false
//                    if mx == 0 || mx == maskWidth - 1 || my == 0 || my == maskHeight - 1 {
//                        isEdge = true
//                    } else {
//                        if mask[idx - 1] == 0 || mask[idx + 1] == 0 ||
//                           mask[idx - maskWidth] == 0 || mask[idx + maskWidth] == 0 {
//                            isEdge = true
//                        }
//                    }
//                    
//                    if isEdge {
//                        let imgX = Int(Float(mx) * scaleX)
//                        let imgY = Int(Float(my) * scaleY)
//                        
//                        for dy in -1...1 {
//                            for dx in -1...1 {
//                                let px = imgX + dx
//                                let py = imgY + dy
//                                if px >= 0 && px < imageWidth && py >= 0 && py < imageHeight {
//                                    let pixelIdx = (py * imageWidth + px) * 4
//                                    pixels[pixelIdx + 0] = 0     // R
//                                    pixels[pixelIdx + 1] = 255   // G
//                                    pixels[pixelIdx + 2] = 255   // B
//                                    pixels[pixelIdx + 3] = 255   // A
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
    
    private func fillGlobalMaskInterior(_ mask: inout [Float], width: Int, height: Int) {
        let count = width * height
        
        // Find bounds of current globalMask
        var minX = width, maxX = 0, minY = height, maxY = 0
        for y in 0..<height {
            for x in 0..<width {
                if mask[y * width + x] > 0 {
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y)
                    maxY = max(maxY, y)
                }
            }
        }
        
        guard minX < maxX && minY < maxY else {
            if self.debugMode { print("⚠️ Empty mask") }
            return
        }
        
        if self.debugMode {
            var beforeCount = 0
            for i in 0..<count { if mask[i] > 0 { beforeCount += 1 } }
            print("🔷 globalMask bounds: (\(minX),\(minY)) → (\(maxX),\(maxY)), area=\(beforeCount)px")
        }
        
        // Dilate to seal perimeter gaps
//        let radius = 4
//        var dilated = [Float](repeating: 0, count: count)
//        
//        for y in minY...maxY {
//            for x in minX...maxX {
//                if mask[y * width + x] > 0 {
//                    for dy in -radius...radius {
//                        for dx in -radius...radius {
//                            let nx = x + dx
//                            let ny = y + dy
//                            if nx >= 0 && nx < width && ny >= 0 && ny < height {
//                                dilated[ny * width + nx] = 1.0
//                            }
//                        }
//                    }
//                }
//            }
//        }
        
        // Flood fill exterior on dilated
        var exterior = [Bool](repeating: false, count: count)
        var queue = [Int](repeating: 0, count: count)
        var head = 0, tail = 0
        
//        // Seed edges
//        for x in 0..<width {
//            if dilated[x] == 0 { exterior[x] = true; queue[tail] = x; tail += 1 }
//            let bot = (height - 1) * width + x
//            if dilated[bot] == 0 && !exterior[bot] { exterior[bot] = true; queue[tail] = bot; tail += 1 }
//        }
//        for y in 0..<height {
//            let left = y * width
//            let right = y * width + width - 1
//            if dilated[left] == 0 && !exterior[left] { exterior[left] = true; queue[tail] = left; tail += 1 }
//            if dilated[right] == 0 && !exterior[right] { exterior[right] = true; queue[tail] = right; tail += 1 }
//        }
        
//        while head < tail {
//            let idx = queue[head]; head += 1
//            let x = idx % width, y = idx / width
//            
//            if x > 0 { let n = idx - 1; if dilated[n] == 0 && !exterior[n] { exterior[n] = true; queue[tail] = n; tail += 1 } }
//            if x < width - 1 { let n = idx + 1; if dilated[n] == 0 && !exterior[n] { exterior[n] = true; queue[tail] = n; tail += 1 } }
//            if y > 0 { let n = idx - width; if dilated[n] == 0 && !exterior[n] { exterior[n] = true; queue[tail] = n; tail += 1 } }
//            if y < height - 1 { let n = idx + width; if dilated[n] == 0 && !exterior[n] { exterior[n] = true; queue[tail] = n; tail += 1 } }
//        }
        
        // Fill: anything inside bounds AND not exterior = fill it
//        var filledCount = 0
//        for y in minY...maxY {
//            for x in minX...maxX {
//                let idx = y * width + x
//                if !exterior[idx] && mask[idx] == 0 {
//                    mask[idx] = 1.0
//                    filledCount += 1
//                }
//            }
//        }
        
//        if self.debugMode {
//            var finalCount = 0
//            for i in 0..<count { if mask[i] > 0 { finalCount += 1 } }
//            print("🔷 fillGlobalMaskInterior: filled \(filledCount) holes → \(finalCount)px total")
//        }
    }
    
    private func useLargestMaskAndFill(
        globalMask: inout [Float],
        allDetections: [DetectionSmarty],
        protoMatrix: [Float],
        primaryBBox: DetectionSmarty,
        C: Int, Wp: Int, Hp: Int
    ) {
        let spatial = Wp * Hp
        let scale = Float(Wp) / 640.0
        
        // Primary bbox bounds
        let bboxX1 = max(0, Int((primaryBBox.x - primaryBBox.width / 2) * scale))
        let bboxY1 = max(0, Int((primaryBBox.y - primaryBBox.height / 2) * scale))
        let bboxX2 = min(Wp, Int((primaryBBox.x + primaryBBox.width / 2) * scale))
        let bboxY2 = min(Hp, Int((primaryBBox.y + primaryBBox.height / 2) * scale))
        
        var largestMask: [Float]? = nil
        var largestArea = 0
        var largestName = ""
        var largestConf = 0
        
        // Find mask with biggest area
        for det in allDetections {
            guard det.maskCoeffs.count == C else { continue }
            guard !det.maskCoeffs.contains(where: { !$0.isFinite }) else { continue }
            
            var rawMask = [Float](repeating: 0, count: spatial)
            vDSP_mmul(det.maskCoeffs, 1,
                      protoMatrix, 1,
                      &rawMask, 1,
                      1, vDSP_Length(spatial), vDSP_Length(C))
            
            // Threshold and clip to bbox, count area
            var area = 0
            for y in bboxY1..<bboxY2 {
                for x in bboxX1..<bboxX2 {
                    let idx = y * Wp + x
                    if rawMask[idx] > maskThreshold {
                        rawMask[idx] = 1.0
                        area += 1
                    } else {
                        rawMask[idx] = 0.0
                    }
                }
            }
            
            // Zero outside bbox
            for y in 0..<Hp {
                for x in 0..<Wp {
                    if x < bboxX1 || x >= bboxX2 || y < bboxY1 || y >= bboxY2 {
                        rawMask[y * Wp + x] = 0.0
                    }
                }
            }
            
            if area > largestArea {
                largestArea = area
                largestMask = rawMask
                largestName = det.className
                largestConf = Int(det.confidence * 100)
            }
        }
        
        guard let bestMask = largestMask else {
            if self.debugMode { print("⚠️ No valid mask found") }
            return
        }
        
        if self.debugMode {
            print("🏆 Largest mask: \(largestName) @ \(largestConf)% → \(largestArea)px")
        }
        
        // Copy largest mask to globalMask
        for i in 0..<spatial {
            globalMask[i] = bestMask[i]
        }
        
        // Fill interior using flood from edges
        fillMaskInterior(&globalMask, width: Wp, height: Hp)
        
        if self.debugMode {
            var finalCount = 0
            for i in 0..<spatial { if globalMask[i] > 0 { finalCount += 1 } }
            print("🔷 After fill: \(finalCount)px")
        }
    }
    
    private func sealAndFillInterior(_ mask: inout [Float], width: Int, height: Int) {
        let count = width * height
        
        // Step 1: Dilate to close gaps
        let dilateRadius = 3
        var dilated = [Float](repeating: 0, count: count)
        
        for y in 0..<height {
            for x in 0..<width {
                if mask[y * width + x] > 0 {
                    for dy in -dilateRadius...dilateRadius {
                        for dx in -dilateRadius...dilateRadius {
                            let nx = x + dx
                            let ny = y + dy
                            if nx >= 0 && nx < width && ny >= 0 && ny < height {
                                dilated[ny * width + nx] = 1.0
                            }
                        }
                    }
                }
            }
        }
        
        // Step 2: Flood fill exterior on dilated mask
        var exterior = [Bool](repeating: false, count: count)
        var queue = [Int](repeating: 0, count: count)
        var head = 0, tail = 0
        
        for x in 0..<width {
            if dilated[x] == 0 { exterior[x] = true; queue[tail] = x; tail += 1 }
            let bot = (height - 1) * width + x
            if dilated[bot] == 0 && !exterior[bot] { exterior[bot] = true; queue[tail] = bot; tail += 1 }
        }
        for y in 0..<height {
            let left = y * width
            let right = y * width + width - 1
            if dilated[left] == 0 && !exterior[left] { exterior[left] = true; queue[tail] = left; tail += 1 }
            if dilated[right] == 0 && !exterior[right] { exterior[right] = true; queue[tail] = right; tail += 1 }
        }
        
        while head < tail {
            let idx = queue[head]; head += 1
            let x = idx % width
            let y = idx / width
            
            if x > 0 { let n = idx - 1; if dilated[n] == 0 && !exterior[n] { exterior[n] = true; queue[tail] = n; tail += 1 } }
            if x < width - 1 { let n = idx + 1; if dilated[n] == 0 && !exterior[n] { exterior[n] = true; queue[tail] = n; tail += 1 } }
            if y > 0 { let n = idx - width; if dilated[n] == 0 && !exterior[n] { exterior[n] = true; queue[tail] = n; tail += 1 } }
            if y < height - 1 { let n = idx + width; if dilated[n] == 0 && !exterior[n] { exterior[n] = true; queue[tail] = n; tail += 1 } }
        }
        
        // Step 3: Interior = not exterior. BUT only fill within original mask's convex bounds
        // Find bounds of original mask
        var minX = width, maxX = 0, minY = height, maxY = 0
        for y in 0..<height {
            for x in 0..<width {
                if mask[y * width + x] > 0 {
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y)
                    maxY = max(maxY, y)
                }
            }
        }
        
        // Step 4: Fill interior pixels that are within original mask bounds
        var filledCount = 0
        for y in minY...maxY {
            for x in minX...maxX {
                let idx = y * width + x
                if !exterior[idx] && mask[idx] == 0 {
                    mask[idx] = 1.0
                    filledCount += 1
                }
            }
        }
        
        if self.debugMode {
            var finalCount = 0
            for i in 0..<count { if mask[i] > 0 { finalCount += 1 } }
            print("🔷 sealAndFillInterior: filled \(filledCount) holes, final=\(finalCount)px")
        }
    }
    
    private func fillEntireInterior(_ mask: inout [Float], width: Int, height: Int) {
        // Step 1: Horizontal fill - each row, fill from leftmost to rightmost
        for y in 0..<height {
            var minX = -1
            var maxX = -1
            for x in 0..<width {
                if mask[y * width + x] > 0 {
                    if minX < 0 { minX = x }
                    maxX = x
                }
            }
            if minX >= 0 && maxX > minX {
                for x in minX...maxX {
                    mask[y * width + x] = 1.0
                }
            }
        }
        
        // Step 2: Vertical fill - each column, fill from topmost to bottommost
        for x in 0..<width {
            var minY = -1
            var maxY = -1
            for y in 0..<height {
                if mask[y * width + x] > 0 {
                    if minY < 0 { minY = y }
                    maxY = y
                }
            }
            if minY >= 0 && maxY > minY {
                for y in minY...maxY {
                    mask[y * width + x] = 1.0
                }
            }
        }
        
        if self.debugMode {
            var finalCount = 0
            for i in 0..<(width * height) { if mask[i] > 0 { finalCount += 1 } }
            print("🔷 fillEntireInterior: \(finalCount)px")
        }
    }
    
    // Close 1px gaps in perimeter before flood fill
    private func closePerimeterGaps(_ mask: inout [Float], width: Int, height: Int) {
        let count = width * height
        var closed = [Float](repeating: 0, count: count)
        
        // Copy original
        for i in 0..<count { closed[i] = mask[i] }
        
        // For each empty pixel, if it has mask neighbors on opposite sides, fill it
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                let idx = y * width + x
                if mask[idx] == 0 {
                    let left = mask[idx - 1] > 0
                    let right = mask[idx + 1] > 0
                    let up = mask[idx - width] > 0
                    let down = mask[idx + width] > 0
                    
                    // Horizontal gap (left AND right are filled)
                    // Vertical gap (up AND down are filled)
                    // Diagonal consideration
                    if (left && right) || (up && down) {
                        closed[idx] = 1.0
                    }
                }
            }
        }
        
        mask = closed
        
        if self.debugMode {
            var closedCount = 0
            for i in 0..<count { if mask[i] > 0 { closedCount += 1 } }
            print("🔒 closePerimeterGaps: \(closedCount)px")
        }
    }
    
    private func buildConnectedMask(
        globalMask: inout [Float],
        allDetections: [DetectionSmarty],
        protoMatrix: [Float],
        primaryBBox: DetectionSmarty,
        C: Int, Wp: Int, Hp: Int
    ) {
        let spatial = Wp * Hp
        let scale = Float(Wp) / 640.0
        
        // Primary bbox bounds in mask coords
        let bboxX1 = max(0, Int((primaryBBox.x - primaryBBox.width / 2) * scale))
        let bboxY1 = max(0, Int((primaryBBox.y - primaryBBox.height / 2) * scale))
        let bboxX2 = min(Wp, Int((primaryBBox.x + primaryBBox.width / 2) * scale))
        let bboxY2 = min(Hp, Int((primaryBBox.y + primaryBBox.height / 2) * scale))
        
        if self.debugMode {
            print("🔲 Primary BBox in mask coords: (\(bboxX1),\(bboxY1)) → (\(bboxX2),\(bboxY2))")
        }
        
        // =====================================================
        // 1) Precompute a bbox mask (Float) for proto space
        //    1.0 inside primary bbox, 0.0 outside.
        //    This lets us use vDSP_vmul to clip each det mask
        //    to the bbox in one shot.
        // =====================================================
        var bboxMask = [Float](repeating: 0.0, count: spatial)
        if bboxX2 > bboxX1 && bboxY2 > bboxY1 {
            for y in bboxY1..<bboxY2 {
                let rowBase = y * Wp
                for x in bboxX1..<bboxX2 {
                    bboxMask[rowBase + x] = 1.0
                }
            }
        }
        
        // =====================================================
        // 2) Pre-compute all detection masks in proto space,
        //    clipped to primary bbox, binary 0/1
        // =====================================================
        var detMasks = [[Float]]()
        detMasks.reserveCapacity(allDetections.count)
        
        for det in allDetections {
            var rawMask = [Float](repeating: 0.0, count: spatial)
            
            // Guard on coeff shape and finiteness (same semantics)
            guard det.maskCoeffs.count == C else {
                detMasks.append(rawMask)
                continue
            }
            let hasInvalid = det.maskCoeffs.contains { !$0.isFinite }
            guard !hasInvalid else {
                detMasks.append(rawMask)
                continue
            }
            
            // (1 × C) * (C × spatial) → (1 × spatial)
            det.maskCoeffs.withUnsafeBufferPointer { coeffPtr in
                protoMatrix.withUnsafeBufferPointer { protoPtr in
                    guard let coeffBase = coeffPtr.baseAddress,
                          let protoBase = protoPtr.baseAddress else { return }
                    vDSP_mmul(coeffBase, 1,
                              protoBase, 1,
                              &rawMask, 1,
                              1, vDSP_Length(spatial), vDSP_Length(C))
                }
            }
            
            // Clip to bbox via element-wise multiply
            // (everything outside bbox becomes 0)
            vDSP_vmul(rawMask, 1,
                      bboxMask, 1,
                      &rawMask, 1,
                      vDSP_Length(spatial))
            
            // Threshold to STRICT 0/1 (same as your loop logic)
            // rawMask[idx] = 1 if > maskThreshold (and inside bbox), else 0
            for i in 0..<spatial {
                rawMask[i] = (rawMask[i] > maskThreshold) ? 1.0 : 0.0
            }
            
            detMasks.append(rawMask)
        }
        
        // =====================================================
        // 3) Start with primary (highest confidence) detection
        //    globalMask = detMasks[primaryIdx]
        // =====================================================
        var used = [Bool](repeating: false, count: allDetections.count)
        
        if let primaryIdx = allDetections.indices.max(by: { allDetections[$0].confidence < allDetections[$1].confidence }) {
            if detMasks.indices.contains(primaryIdx) {
                // Copy detMasks[primaryIdx] → globalMask
                for i in 0..<spatial {
                    globalMask[i] = detMasks[primaryIdx][i]
                }
            } else {
                // Safety: zero if something is off
                globalMask = [Float](repeating: 0.0, count: spatial)
            }
            used[primaryIdx] = true
            
            if self.debugMode {
                var cnt = 0
                for v in globalMask where v > 0 { cnt += 1 }
                let det = allDetections[primaryIdx]
                print("🔷 Started with \(det.className) @ \(Int(det.confidence * 100))%: \(cnt)px")
            }
        } else {
            // No detections at all
            globalMask = [Float](repeating: 0.0, count: spatial)
            if self.debugMode {
                print("🔷 buildConnectedMask: no primary detection found")
            }
            return
        }
        
        // =====================================================
        // 4) Iteratively absorb touching detections
        //    TOUCHING definition is unchanged:
        //    detMask pixel inside bbox has any 8-neighbor in globalMask
        // =====================================================
        var changed = true
        var iteration = 0
        
        while changed {
            changed = false
            iteration += 1
            
            for (detIdx, detMask) in detMasks.enumerated() {
                if used[detIdx] { continue }
                
                // Touch check (unchanged logic)
                var touches = false
                outer: for y in bboxY1..<bboxY2 {
                    let rowBase = y * Wp
                    for x in bboxX1..<bboxX2 {
                        let idx = rowBase + x
                        if detMask[idx] > 0 {
                            // Check 8-neighbors in globalMask
                            for dy in -1...1 {
                                for dx in -1...1 {
                                    if dx == 0 && dy == 0 { continue }
                                    let nx = x + dx
                                    let ny = y + dy
                                    if nx >= 0, nx < Wp, ny >= 0, ny < Hp {
                                        if globalMask[ny * Wp + nx] > 0 {
                                            touches = true
                                            break outer
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                if touches {
                    // =================================================
                    // Absorb detection into globalMask:
                    //   newGlobal = max(globalMask, detMask)
                    //   added = pixels which changed 0 → 1
                    // =================================================
                    var newGlobal = [Float](repeating: 0.0, count: spatial)
                    vDSP_vmax(globalMask, 1,
                              detMask, 1,
                              &newGlobal, 1,
                              vDSP_Length(spatial))
                    
                    var added = 0
                    for i in 0..<spatial {
                        if newGlobal[i] > 0, globalMask[i] == 0 {
                            added += 1
                        }
                    }
                    
                    globalMask = newGlobal
                    used[detIdx] = true
                    changed = true
                    
                    if self.debugMode {
                        let det = allDetections[detIdx]
                        print("🔗 Iter \(iteration): absorbed \(det.className) @ \(Int(det.confidence * 100))%: +\(added)px")
                    }
                }
            }
        }
        
        // =====================================================
        // 5) Final debug stats (unchanged semantics)
        // =====================================================
        if self.debugMode {
            var finalCount = 0
            for v in globalMask where v > 0 { finalCount += 1 }
            let absorbedCount = used.filter { $0 }.count
            print("🔷 buildConnectedMask: \(absorbedCount)/\(allDetections.count) detections absorbed, \(finalCount)px total")
        }
    }

//    // MARK: - Debug: Save Mask to Photos
//    private func saveMaskToPhotos(_ mask: [Float], width: Int, height: Int, label: String = "mask") {
//        let count = width * height
//        
//        // Create grayscale image from mask
//        var pixels = [UInt8](repeating: 0, count: count * 4)
//        
//        for i in 0..<count {
//            let val: UInt8 = mask[i] > 0 ? 255 : 0
//            pixels[i * 4 + 0] = val  // R
//            pixels[i * 4 + 1] = val  // G
//            pixels[i * 4 + 2] = val  // B
//            pixels[i * 4 + 3] = 255  // A
//        }
//        
//        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else {
//            print("❌ Failed to create data provider")
//            return
//        }
//        
//        guard let cgImage = CGImage(
//            width: width,
//            height: height,
//            bitsPerComponent: 8,
//            bitsPerPixel: 32,
//            bytesPerRow: width * 4,
//            space: CGColorSpaceCreateDeviceRGB(),
//            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
//            provider: provider,
//            decode: nil,
//            shouldInterpolate: false,
//            intent: .defaultIntent
//        ) else {
//            print("❌ Failed to create CGImage")
//            return
//        }
//        
//        let uiImage = UIImage(cgImage: cgImage)
//        
//        UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil)
//        print("📸 Saved \(label) (\(width)x\(height)) to Photos")
//    }

    // MARK: - Fill Mask Interior (flood from edges, invert)
//    private func fillMaskInterior(_ mask: inout [Float], width: Int, height: Int) {
//        let count = width * height
//        guard count > 0 else { return }
//        
//        // exterior[i] = true means pixel is outside (reachable from edge)
//        var exterior = [Bool](repeating: false, count: count)
//        
//        // Queue for BFS (using array index, not append/remove)
//        var queue = [Int](repeating: 0, count: count)
//        var head = 0
//        var tail = 0
//        
//        // Seed: all edge pixels that are background
//        for x in 0..<width {
//            let topIdx = x
//            let botIdx = (height - 1) * width + x
//            if mask[topIdx] == 0 && !exterior[topIdx] {
//                exterior[topIdx] = true
//                queue[tail] = topIdx; tail += 1
//            }
//            if mask[botIdx] == 0 && !exterior[botIdx] {
//                exterior[botIdx] = true
//                queue[tail] = botIdx; tail += 1
//            }
//        }
//        for y in 0..<height {
//            let leftIdx = y * width
//            let rightIdx = y * width + width - 1
//            if mask[leftIdx] == 0 && !exterior[leftIdx] {
//                exterior[leftIdx] = true
//                queue[tail] = leftIdx; tail += 1
//            }
//            if mask[rightIdx] == 0 && !exterior[rightIdx] {
//                exterior[rightIdx] = true
//                queue[tail] = rightIdx; tail += 1
//            }
//        }
//        
//        // BFS flood fill
//        while head < tail {
//            let idx = queue[head]
//            head += 1
//            
//            let x = idx % width
//            let y = idx / width
//            
//            // 4-connectivity neighbors
//            if x > 0 {
//                let n = idx - 1
//                if mask[n] == 0 && !exterior[n] {
//                    exterior[n] = true
//                    queue[tail] = n; tail += 1
//                }
//            }
//            if x < width - 1 {
//                let n = idx + 1
//                if mask[n] == 0 && !exterior[n] {
//                    exterior[n] = true
//                    queue[tail] = n; tail += 1
//                }
//            }
//            if y > 0 {
//                let n = idx - width
//                if mask[n] == 0 && !exterior[n] {
//                    exterior[n] = true
//                    queue[tail] = n; tail += 1
//                }
//            }
//            if y < height - 1 {
//                let n = idx + width
//                if mask[n] == 0 && !exterior[n] {
//                    exterior[n] = true
//                    queue[tail] = n; tail += 1
//                }
//            }
//        }
//        
//        // Invert: not exterior = interior = fill
//        for i in 0..<count {
//            mask[i] = exterior[i] ? 0.0 : 1.0
//        }
//    }

    // ======================================================
    // SAFE LABEL + BOX DRAWING (CYAN, BIG FONT, FIXED FLIP)
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

    // ======================================================
    // PERIMETER LINES DRAWING - Shows live perimeter tracking
    // ======================================================
    private func drawPerimeterLines(
        ctx: CGContext,
        imageWidth: Int,
        imageHeight: Int,
        maskWidth: Int,
        maskHeight: Int
    ) {
        guard let perimeterMask = bestPerimeterMask else { 
            if debugMode { print("🔴 [PERIMETER DRAW] No perimeter mask to draw") }
            return 
        }

        if debugMode { 
            print("🔴 [PERIMETER DRAW] Drawing perimeter with area: \(bestPerimeterArea)px") 
            print("🔴 [PERIMETER DRAW] Mask dimensions: \(maskWidth)x\(maskHeight) → Image: \(imageWidth)x\(imageHeight)")
        }

        ctx.saveGState()
        
        // Fix UIKit upside-down drawing in CGContexts - SAME AS LABELS
        let W = CGFloat(imageWidth)
        let H = CGFloat(imageHeight)
        ctx.translateBy(x: 0, y: H)
        ctx.scaleBy(x: 1, y: -1)
        
        // Set perimeter line style - bright magenta/pink for visibility
        ctx.setStrokeColor(CGColor(red: 1.0, green: 0.2, blue: 0.8, alpha: 0.8))
        ctx.setLineWidth(2.0)
        ctx.setLineDash(phase: 0, lengths: [8, 4]) // Dashed line for distinction
        
        let scaleX = CGFloat(imageWidth) / CGFloat(maskWidth)
        let scaleY = CGFloat(imageHeight) / CGFloat(maskHeight)
        
        // Draw perimeter contour by connecting edge pixels
        var perimeterPoints: [(Int, Int)] = []
        
        // Find all perimeter pixels
        for y in 0..<maskHeight {
            for x in 0..<maskWidth {
                let idx = y * maskWidth + x
                if perimeterMask[idx] > 0 {
                    // Check if this is an edge pixel (has at least one background neighbor)
                    var isEdge = false
                    
                    // Check 8-connected neighbors
                    for dy in -1...1 {
                        for dx in -1...1 {
                            if dx == 0 && dy == 0 { continue }
                            let nx = x + dx
                            let ny = y + dy
                            
                            if nx < 0 || nx >= maskWidth || ny < 0 || ny >= maskHeight {
                                isEdge = true
                                break
                            }
                            
                            let nidx = ny * maskWidth + nx
                            if perimeterMask[nidx] == 0 {
                                isEdge = true
                                break
                            }
                        }
                        if isEdge { break }
                    }
                    
                    if isEdge {
                        perimeterPoints.append((x, y))
                    }
                }
            }
        }
        
        if debugMode { 
            print("🔴 [PERIMETER DRAW] Found \(perimeterPoints.count) edge pixels") 
        }
        
        // Draw perimeter points as connected lines
        if perimeterPoints.count > 1 {
            // Sort points to create a more coherent outline (simple row-major order)
            perimeterPoints.sort { 
                if $0.1 == $1.1 { return $0.0 < $1.0 }
                return $0.1 < $1.1 
            }
            
            // Draw connected segments
            var currentPath: [(CGFloat, CGFloat)] = []
            
            for (i, point) in perimeterPoints.enumerated() {
                let imageX = CGFloat(point.0) * scaleX
                let imageY = CGFloat(point.1) * scaleY
                
                if i == 0 {
                    currentPath.append((imageX, imageY))
                    ctx.move(to: CGPoint(x: imageX, y: imageY))
                } else {
                    let prevPoint = perimeterPoints[i-1]
                    let prevImageX = CGFloat(prevPoint.0) * scaleX
                    let prevImageY = CGFloat(prevPoint.1) * scaleY
                    
                    // If points are close, connect them
                    let distance = sqrt(pow(imageX - prevImageX, 2) + pow(imageY - prevImageY, 2))
                    if distance < 50 { // Threshold for connecting points
                        ctx.addLine(to: CGPoint(x: imageX, y: imageY))
                        currentPath.append((imageX, imageY))
                    } else {
                        // Start new path segment
                        if currentPath.count > 1 {
                            ctx.strokePath()
                        }
                        ctx.move(to: CGPoint(x: imageX, y: imageY))
                        currentPath = [(imageX, imageY)]
                    }
                }
            }
            
            // Stroke the final path
            if currentPath.count > 1 {
                ctx.strokePath()
            }
            
            if debugMode { 
                print("🔴 [PERIMETER DRAW] Drew perimeter outline with \(currentPath.count) connected points") 
            }
        }
        
        // Also draw some key perimeter points as small circles for better visibility
        ctx.setLineDash(phase: 0, lengths: []) // Remove dash for circles
        ctx.setFillColor(CGColor(red: 1.0, green: 0.0, blue: 0.5, alpha: 0.6))
        
        let circleRadius: CGFloat = 3.0
        let stepSize = max(1, perimeterPoints.count / 20) // Draw max 20 circles
        
        for i in stride(from: 0, to: perimeterPoints.count, by: stepSize) {
            let point = perimeterPoints[i]
            let imageX = CGFloat(point.0) * scaleX
            let imageY = CGFloat(point.1) * scaleY
            
            let circle = CGRect(
                x: imageX - circleRadius, 
                y: imageY - circleRadius,
                width: circleRadius * 2, 
                height: circleRadius * 2
            )
            ctx.fillEllipse(in: circle)
        }
        
        ctx.restoreGState()
        
        if debugMode { 
            print("🔴 [PERIMETER DRAW] Completed perimeter visualization") 
        }
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


    private func extractDetections(from detections: MLMultiArray) -> [DetectionSmarty] {
        let t0 = Date()
        var all: [DetectionSmarty] = []

        // Shape: [1, features, anchors]
        guard detections.shape.count == 3 else {
            if self.debugMode { print("⚠️ extractDetections: Unexpected tensor rank: \(detections.shape)") }
            return []
        }

        let numFeatures = detections.shape[1].intValue
        let numAnchors  = detections.shape[2].intValue
        let numClasses  = numFeatures - 4 - 32

        // Validate tensor dimensions
        guard numFeatures >= 36, numAnchors > 0, numClasses > 0 else {
            if self.debugMode {
                print("⚠️ extractDetections: Invalid tensor dims — features:\(numFeatures) anchors:\(numAnchors) classes:\(numClasses)")
            }
            return []
        }

        let expectedCount = numFeatures * numAnchors
        let totalCount = detections.count
        guard totalCount >= expectedCount else {
            if self.debugMode { print("⚠️ extractDetections: count mismatch expected=\(expectedCount) got=\(totalCount)") }
            return []
        }

        // Copy/convert to Float buffer safely
        let detBuf = UnsafeMutablePointer<Float>.allocate(capacity: totalCount)
        defer { detBuf.deallocate() }

        let copyStart = Date()
        switch detections.dataType {
        case .float32:
            let src = detections.dataPointer.assumingMemoryBound(to: Float.self)
            memcpy(detBuf, src, totalCount * MemoryLayout<Float>.size)
        case .float16:
            let src = detections.dataPointer.bindMemory(to: UInt16.self, capacity: totalCount)
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src),
                                       height: 1, width: vImagePixelCount(totalCount),
                                       rowBytes: totalCount * MemoryLayout<UInt16>.size)
            var dstBuf = vImage_Buffer(data: UnsafeMutableRawPointer(detBuf),
                                       height: 1, width: vImagePixelCount(totalCount),
                                       rowBytes: totalCount * MemoryLayout<Float>.size)
            let result = vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
            if result != kvImageNoError && self.debugMode { print("⚠️ extractDetections: vImage 16F→F failed: \(result)") }
        default:
            // Fallback: element access
            for i in 0..<totalCount { detBuf[i] = detections[i].floatValue }
        }
        let copyEnd = Date()
        if self.debugMode {
            print(String(format: "⏱ extractDetections copy/convert: %.2f ms", copyEnd.timeIntervalSince(copyStart) * 1000.0))
        }

        let stride = numAnchors
        let coeffOffset = 4 + numClasses

        if self.debugMode {
            print("📝 Tensor shape: [1, \(numFeatures), \(numAnchors)]")
            print("   → \(numClasses) classes, \(numAnchors) predictions")
            print("   → Mode: CLASS-AGNOSTIC")
        }

        let decodeStart = Date()
        for anchor in 0..<numAnchors {
            // Bounds check for bbox
            guard anchor < stride,
                  1 * stride + anchor < totalCount,
                  2 * stride + anchor < totalCount,
                  3 * stride + anchor < totalCount else {
                if self.debugMode { print("⚠️ bbox OOB for anchor \(anchor)") }
                continue
            }

            let x = detBuf[0 * stride + anchor]
            let y = detBuf[1 * stride + anchor]
            let w = detBuf[2 * stride + anchor]
            let h = detBuf[3 * stride + anchor]
            guard x.isFinite, y.isFinite, w.isFinite, h.isFinite, w > 0, h > 0 else { continue }

            // Class-agnostic: take max over class scores
            var bestConf: Float = 0
            let baseConfIdx = 4 * stride + anchor
            for classIdx in 0..<numClasses {
                let confIndex = baseConfIdx + classIdx * stride
                guard confIndex < totalCount else {
                    if self.debugMode { print("⚠️ conf OOB class=\(classIdx) anchor=\(anchor)") }
                    break
                }
                let conf = detBuf[confIndex]
                if conf.isFinite, conf > bestConf { bestConf = conf }
            }
            guard bestConf > confidenceThreshold else { continue }

            // Mask coefficients (32)
            var coeffs = [Float](repeating: 0, count: 32)
            let coeffStartIdx = coeffOffset * stride + anchor
            var coeffsOK = true
            for k in 0..<32 {
                let idx = coeffStartIdx + k * stride
                if idx < totalCount {
                    coeffs[k] = detBuf[idx]
                } else {
                    if self.debugMode { print("⚠️ coeff OOB k=\(k) anchor=\(anchor)") }
                    coeffsOK = false
                    break
                }
            }
            if !coeffsOK { continue }

            all.append(DetectionSmarty(
                x: x, y: y, width: w, height: h,
                confidence: bestConf,
                classIdx: -1,
                className: "object",
                maskCoeffs: coeffs
            ))
        }
        let decodeEnd = Date()

        if self.debugMode {
            print(String(format: "⏱ extractDetections decode loop: %.2f ms", decodeEnd.timeIntervalSince(decodeStart) * 1000.0))
            print("\n📊 DETECTION SUMMARY: \(all.count) total")
            let grouped = Dictionary(grouping: all) { $0.className }
            for (className, dets) in grouped.sorted(by: { $0.value.count > $1.value.count }).prefix(20) {
                let confidences = dets.map { Int($0.confidence * 100) }
                print("  - \(className): \(dets.count)x, conf: \(confidences)%")
            }
            if grouped.count > 20 { print("  ... and \(grouped.count - 20) more classes") }
            let tEnd = Date()
            print(String(format: "⏱ extractDetections total: %.2f ms", tEnd.timeIntervalSince(t0) * 1000.0))
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

            // Safe copy into Swift array storage
            rowUInt8.withUnsafeMutableBytes { dstBytes in
                memcpy(dstBytes.baseAddress!, rowStart, width * 4)
            }

            // Convert to Float and scale
            rowUInt8.withUnsafeBufferPointer { u8Ptr in
                rowFloat.withUnsafeMutableBufferPointer { fPtr in
                    vDSP_vfltu8(u8Ptr.baseAddress!, 1, fPtr.baseAddress!, 1, vDSP_Length(width * 4))
                    vDSP_vsmul(fPtr.baseAddress!, 1, &scaleF, fPtr.baseAddress!, 1, vDSP_Length(width * 4))
                }
            }

            // Gather R/G/B into channel planes
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
