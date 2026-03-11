import UIKit
#if canImport(onnxruntime_objc)
import onnxruntime_objc
#endif

/// Experimental ONNX Runtime-based YOLOE segmentation service for the new blue icon.
/// - Uses the same yoloe-11l-seg-pf model as Android (`yoloe-11l-seg-pf.onnx`).
/// - Kept completely separate from the existing CoreML FurnitureFit pipeline (brain icon path).
///
/// NOTE:
/// - This file compiles even when ONNX Runtime is not integrated; in that case `isAvailable` is false
///   and callers should fall back to the existing CoreML pipeline.
/// - To enable ONNX Runtime:
///     1. Run `pod install` in the repo root (Podfile adds `onnxruntime-objc`).
///     2. Open `Furnit.xcworkspace`.
///     3. Add `yoloe-11l-seg-pf.onnx` to the Furnit target (e.g. add a file reference that points to
///        `android/yoloe-11l-seg-pf.onnx` and is included in the app bundle).
final class OnnxFurnitureFitService {

    static let shared = OnnxFurnitureFitService()

    /// True when ONNX Runtime is available and the session is ready.
    var isAvailable: Bool {
        #if canImport(onnxruntime_objc)
        return session != nil
        #else
        return false
        #endif
    }

    // MARK: - Public API

    /// Ensure the ONNX model is loaded. Safe to call repeatedly.
    func ensureModelLoaded() {
        #if canImport(onnxruntime_objc)
        guard session == nil else { return }
        loadModel()
        #endif
    }

    /// Run ONNX-based segmentation and return a single-furniture cutout.
    /// Falls back to `nil` when ONNX Runtime is not available or inference fails.
    func segmentPrimaryFurniture(from frame: UIImage) -> UIImage? {
        #if canImport(onnxruntime_objc)
        guard let session = session else { return nil }
        return runOnnxSegmentation(frame: frame, session: session)
        #else
        return nil
        #endif
    }

    // MARK: - Internal (ONNX Runtime)

    #if canImport(onnxruntime_objc)
    private var env: ORTEnv?
    private var session: ORTSession?

    private func loadModel() {
        do {
            let env = try ORTEnv(loggingLevel: .warning)
            self.env = env

            guard let modelURL = Bundle.main.url(forResource: "yoloe-11l-seg-pf", withExtension: "onnx") else {
                logDebug("ONNX: yoloe-11l-seg-pf.onnx not found in bundle; add it to the Furnit target.")
                return
            }

            let options = try ORTSessionOptions()
            self.session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
            logDebug("ONNX: Session created for \(modelURL.lastPathComponent)")
        } catch {
            logDebug("ONNX: Failed to create session: \(error)")
            self.session = nil
        }
    }

    /// Rough Swift translation of `runOnnxInferenceWithDetections` in `FurnitureFitManager.kt`.
    /// This is intentionally scoped to **primary detection only** (one furniture).
    private func runOnnxSegmentation(frame: UIImage, session: ORTSession) -> UIImage? {
        guard let cgImage = frame.cgImage else { return nil }

        // Read input/output names. For simplicity (and to mirror the Android code path),
        // we assume a single input and two outputs (detections + prototypes).
        let inputName: String
        let outputNameSet: Set<String>
        do {
            let inputNames = try session.inputNames()
            guard let first = inputNames.first else { return nil }
            inputName = first
            let outputs = try session.outputNames()
            outputNameSet = Set(outputs)
        } catch {
            logDebug("ONNX: Failed to query model IO: \(error)")
            return nil
        }

        // For simplicity and parity with the Android implementation, use fixed 640x640 here.
        let inputH = 640
        let inputW = 640

        // Resize frame to model input size
        let resized = frame.resized(to: CGSize(width: inputW, height: inputH))
        guard let resizedCG = resized.cgImage else { return nil }

        let hw = inputH * inputW
        var inputFloats = [Float](repeating: 0, count: 1 * 3 * hw)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var intPixels = [UInt8](repeating: 0, count: inputW * inputH * 4)
        guard let ctx = CGContext(data: &intPixels,
                                  width: inputW,
                                  height: inputH,
                                  bitsPerComponent: 8,
                                  bytesPerRow: inputW * 4,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.draw(resizedCG, in: CGRect(x: 0, y: 0, width: inputW, height: inputH))

        // NHWC (RGB) -> NCHW float32 [1,3,H,W], normalized to [0,1]
        for y in 0..<inputH {
            for x in 0..<inputW {
                let idx = (y * inputW + x) * 4
                let r = Float(intPixels[idx + 0]) / 255.0
                let g = Float(intPixels[idx + 1]) / 255.0
                let b = Float(intPixels[idx + 2]) / 255.0
                let pixelIdx = y * inputW + x
                inputFloats[0 * hw + pixelIdx] = r
                inputFloats[1 * hw + pixelIdx] = g
                inputFloats[2 * hw + pixelIdx] = b
            }
        }

        do {
            let tensorData = NSMutableData(bytes: &inputFloats, length: inputFloats.count * MemoryLayout<Float>.size)
            let shape: [NSNumber] = [1, 3, NSNumber(value: inputH), NSNumber(value: inputW)]
            let inputValue = try ORTValue(tensorData: tensorData,
                                          elementType: .float,
                                          shape: shape)

            let outputs = try session.run(withInputs: [inputName: inputValue],
                                          outputNames: outputNameSet,
                                          runOptions: nil)

            // For now, we simply return nil here to keep the compile-time integration minimal.
            // A full port would:
            //  - Identify det/proto outputs (as in FurnitureFitManager.runOnnxInferenceWithDetections)
            //  - Decode detections, run NMS, pick primaryDet
            //  - Build maskProto over primary bbox from proto tensor
            //  - Upsample mask to frame size and composite into a transparent-background UIImage.
            logDebug("ONNX: session.run completed with outputs: \(outputs.keys)")
            return nil
        } catch {
            logDebug("ONNX: Inference failed: \(error)")
            return nil
        }
    }
    #endif
}

private extension UIImage {
    func resized(to size: CGSize) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        self.draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext() ?? self
    }
}

