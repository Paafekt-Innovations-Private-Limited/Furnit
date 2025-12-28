import AVFoundation

final class VideoFrameFeeder {
    private let url: URL
    private var asset: AVAsset
    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?

    private let queue = DispatchQueue(label: "video.frame.feeder", qos: .userInitiated)

    // control playback speed for testing
    var targetFPS: Double = 10
    private var timer: DispatchSourceTimer?

    var onFrame: ((CVPixelBuffer) -> Void)?

    init(url: URL) {
        self.url = url
        self.asset = AVAsset(url: url)
    }

    func start(loop: Bool = true) {
        stop()

        queue.async { [weak self] in
            guard let self else { return }
            self.setupReader()

            let intervalNs = UInt64((1.0 / max(self.targetFPS, 1)) * 1_000_000_000.0)
            let t = DispatchSource.makeTimerSource(queue: self.queue)
            t.schedule(deadline: .now(), repeating: .nanoseconds(Int(intervalNs)))
            t.setEventHandler { [weak self] in
                self?.tick(loop: loop)
            }
            self.timer = t
            t.resume()
        }
    }

    func stop() {
        queue.sync {
            timer?.cancel()
            timer = nil
            reader?.cancelReading()
            reader = nil
            output = nil
        }
    }

    private func setupReader() {
        reader?.cancelReading()
        reader = nil
        output = nil

        let asset = self.asset
        guard let track = asset.tracks(withMediaType: .video).first else { return }

        do {
            let r = try AVAssetReader(asset: asset)

            // Pixel buffer format similar to your camera output (BGRA)
            let settings: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]

            let out = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
            out.alwaysCopiesSampleData = false

            guard r.canAdd(out) else { return }
            r.add(out)

            self.reader = r
            self.output = out
            r.startReading()
        } catch {
            print("❌ AVAssetReader init failed:", error)
        }
    }

    private func tick(loop: Bool) {
        guard let reader, let output else { return }

        if reader.status == .failed || reader.status == .cancelled {
            return
        }

        if reader.status == .completed {
            if loop {
                setupReader()
            }
            return
        }

        guard let sbuf = output.copyNextSampleBuffer(),
              let pb = CMSampleBufferGetImageBuffer(sbuf) else {
            // End reached (sometimes status flips to completed next tick)
            return
        }

        onFrame?(pb)
    }
}
