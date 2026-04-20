import Foundation
import UIKit

enum FurnitureFitOneImageDebugSupport {
    /// Set to `true` to run one still image through the full composite path. Input is resolved by
    /// `resolvedInputURL()`; output is written under the app Documents directory (and best-effort
    /// to the Mac repo path on Simulator). Default `false` restores normal live segmentation.
    static let runEnabled = false

    /// Optional: copy `alchair.jpeg` into the app target (e.g. `Furnit/test_images/alchair.jpeg`)
    /// or into Documents — see `resolvedInputURL()`.
    private static let oneImageInputPathMacDev = "/Users/al/Documents/tries01/Furnit/test_images/alchair.jpeg"

    /// Bundle `test_images/alchair.{jpeg,jpg,png}` -> bundle root -> `Documents/test_images/` ->
    /// `Documents/` -> optional Mac dev path (Simulator/host only).
    static func resolvedInputURL() -> URL? {
        let fileManager = FileManager.default
        var candidates: [URL] = []
        for ext in ["jpeg", "jpg", "png"] {
            if let bundleURL = Bundle.main.url(forResource: "alchair", withExtension: ext, subdirectory: "test_images") {
                candidates.append(bundleURL)
            }
            if let bundleURL = Bundle.main.url(forResource: "alchair", withExtension: ext) {
                candidates.append(bundleURL)
            }
        }
        if let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first {
            candidates.append(documentsURL.appendingPathComponent("test_images/alchair.jpeg"))
            candidates.append(documentsURL.appendingPathComponent("test_images/alchair.jpg"))
            candidates.append(documentsURL.appendingPathComponent("test_images/alchair.png"))
            candidates.append(documentsURL.appendingPathComponent("alchair.jpeg"))
        }
        candidates.append(URL(fileURLWithPath: oneImageInputPathMacDev))
        var seenPaths = Set<String>()
        for candidateURL in candidates {
            let path = candidateURL.path
            guard !seenPaths.contains(path) else { continue }
            seenPaths.insert(path)
            if fileManager.fileExists(atPath: path) {
                return candidateURL
            }
        }
        return nil
    }

    static func logInputNotFound() {
        logDebug(
            "🖼️ oneImageRun: missing alchair image. Add `test_images/alchair.jpeg` to the Furnit target, " +
            "or copy the file to the app Documents folder (Files app / container), " +
            "or place at \(oneImageInputPathMacDev) when the sandbox allows (often Simulator only)."
        )
    }

    static func writeOutputPNG(_ image: CGImage, filename: String, logLabel: String) {
        let uiImage = UIImage(cgImage: image, scale: 1, orientation: .up)
        guard let pngData = uiImage.pngData() else {
            logDebug("🖼️ oneImageRun: failed to encode \(logLabel) PNG")
            return
        }

        let primaryURL = resolvedOutputPrimaryURL(filename: filename)
        let repoURL = outputURLMacRepo(filename: filename)

        do {
            try FileManager.default.createDirectory(
                at: primaryURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try pngData.write(to: primaryURL, options: .atomic)
            logDebug("🖼️ oneImageRun: wrote \(logLabel) to \(primaryURL.path)")
        } catch {
            logDebug("🖼️ oneImageRun: failed writing \(logLabel) to Documents — \(error)")
        }

        do {
            try FileManager.default.createDirectory(
                at: repoURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try pngData.write(to: repoURL, options: .atomic)
            logDebug("🖼️ oneImageRun: also wrote \(logLabel) to \(repoURL.path)")
        } catch {
            // Expected on-device; repo path is best-effort only.
        }
    }

    private static func resolvedOutputPrimaryURL(filename: String = "alchair_furniturefit_result.png") -> URL {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsURL.appendingPathComponent(filename)
    }

    private static func outputURLMacRepo(filename: String = "alchair_furniturefit_result.png") -> URL {
        URL(fileURLWithPath: oneImageInputPathMacDev).deletingLastPathComponent().appendingPathComponent(filename)
    }
}
