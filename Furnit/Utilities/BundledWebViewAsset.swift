import Foundation
import WebKit

enum BundledWebViewAsset {
    static let scheme = "viewer-assets"
    static let host = "local"
    static let bundleSubdirectoryCandidates = [
        "Resources/WebViewVendor",
        "WebViewVendor",
    ]
    private static let probeRelativePath = "three/build/three.module.js"

    static func assetURLString(for relativePath: String) -> String {
        "\(scheme)://\(host)/\(relativePath)"
    }

    static func bundledBaseURL() -> URL? {
        let fileManager = FileManager.default
        let roots = [Bundle.main.resourceURL, Bundle.main.bundleURL].compactMap { $0 }

        for root in roots {
            for subdirectory in bundleSubdirectoryCandidates {
                let candidateURL = root.appendingPathComponent(subdirectory, isDirectory: true)
                let probeURL = candidateURL.appendingPathComponent(probeRelativePath)
                if fileManager.fileExists(atPath: probeURL.path) {
                    logDebug("✅ [BundledWebViewAsset] Found vendor base at \(candidateURL.path)")
                    return candidateURL
                }
            }
        }

        // Fall back to a recursive search in case Xcode flattened/moved resources inside the app bundle.
        for root in roots {
            if let enumerator = fileManager.enumerator(
                at: root,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) {
                for case let fileURL as URL in enumerator {
                    if fileURL.lastPathComponent == "three.module.js",
                       fileURL.path.contains("/three/build/three.module.js") {
                        let baseURL = fileURL
                            .deletingLastPathComponent() // build
                            .deletingLastPathComponent() // three
                        logDebug("✅ [BundledWebViewAsset] Found vendor base via recursive search at \(baseURL.path)")
                        return baseURL
                    }
                }
            }
        }

        logDebug("❌ [BundledWebViewAsset] Could not locate bundled vendor assets under resourceURL=\(Bundle.main.resourceURL?.path ?? "nil") bundleURL=\(Bundle.main.bundleURL.path)")
        return nil
    }

    static func bundledFileURL(for relativePath: String) -> URL? {
        bundledBaseURL()?.appendingPathComponent(relativePath)
    }

    static func bundledFileURLString(for relativePath: String) -> String? {
        bundledFileURL(for: relativePath)?.absoluteString
    }
}

/// Serves bundled WebViewVendor assets from the `viewer-assets://local/` custom URL scheme.
/// Used with `loadHTMLString(baseURL: viewer-assets://local/)` so that ES module importmap
/// entries pointing to `viewer-assets://local/three/...` are same-origin and resolve here.
final class BundledWebViewAssetSchemeHandler: NSObject, WKURLSchemeHandler {
    private let fileManager = FileManager.default

    func webView(_ webView: WKWebView, start urlSchemeTask: any WKURLSchemeTask) {
        guard let requestURL = urlSchemeTask.request.url else {
            urlSchemeTask.didFailWithError(NSError(domain: NSURLErrorDomain, code: NSURLErrorBadURL))
            return
        }

        let relativePath = requestURL.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard !relativePath.contains(".."), let fileURL = bundledFileURL(for: relativePath) else {
            urlSchemeTask.didFailWithError(NSError(domain: NSCocoaErrorDomain, code: NSFileNoSuchFileError))
            return
        }

        do {
            let data = try Data(contentsOf: fileURL)
            let response = HTTPURLResponse(
                url: requestURL,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: [
                    "Content-Type": contentType(for: fileURL.pathExtension),
                    "Content-Length": "\(data.count)",
                    "Cache-Control": "public, max-age=31536000",
                ]
            )!
            urlSchemeTask.didReceive(response)
            urlSchemeTask.didReceive(data)
            urlSchemeTask.didFinish()
        } catch {
            urlSchemeTask.didFailWithError(error)
        }
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: any WKURLSchemeTask) {
    }

    private func bundledFileURL(for relativePath: String) -> URL? {
        BundledWebViewAsset.bundledBaseURL()?
            .appendingPathComponent(relativePath)
    }

    private func contentType(for pathExtension: String) -> String {
        switch pathExtension.lowercased() {
        case "js", "mjs":
            return "application/javascript; charset=utf-8"
        case "json":
            return "application/json; charset=utf-8"
        case "css":
            return "text/css; charset=utf-8"
        default:
            return "application/octet-stream"
        }
    }
}
