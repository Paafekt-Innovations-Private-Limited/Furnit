import Foundation
import WebKit

enum BundledWebViewAsset {
    static let scheme = "viewer-assets"
    static let host = "local"
    static let bundleSubdirectoryCandidates = [
        "Resources/WebViewVendor",
        "WebViewVendor",
    ]

    static func assetURLString(for relativePath: String) -> String {
        "\(scheme)://\(host)/\(relativePath)"
    }
}

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
        for subdirectory in BundledWebViewAsset.bundleSubdirectoryCandidates {
            let candidateURL = Bundle.main.bundleURL
                .appendingPathComponent(subdirectory, isDirectory: true)
                .appendingPathComponent(relativePath)
            if fileManager.fileExists(atPath: candidateURL.path) {
                return candidateURL
            }
        }
        return nil
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
