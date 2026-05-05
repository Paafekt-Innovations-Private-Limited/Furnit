import Foundation

/// Bridges ``FurnitODRResourceAccess`` into async Swift. Use instead of `request.beginAccessingResources()`
/// so Obj-C ``NSException`` from the streaming-unzip path becomes a caught ``NSError`` (4099), not a crash.
enum FurnitODRBeginAccessing {
    static func beginAccessingResources(_ request: NSBundleResourceRequest) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            FurnitODRResourceAccess.beginAccessingResources(request) { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            }
        }
    }
}
