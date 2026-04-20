// FurnitureFitClassBlacklist.swift
// Loads blacklist.json (string keys → YOLO class indices to ignore) once per Furniture Fit session.

import Foundation

/// Loads `blacklist.json` once; thread-safe for main + detection queue.
final class FurnitureFitClassBlacklist {
    private let lock = NSLock()
    private var _ignoredIndices: Set<Int> = []
    private var didLoad = false

    var ignoredIndices: Set<Int> {
        lock.lock()
        defer { lock.unlock() }
        return _ignoredIndices
    }

    func loadBlacklistOnce(debugMode: Bool, log: (String) -> Void) {
        lock.lock()
        defer { lock.unlock() }

        guard !didLoad else { return }
        defer { didLoad = true }

        guard let url = Bundle.main.url(forResource: "blacklist", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] else {
            if debugMode { log("⚠️ Failed to load blacklist.json") }
            _ignoredIndices = []
            return
        }
        _ignoredIndices = Set(dict.keys.compactMap { Int($0) })
        if debugMode { log("✅ Loaded \(_ignoredIndices.count) blacklisted classes") }
    }
}
