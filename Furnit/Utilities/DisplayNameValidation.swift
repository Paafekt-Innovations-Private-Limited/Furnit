import Foundation

/// Shared rule for login name and room name: ≥3 characters with at least one letter.
enum DisplayNameValidation {
    static func isValid(_ raw: String) -> Bool {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 3 else { return false }
        return trimmed.contains { $0.isLetter }
    }
}
