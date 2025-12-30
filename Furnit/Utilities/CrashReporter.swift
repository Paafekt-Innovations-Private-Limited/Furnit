import Foundation
import SwiftUI
import MessageUI

// MARK: - Crash Reporter
/// Centralized error reporting utility for the app
final class CrashReporter: ObservableObject {
    static let shared = CrashReporter()

    @Published var showingErrorAlert = false
    @Published var currentError: AppError?

    private init() {}

    // MARK: - Error Reporting

    /// Report an error with context information
    /// - Parameters:
    ///   - error: The error that occurred
    ///   - context: Additional context about where the error occurred
    ///   - file: Source file (auto-populated)
    ///   - function: Function name (auto-populated)
    ///   - line: Line number (auto-populated)
    func report(
        _ error: Error,
        context: String,
        file: String = #file,
        function: String = #function,
        line: Int = #line
    ) {
        let fileName = (file as NSString).lastPathComponent
        let appError = AppError(
            error: error,
            context: context,
            file: fileName,
            function: function,
            line: line,
            timestamp: Date()
        )

        // Log the error
        logDebug("❌ [CrashReporter] Error in \(fileName):\(line) - \(function)")
        logDebug("   Context: \(context)")
        logDebug("   Error: \(error.localizedDescription)")

        // Show alert on main thread
        DispatchQueue.main.async {
            self.currentError = appError
            self.showingErrorAlert = true
        }
    }

    /// Generate email body for crash report
    func generateReportBody() -> String {
        guard let error = currentError else { return "" }

        let deviceInfo = """
        Device: \(UIDevice.current.model)
        iOS Version: \(UIDevice.current.systemVersion)
        App Version: \(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown")
        Build: \(Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "Unknown")
        """

        let errorDetails = """
        --- Error Report ---

        Context: \(error.context)
        Error: \(error.error.localizedDescription)

        Location: \(error.file):\(error.line)
        Function: \(error.function)
        Timestamp: \(error.formattedTimestamp)

        --- Device Info ---
        \(deviceInfo)

        --- Additional Notes ---
        (Please describe what you were doing when this error occurred)

        """

        return errorDetails
    }

    /// Get email URL for submitting report
    func getEmailURL() -> URL? {
        guard let error = currentError else { return nil }

        let subject = "Paafekt Crash Report: \(error.context)"
        let body = generateReportBody()

        let encodedSubject = subject.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        let encodedBody = body.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""

        return URL(string: "mailto:support@paafekt.com?subject=\(encodedSubject)&body=\(encodedBody)")
    }
}

// MARK: - App Error Model
struct AppError: Identifiable {
    let id = UUID()
    let error: Error
    let context: String
    let file: String
    let function: String
    let line: Int
    let timestamp: Date

    var formattedTimestamp: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return formatter.string(from: timestamp)
    }
}

// MARK: - Error Alert View Modifier
struct CrashReportAlertModifier: ViewModifier {
    @ObservedObject var reporter = CrashReporter.shared
    @State private var showingMailError = false

    func body(content: Content) -> some View {
        content
            .alert(L10n.Common.error, isPresented: $reporter.showingErrorAlert) {
                Button(L10n.CrashReport.submitReport) {
                    submitReport()
                }
                Button(L10n.Common.ok, role: .cancel) {}
            } message: {
                if let error = reporter.currentError {
                    Text("\(error.context)\n\n\(error.error.localizedDescription)")
                }
            }
            .alert(L10n.CrashReport.emailNotConfigured, isPresented: $showingMailError) {
                Button(L10n.CrashReport.copyDetails) {
                    copyErrorDetails()
                }
                Button(L10n.Common.ok, role: .cancel) {}
            } message: {
                Text(L10n.CrashReport.emailNotConfiguredMessage)
            }
    }

    private func submitReport() {
        if let url = reporter.getEmailURL(), UIApplication.shared.canOpenURL(url) {
            UIApplication.shared.open(url)
        } else {
            showingMailError = true
        }
    }

    private func copyErrorDetails() {
        UIPasteboard.general.string = reporter.generateReportBody()
    }
}

// MARK: - View Extension
extension View {
    /// Adds crash report alert handling to the view
    func crashReportAlert() -> some View {
        modifier(CrashReportAlertModifier())
    }
}

// MARK: - Localization Extension
extension L10n {
    enum CrashReport {
        static let submitReport = "crashReport.submitReport".localized
        static let emailNotConfigured = "crashReport.emailNotConfigured".localized
        static let emailNotConfiguredMessage = "crashReport.emailNotConfiguredMessage".localized
        static let copyDetails = "crashReport.copyDetails".localized
    }
}
