import Foundation
import SwiftUI

// MARK: - String Localization Extension
extension String {
    /// Returns the localized version of this string key
    var localized: String {
        NSLocalizedString(self, comment: "")
    }

    /// Returns the localized version with format arguments
    func localized(_ args: CVarArg...) -> String {
        String(format: NSLocalizedString(self, comment: ""), arguments: args)
    }
}

// MARK: - Localized String Keys
/// Centralized localization keys for type-safe access
enum L10n {
    // MARK: App General
    enum App {
        static let name = "app.name".localized
        static let tagline = "app.tagline".localized
        static let version = "app.version".localized
        static let developer = "app.developer".localized
    }

    // MARK: Common Actions
    enum Common {
        static let done = "common.done".localized
        static let cancel = "common.cancel".localized
        static let ok = "common.ok".localized
        static let back = "common.back".localized
        static let reset = "common.reset".localized
        static let delete = "common.delete".localized
        static let error = "common.error".localized
        static let save = "common.save".localized
        static let close = "common.close".localized
    }

    // MARK: Login
    enum Login {
        static let yourName = "login.yourName".localized
        static let enterName = "login.enterName".localized
        static let phoneNumber = "login.phoneNumber".localized
        static let phonePlaceholder = "login.phonePlaceholder".localized
        static let sendOTP = "login.sendOTP".localized
        static let sending = "login.sending".localized
        static let otpHint = "login.otpHint".localized
        static let validationError = "login.validationError".localized
        static let sendFailed = "login.sendFailed".localized
    }

    // MARK: Country Picker
    enum Country {
        static let selectTitle = "country.selectTitle".localized
        static let searchPlaceholder = "country.searchPlaceholder".localized
    }

    // MARK: OTP
    enum OTP {
        static let title = "otp.title".localized
        static let subtitle = "otp.subtitle".localized
        static let verify = "otp.verify".localized
        static let verifying = "otp.verifying".localized
        static let resend = "otp.resend".localized
        static func resendIn(_ seconds: Int) -> String {
            "otp.resendIn".localized(seconds)
        }
        static let invalidError = "otp.invalidError".localized
        static let resendFailed = "otp.resendFailed".localized
    }

    // MARK: Home
    enum Home {
        static let title = "home.title".localized
        static let noModels = "home.noModels".localized
        static let noModelsDescription = "home.noModelsDescription".localized
        static let createRoom = "home.createRoom".localized
        static func roomsRemaining(_ remaining: Int, _ total: Int) -> String {
            "home.roomsRemaining".localized(remaining, total)
        }
        static let deleteHint = "home.deleteHint".localized
        static let swipeHint = "home.swipeHint".localized
        static let roomModel = "home.roomModel".localized
        static let renameRoom = "home.renameRoom".localized
        static let renameRoomMessage = "home.renameRoomMessage".localized
        static let roomNamePlaceholder = "home.roomNamePlaceholder".localized
    }

    // MARK: Room Limit
    enum RoomLimit {
        static let title = "roomLimit.title".localized
        static func message(_ limit: Int) -> String {
            "roomLimit.message".localized(limit)
        }
    }

    // MARK: Delete Room
    enum DeleteRoom {
        static let title = "deleteRoom.title".localized
        static func message(_ name: String) -> String {
            "deleteRoom.message".localized(name)
        }
    }

    // MARK: Settings
    enum Settings {
        static let title = "settings.title".localized
        static let quality = "settings.quality".localized
        static let qualityFooter = "settings.qualityFooter".localized
        static let appInfo = "settings.appInfo".localized
        static let currentQuality = "settings.currentQuality".localized
        static let movementSpeed = "settings.movementSpeed".localized
        static let movementSpeedFooter = "settings.movementSpeedFooter".localized
        static let roomDimensions = "settings.roomDimensions".localized
        static let roomDimensionsFooter = "settings.roomDimensionsFooter".localized
        static func width(_ value: Double) -> String {
            "settings.width".localized(value)
        }
        static func depth(_ value: Double) -> String {
            "settings.depth".localized(value)
        }
        static func height(_ value: Double) -> String {
            "settings.height".localized(value)
        }
        static let developer = "settings.developer".localized
        static let developerFooter = "settings.developerFooter".localized
        static let debugMode = "settings.debugMode".localized
        static let debugModeDescription = "settings.debugModeDescription".localized
        static let currentValue = "settings.currentValue".localized
        static let account = "settings.account".localized
        static let loggedInAs = "settings.loggedInAs".localized
        static let legal = "settings.legal".localized
        static let privacyPolicy = "settings.privacyPolicy".localized
        static let termsOfService = "settings.termsOfService".localized
        static let support = "settings.support".localized
        static let licenses = "settings.licenses".localized
        static let roomViewerSection = "settings.roomViewerSection".localized
        static let autoOrbit = "settings.autoOrbit".localized
        static let autoOrbitDescription = "settings.autoOrbitDescription".localized
        static let infiniteZoom = "settings.infiniteZoom".localized
        static let infiniteZoomDescription = "settings.infiniteZoomDescription".localized
        static let furnitureSegmentationSection = "settings.furnitureSegmentationSection".localized
        static let arAssistedFurnitureSizing = "settings.arAssistedFurnitureSizing".localized
        static let arAssistedFurnitureSizingDescription = "settings.arAssistedFurnitureSizingDescription".localized
        static let arAssistedNotSupported = "settings.arAssistedNotSupported".localized
    }

    // MARK: Licenses & Attributions
    enum Licenses {
        static let title = "licenses.title".localized
        static let phase1Notice = "licenses.phase1Notice".localized
        static let openSourceSection = "licenses.openSourceSection".localized
        static let openSourceIntro = "licenses.openSourceIntro".localized
        static let viewFullLicense = "licenses.viewFullLicense".localized
        static let yoloeTitle = "licenses.yoloeTitle".localized
        static let yoloe = "licenses.yoloe".localized
        static let sharpTitle = "licenses.sharpTitle".localized
        static let sharp = "licenses.sharp".localized
        static let firebaseTitle = "licenses.firebaseTitle".localized
        static let firebase = "licenses.firebase".localized
    }

    // MARK: Help & Support
    enum Help {
        static let title = "help.title".localized
        static let cantFind = "help.cantFind".localized
        static let contactDescription = "help.contactDescription".localized
        static let contactSupport = "help.contactSupport".localized
        static let emailSupport = "help.emailSupport".localized
        static let copyEmail = "help.copyEmail".localized
        static let emailCopied = "help.emailCopied".localized
        static let emailCopiedMessage = "help.emailCopiedMessage".localized
        static let emailSubject = "help.emailSubject".localized
        static let emailBody = "help.emailBody".localized
    }

    // MARK: Profile
    enum Profile {
        static let title = "profile.title".localized
        static let editProfile = "profile.editProfile".localized
        static let notifications = "profile.notifications".localized
        static let privacy = "profile.privacy".localized
        static let privacySettings = "profile.privacySettings".localized
        static let general = "profile.general".localized
        static let generalSettings = "profile.generalSettings".localized
        static let about = "profile.about".localized
        static let helpSupport = "profile.helpSupport".localized
        static let logout = "profile.logout".localized
        static let logoutConfirmTitle = "profile.logoutConfirmTitle".localized
        static let logoutConfirmMessage = "profile.logoutConfirmMessage".localized
        static let sectionAccount = "profile.sectionAccount".localized
        static let sectionSettings = "profile.sectionSettings".localized
        static let sectionAbout = "profile.sectionAbout".localized
        static let notificationSettings = "profile.notificationSettings".localized
    }

    // MARK: Boundary
    enum Boundary {
        static let title = "boundary.title".localized
        static let instructions = "boundary.instructions".localized
        static let floor = "boundary.floor".localized
        static let ceiling = "boundary.ceiling".localized
        static let walls = "boundary.walls".localized
        static let vanish = "boundary.vanish".localized
    }

    // MARK: Photo Room
    enum PhotoRoom {
        static let title = "photoRoom.title".localized
        static let createTitle = "photoRoom.createTitle".localized
        static let createSubtitle = "photoRoom.createSubtitle".localized
        static let quickPhoto = "photoRoom.quickPhoto".localized
        static let quickPhotoSubtitle = "photoRoom.quickPhotoSubtitle".localized
    }

    // MARK: Room Viewer
    enum RoomViewer {
        static let title = "roomViewer.title".localized
        static let controls = "roomViewer.controls".localized
        static let saveRoom = "roomViewer.saveRoom".localized
        static let roomName = "roomViewer.roomName".localized
        static let enterName = "roomViewer.enterName".localized
        static let savingRoom = "roomViewer.savingRoom".localized
        static let preparingModel = "roomViewer.preparingModel".localized
        static let exportingUSDZ = "roomViewer.exportingUSDZ".localized
        static let savingToLibrary = "roomViewer.savingToLibrary".localized
        static let almostDone = "roomViewer.almostDone".localized
        static func saveSuccess(_ name: String) -> String {
            "roomViewer.saveSuccess".localized(name)
        }
        static func saveFailed(_ error: String) -> String {
            "roomViewer.saveFailed".localized(error)
        }
        static let roomSaveTitle = "roomViewer.roomSaveTitle".localized
    }

    // MARK: Model Viewer
    enum Viewer {
        static let furniture = "viewer.furniture".localized
        static let capture = "viewer.capture".localized
    }

    // MARK: Quality Options
    enum Quality {
        static let standard = "quality.standard".localized
        static let standardDescription = "quality.standard.description".localized
        static let high = "quality.high".localized
        static let highDescription = "quality.high.description".localized
        static let best = "quality.best".localized
        static let bestDescription = "quality.best.description".localized
        static let bestUnavailable = "quality.best.unavailable".localized
    }

    // MARK: Movement Speed Options
    enum Speed {
        static let slow = "speed.slow".localized
        static let slowDescription = "speed.slow.description".localized
        static let normal = "speed.normal".localized
        static let normalDescription = "speed.normal.description".localized
        static let fast = "speed.fast".localized
        static let fastDescription = "speed.fast.description".localized
    }
}
