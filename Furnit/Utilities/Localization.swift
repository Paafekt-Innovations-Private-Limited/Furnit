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
        static let apply = "common.apply".localized
        static let or = "common.or".localized
        static let retry = "common.retry".localized
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
        static let createRoomToolbarLabel = "home.createRoomToolbarLabel".localized
        static let createRoomHint = "home.createRoomHint".localized
        static func roomsRemaining(_ remaining: Int, _ total: Int) -> String {
            "home.roomsRemaining".localized(remaining, total)
        }
        static let deleteHint = "home.deleteHint".localized
        static let swipeHint = "home.swipeHint".localized
        static let roomModel = "home.roomModel".localized
        /// Saved PLY / Splat Gaussian room (home list subtitle under room name).
        static let aiBased3DRoom = "home.aiBased3DRoom".localized
        /// Saved manual mesh / GLB room (home list subtitle under room name).
        static let manualBased3DRoom = "home.manualBased3DRoom".localized
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
        static let message = "deleteRoom.message".localized
    }

    // MARK: Settings
    enum Settings {
        static let title = "settings.title".localized
        static let appInfo = "settings.appInfo".localized
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
        static let showTipsAgain = "settings.showTipsAgain".localized
        static let legal = "settings.legal".localized
        static let privacyPolicy = "settings.privacyPolicy".localized
        static let termsOfService = "settings.termsOfService".localized
        static let support = "settings.support".localized
        static let credits = "settings.credits".localized
        static let licenses = "settings.licenses".localized
        static let roomViewerSection = "settings.roomViewerSection".localized
        static let autoOrbit = "settings.autoOrbit".localized
        static let autoOrbitDescription = "settings.autoOrbitDescription".localized
        static let infiniteZoom = "settings.infiniteZoom".localized
        static let infiniteZoomDescription = "settings.infiniteZoomDescription".localized
        static let fullVideoWithIdentifications = "settings.fullVideoWithIdentifications".localized
        static let fullVideoWithIdentificationsDescription = "settings.fullVideoWithIdentificationsDescription".localized
        static let singleImageScanSection = "settings.singleImageScanSection".localized
        
        static let imageScan = "settings.imageScan".localized
        static let imageScanDescription = "settings.imageScanDescription".localized
        static let imageScanTapToChoose = "settings.imageScanTapToChoose".localized
        static let imageScanTapToChooseSubtitle = "settings.imageScanTapToChooseSubtitle".localized
        static let imageScanFootnote = "settings.imageScanFootnote".localized
        static let imageScanLoadingPhoto = "settings.imageScanLoadingPhoto".localized
        static let imageScanPreparingModel = "settings.imageScanPreparingModel".localized
        static let imageScanLoadFailed = "settings.imageScanLoadFailed".localized
        static let accountDeleteFooter = "settings.accountDeleteFooter".localized
    }

    /// Splat on-device status messages (model load + generation)
    enum RoomGeneration {
        static let downloadingEngine = "roomGeneration.downloadingEngine".localized
        static func downloadingEnginePercent(_ percent: Int) -> String {
            String(format: "roomGeneration.downloadingEnginePercent".localized, locale: .current, percent)
        }
        static let downloadComplete = "roomGeneration.downloadComplete".localized
        static let downloadFailed = "roomGeneration.downloadFailed".localized
        static let gettingReady = "roomGeneration.gettingReady".localized
        static let notEnoughSpace = "roomGeneration.notEnoughSpace".localized
        static let settingThingsUp = "roomGeneration.settingThingsUp".localized
        static let ready = "roomGeneration.ready".localized
        static let couldNotGetReady = "roomGeneration.couldNotGetReady".localized
        static let somethingWentWrong = "roomGeneration.somethingWentWrong".localized
        static let preparingPhoto = "roomGeneration.preparingPhoto".localized
        static let creatingRoom = "roomGeneration.creatingRoom".localized
        static let almostDone = "roomGeneration.almostDone".localized
        static let done = "roomGeneration.done".localized
        static let couldNotCreateRoom = "roomGeneration.couldNotCreateRoom".localized
        static let cancelled = "roomGeneration.cancelled".localized
        /// Motion-tracked splat camera (formerly labeled “AR”).
        static let liveRoom = "roomGeneration.liveRoom".localized
        /// Touch/orbit splat camera (formerly labeled “Touch”).
        static let stillRoom = "roomGeneration.stillRoom".localized
        static let liveRoomCameraMode = "roomGeneration.liveRoomCameraMode".localized
        static let stillRoomCameraMode = "roomGeneration.stillRoomCameraMode".localized
        static let cameraModeToggleAccessibilityHint = "roomGeneration.cameraModeToggleAccessibilityHint".localized
    }

    // MARK: Licenses & Attributions
    enum Licenses {
        static let title = "licenses.title".localized
        static let phase1Notice = "licenses.phase1Notice".localized
        static let openSourceSection = "licenses.openSourceSection".localized
        static let openSourceIntro = "licenses.openSourceIntro".localized
        static let viewFullLicense = "licenses.viewFullLicense".localized
        static let depthAnythingTitle = "licenses.depthAnythingTitle".localized
        static let depthAnything = "licenses.depthAnything".localized
        static let geoCalibTitle = "licenses.geoCalibTitle".localized
        static let geoCalib = "licenses.geoCalib".localized
        static let metalSplatterTitle = "licenses.metalSplatterTitle".localized
        static let metalSplatter = "licenses.metalSplatter".localized
        static let firebaseTitle = "licenses.firebaseTitle".localized
        static let firebase = "licenses.firebase".localized
        static let rtmdetTitle = "licenses.rtmdetTitle".localized
        static let rtmdet = "licenses.rtmdet".localized
        static let threeTitle = "licenses.threeTitle".localized
        static let three = "licenses.three".localized
    }

    enum Credits {
        static let title = "credits.title".localized
        static let intro = "credits.intro".localized
        static let disclaimer = "credits.disclaimer".localized
        static let visitWebsite = "credits.visitWebsite".localized
        static let appleTitle = "credits.appleTitle".localized
        static let appleBody = "credits.appleBody".localized
        static let openAITitle = "credits.openAITitle".localized
        static let openAIBody = "credits.openAIBody".localized
        static let anthropicTitle = "credits.anthropicTitle".localized
        static let anthropicBody = "credits.anthropicBody".localized
        static let lumaTitle = "credits.lumaTitle".localized
        static let lumaBody = "credits.lumaBody".localized
    }

    // MARK: Help & Support
    enum Help {
        static let title = "help.title".localized
        static let cantFind = "help.cantFind".localized
        static let measurementAccuracyTitle = "help.measurementAccuracyTitle".localized
        static let measurementAccuracyBody = "help.measurementAccuracyBody".localized
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
        static let deleteAccount = "profile.deleteAccount".localized
        static let deleteAccountConfirmTitle = "profile.deleteAccountConfirmTitle".localized
        static let deleteAccountConfirmMessage = "profile.deleteAccountConfirmMessage".localized
        static let deleteAccountSuccessTitle = "profile.deleteAccountSuccessTitle".localized
        static let deleteAccountSuccessMessage = "profile.deleteAccountSuccessMessage".localized
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
        static let howToCreate = "photoRoom.howToCreate".localized
        static let tapOption = "photoRoom.tapOption".localized
        static let aiPowered = "photoRoom.aiPowered".localized
        static let manualSetup = "photoRoom.manualSetup".localized
        static let manualSetupDesc = "photoRoom.manualSetupDesc".localized
        static let buildingRoom = "photoRoom.buildingRoom".localized
        static let buildingRoomSubtext = "photoRoom.buildingRoomSubtext".localized
        /// Manual single-photo room pipeline (`SinglePhotoRoomReconstructor`) status line on progress overlay.
        static let reconstructorReady = "photoRoom.reconstructorReady".localized
        static let reconstructorAnalyzingPhoto = "photoRoom.reconstructorAnalyzingPhoto".localized
        static let reconstructorExtractingDepth = "photoRoom.reconstructorExtractingDepth".localized
        static let reconstructorFindingWalls = "photoRoom.reconstructorFindingWalls".localized
        static let reconstructorCalculatingDimensions = "photoRoom.reconstructorCalculatingDimensions".localized
        static let reconstructorBuilding3D = "photoRoom.reconstructorBuilding3D".localized
        static let reconstructorRoomReady = "photoRoom.reconstructorRoomReady".localized
        static let reconstructorDepthFailed = "photoRoom.reconstructorDepthFailed".localized
        static let reconstructorStarting = "photoRoom.reconstructorStarting".localized
        static let reconstructorPreparingImage = "photoRoom.reconstructorPreparingImage".localized
        static let reconstructorAnalyzingBoundaries = "photoRoom.reconstructorAnalyzingBoundaries".localized
        static let reconstructorCreatingTextures = "photoRoom.reconstructorCreatingTextures".localized
        static let reconstructorFinalizing = "photoRoom.reconstructorFinalizing".localized
        static let backAlertTitle = "photoRoom.backAlertTitle".localized
        static let backAlertMessage = "photoRoom.backAlertMessage".localized
        static let backAlertAI = "photoRoom.backAlertAI".localized
        static let backAlertManual = "photoRoom.backAlertManual".localized
        static let selectPhoto = "photoRoom.selectPhoto".localized
        static let fromLibrary = "photoRoom.fromLibrary".localized
        static let screenshotWarning = "photoRoom.screenshotWarning".localized
        static let odrOneTimeDownload = "photoRoom.odrOneTimeDownload".localized
        static let modelGeneratedTitle = "photoRoom.modelGeneratedTitle".localized
        static let generationFailedTitle = "photoRoom.generationFailedTitle".localized
        static let loading3DRoom = "photoRoom.loading3DRoom".localized
        static let saveSuccessMessage = "photoRoom.saveSuccess".localized
        static let errorMessage = "photoRoom.error".localized
        static func downloadSuccess(fileName: String) -> String {
            String(format: "photoRoom.downloadSuccess".localized, locale: .current, fileName)
        }
    }

    enum Camera {
        static let takePhoto = "camera.takePhoto".localized
        static let chooseOrientationShort = "camera.chooseOrientationShort".localized
    }

    /// Progress overlay during remote/API 3D generation (upload → process → download)
    enum GenerationProgress {
        static let uploadingImage = "generationProgress.uploadingImage".localized
        static let generating3DModel = "generationProgress.generating3DModel".localized
        static let downloadingModel = "generationProgress.downloadingModel".localized
        static let complete = "generationProgress.complete".localized
        static let preparing = "generationProgress.preparing".localized
        static let mayTakeFewMinutes = "generationProgress.mayTakeFewMinutes".localized
    }

    // MARK: Room Viewer
    enum RoomViewer {
        static let title = "roomViewer.title".localized
        static let controls = "roomViewer.controls".localized
        static let saveRoom = "roomViewer.saveRoom".localized
        static let nameYourRoom = "roomViewer.nameYourRoom".localized
        static let roomName = "roomViewer.roomName".localized
        static let roomNamePlaceholder = "roomViewer.roomNamePlaceholder".localized
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
        /// Success alert title after mesh/GLB manual save (distinct from ``roomSaveTitle``).
        static let roomSavedAlertTitle = "roomViewer.roomSavedAlertTitle".localized
        /// Mesh export JS error (manual room save flow).
        static let meshExportFailed = "roomViewer.meshExportFailed".localized
        /// Generic save failure when no server message (manual mesh room).
        static let meshSaveFailedGeneric = "roomViewer.meshSaveFailedGeneric".localized
        static let exporting3DModelEllipsis = "roomViewer.exporting3DModelEllipsis".localized
        /// Shown when saving a Splat/ML room and the chosen name collides with an existing saved room.
        static let duplicateRoomName = "roomViewer.duplicateRoomName".localized
        static let share = "roomViewer.share".localized
        static let calibrateWall = "roomViewer.calibrateWall".localized
        static let recenterView = "roomViewer.recenterView".localized
        static let resetOverlayScale = "roomViewer.resetOverlayScale".localized
        static let pinchGestureHintExplanation = "roomViewer.pinchGestureHintExplanation".localized
        static let navigationTeachingHint = "roomViewer.navigationTeachingHint".localized
        static let firstRunCoachMarkTitle = "roomViewer.firstRunCoachMarkTitle".localized
        static let firstRunCoachMarkBody = "roomViewer.firstRunCoachMarkBody".localized
        static let firstRunGotIt = "roomViewer.firstRunGotIt".localized
        static let heroActionsTeachingHint = "roomViewer.heroActionsTeachingHint".localized
        static let heroFitFurniture = "roomViewer.heroFitFurniture".localized
        static let heroCapture = "roomViewer.heroCapture".localized
        static let immersiveShowControls = "roomViewer.immersiveShowControls".localized
        static let immersiveTapToHide = "roomViewer.immersiveTapToHide".localized
        static let immersiveFitShort = "roomViewer.immersiveFitShort".localized
        static let immersiveCaptureShort = "roomViewer.immersiveCaptureShort".localized
        static let brainGestureHintExplanation = "roomViewer.brainGestureHintExplanation".localized
        /// Shown at the top when full-video identifications are on and furniture fit starts; dismissed when Segment is tapped or furniture fit exits.
        static let fullVideoFurnitureTapHint = "roomViewer.fullVideoFurnitureTapHint".localized
        static let displayAllHelpers = "roomViewer.displayAllHelpers".localized
        static let segmentFurnitureAction = "roomViewer.segmentFurnitureAction".localized
        static let stopSegmentationAction = "roomViewer.stopSegmentationAction".localized
        static let segmentationDone = "roomViewer.segmentationDone".localized
        static let segmentFurnitureAccessibility = "roomViewer.segmentFurnitureAccessibility".localized
        static let snapshotGestureHintExplanation = "roomViewer.snapshotGestureHintExplanation".localized
        /// Short tip for the AR camera-sizing control in the room toolbar (user-facing, not technical).
        static let arFurnitureSizingHint = "roomViewer.arFurnitureSizingHint".localized
        static let arFurnitureSizingRequiresBrainHint = "roomViewer.arFurnitureSizingRequiresBrainHint".localized
        static let arSizingEnable = "roomViewer.arSizingEnable".localized
        static let arSizingDisable = "roomViewer.arSizingDisable".localized
        static let gestureHintToggleAccessibility = "roomViewer.gestureHintToggleAccessibility".localized
        static let checkMeasurement = "roomViewer.checkMeasurement".localized
        /// User-facing room height label (width/depth kept internally; UI shows height only).
        static func approximateRoomHeight(_ height: Float) -> String {
            String(
                format: "roomViewer.approximateRoomHeight".localized,
                locale: .current,
                height
            )
        }
        static func roomDimensionsWHDManualChip(width: Float, height: Float, depth: Float) -> String {
            roomDimensionsWHDNearAccurateChip(width: width, height: height, depth: depth)
        }
        static func roomDimensionsWHManualChip(width: Float, height: Float) -> String {
            approximateRoomHeight(height)
        }
        static func roomDimensionsWHDAIChip(width: Float, height: Float, depth: Float) -> String {
            roomDimensionsWHDNearAccurateChip(width: width, height: height, depth: depth)
        }
        static func roomDimensionsWHDNearAccurateChip(width: Float, height: Float, depth: Float) -> String {
            String(
                format: "roomViewer.roomDimensionsWHDWithNearAccurate".localized,
                locale: .current,
                width,
                height,
                depth,
                "roomViewer.roomDimensionsNearAccurateValues".localized
            )
        }
        static let measuringRoom = "roomViewer.measuringRoom".localized
        static let goingBack = "roomViewer.goingBack".localized
        static let savingRoomEllipsis = "roomViewer.savingRoomEllipsis".localized
        static let calibrateRoomTitle = "roomViewer.calibrateRoomTitle".localized
        static let enterFurnitureHeightMeters = "roomViewer.enterFurnitureHeightMeters".localized
        static let furnitureFullHeightHint = "roomViewer.furnitureFullHeightHint".localized
        static let calibrateByWallTitle = "roomViewer.calibrateByWallTitle".localized
        static let enterWallDimensionsHint = "roomViewer.enterWallDimensionsHint".localized
        static let tapToCalibrate = "roomViewer.tapToCalibrate".localized
        static let fullVideoSelectionHelper = "roomViewer.fullVideoSelectionHelper".localized
        static func detectedMeters(_ value: Float) -> String {
            String(format: "roomViewer.detectedMeters".localized, locale: .current, value)
        }
        static func furnitureHeightMustBeLessThanRoomHeight(_ value: Float) -> String {
            String(
                format: "roomViewer.furnitureHeightMustBeLessThanRoomHeight".localized,
                locale: .current,
                value
            )
        }
        static func roomMetersShort(_ value: Float) -> String {
            approximateRoomHeight(value)
        }
        static func furnitureMetersShort(_ value: Float) -> String {
            String(format: "roomViewer.furnitureMetersShort".localized, locale: .current, value)
        }

        static func furnitureHeightEstimate(_ value: Float) -> String {
            String(format: "roomViewer.furnitureHeightEstimate".localized, locale: .current, value)
        }
        static let wallWidthPlaceholder = "roomViewer.wallWidthPlaceholder".localized
        static let wallHeightPlaceholder = "roomViewer.wallHeightPlaceholder".localized
        static let saveErrorUnknown = "roomViewer.saveErrorUnknown".localized
        static let placementIntelligenceTitle = "roomViewer.placementIntelligenceTitle".localized
        static let placementBadgeStyleOnly = "roomViewer.placementBadgeStyleOnly".localized
        static let placementNoFit = "roomViewer.placementNoFit".localized
        static func placementFitCount(_ count: Int) -> String {
            String(format: "roomViewer.placementFitCount".localized, locale: .current, count)
        }
        static func placementDetectedSizeMeters(width: Double, height: Double, depth: Double) -> String {
            String(
                format: "roomViewer.placementDetectedSize".localized,
                locale: .current,
                width,
                height,
                depth
            )
        }
        static let placementMetricUnavailableNote = "roomViewer.placementMetricUnavailableNote".localized
        static let placementFitsRoom = "roomViewer.placementFitsRoom".localized
        static let placementExceedsRoom = "roomViewer.placementExceedsRoom".localized
        static func placementHarmonySummary(
            harmonyScore: Float,
            harmonyTypeName: String,
            contrastScore: Float,
            styleFit: Float
        ) -> String {
            String(
                format: "roomViewer.placementHarmonySummary".localized,
                locale: .current,
                harmonyScore,
                harmonyTypeName,
                contrastScore,
                styleFit
            )
        }
    }

    /// Unsaved room preview (back without saving)
    enum RoomPreview {
        static let unsavedTitle = "roomPreview.unsavedTitle".localized
        static let unsavedMessage = "roomPreview.unsavedMessage".localized
        static let stay = "roomPreview.stay".localized
        static let leave = "roomPreview.leave".localized
        static let prepareSourceImageFailed = "roomPreview.prepareSourceImageFailed".localized
        static let sourceImageUnavailable = "roomPreview.sourceImageUnavailable".localized
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

// MARK: - Placement intelligence / aesthetic (HarmonyType lives in AestheticAdvisor)
extension HarmonyType {
    var localizedDisplayName: String {
        switch self {
        case .analogous: return "roomViewer.harmonyTypeAnalogous".localized
        case .complementary: return "roomViewer.harmonyTypeComplementary".localized
        case .triadic: return "roomViewer.harmonyTypeTriadic".localized
        case .splitComplementary: return "roomViewer.harmonyTypeSplitComplementary".localized
        case .neutral: return "roomViewer.harmonyTypeNeutral".localized
        case .clash: return "roomViewer.harmonyTypeClash".localized
        }
    }
}
