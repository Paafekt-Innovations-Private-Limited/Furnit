import UIKit

// MARK: - Detection & Sizing Types
/// High-level furniture size estimate surfaced to SplatRoom / viewers.
struct FurnitureSizeEstimate {
    /// Width from room-model intrinsics when available, else bbox × room width (no-LiDAR fallback).
    let widthMeters: Float
    /// Display height: AR when available, else bbox × room height (no-LiDAR fallback).
    let heightMeters: Float
    /// ARKit/LiDAR height when available.
    let arHeightMeters: Float?
}

enum FurnitureFitSegmentationMode: Equatable {
    case identifyOnly
    /// Brain default: auto-segment the single highest-confidence detection with no tap.
    case segmentPrimary
    /// Segment user-selected detections. In full-video mode inference remains live while the
    /// camera passthrough hides and the cutout becomes an interactive overlay over the room.
    case segmentSelected
}

/// Separates camera-frame layout from overlay interaction. Full-video selected segmentation still
/// consumes live, frame-aligned masks, but once the camera passthrough is hidden the cutout is a
/// placement overlay and must accept the same pinch/pan gestures as the default segmentation flow.
enum FurnitureFitOverlayInteractionMode: Equatable {
    case standardPlacement
    case liveIdentification
    case liveSelectedPlacement

    static func resolve(
        showFullVideoWithIdentifications: Bool,
        showIdentifyLivePreview: Bool,
        segmentationMode: FurnitureFitSegmentationMode
    ) -> FurnitureFitOverlayInteractionMode {
        guard showFullVideoWithIdentifications, showIdentifyLivePreview else {
            return .standardPlacement
        }

        switch segmentationMode {
        case .identifyOnly:
            return .liveIdentification
        case .segmentSelected:
            return .liveSelectedPlacement
        case .segmentPrimary:
            return .standardPlacement
        }
    }

    var usesLiveFrameGeometry: Bool {
        self == .liveIdentification || self == .liveSelectedPlacement
    }

    var allowsPlacementGestures: Bool {
        self != .liveIdentification
    }

    var preservesIdentityTransform: Bool {
        self == .liveIdentification
    }
}

struct DetectionOverlayItem {
    let rectInView: CGRect
    let label: String
    let confidence: Float
    let isSelected: Bool
}

final class DetectionBBoxOverlayView: UIView {
    var items: [DetectionOverlayItem] = [] {
        didSet { setNeedsDisplay() }
    }
    /// When false, labels are hidden; callers decide which cluster boxes are visible.
    var showsDiagnosticLabels = false

    private static let accentGold = UIColor(red: 201 / 255, green: 162 / 255, blue: 75 / 255, alpha: 1)

    override init(frame: CGRect) {
        super.init(frame: frame)
        commonInit()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    private func commonInit() {
        backgroundColor = .clear
        isOpaque = false
        isUserInteractionEnabled = false
        contentMode = .redraw
    }

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }
        ctx.clear(rect)
        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.lineBreakMode = .byTruncatingTail
        for item in items {
            let strokeColor: UIColor = item.isSelected
                ? Self.accentGold
                : UIColor.white.withAlphaComponent(0.55)
            let lineWidth: CGFloat = item.isSelected ? 2.5 : 1.2
            let boxPath = UIBezierPath(roundedRect: item.rectInView, cornerRadius: 6)
            strokeColor.setStroke()
            boxPath.lineWidth = lineWidth
            boxPath.stroke()

            guard showsDiagnosticLabels else { continue }

            let fillColor = UIColor.black.withAlphaComponent(item.isSelected ? 0.55 : 0.38)
            let scoreText = String(format: "%.2f", item.confidence)
            let text = item.label.isEmpty ? "" : "\(item.label) \(scoreText)"
            guard !text.isEmpty else { continue }
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: item.isSelected ? 11 : 10, weight: item.isSelected ? .semibold : .medium),
                .foregroundColor: UIColor.white,
                .paragraphStyle: paragraphStyle
            ]
            let maxLabelWidth = min(max(item.rectInView.width, 56), 140)
            let textSize = (text as NSString).size(withAttributes: attributes)
            let labelRect = CGRect(
                x: item.rectInView.minX,
                y: max(0, item.rectInView.minY - textSize.height - 8),
                width: min(maxLabelWidth, textSize.width + 10),
                height: textSize.height + 6
            )
            let labelPath = UIBezierPath(roundedRect: labelRect, cornerRadius: 6)
            fillColor.setFill()
            labelPath.fill()
            (text as NSString).draw(
                in: labelRect.insetBy(dx: 5, dy: 3),
                withAttributes: attributes
            )
        }
    }
}

extension UIButton.Configuration {
    static func furnitureSelectionChip() -> UIButton.Configuration {
        var config = UIButton.Configuration.plain()
        config.baseForegroundColor = .white
        config.background.backgroundColor = UIColor.black.withAlphaComponent(0.72)
        config.background.cornerRadius = 18
        config.contentInsets = NSDirectionalEdgeInsets(top: 8, leading: 14, bottom: 8, trailing: 14)
        config.image = UIImage(systemName: "xmark.circle.fill")
        config.imagePlacement = .trailing
        config.imagePadding = 8
        return config
    }
}
