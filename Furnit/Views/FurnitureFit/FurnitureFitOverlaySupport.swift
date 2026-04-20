import UIKit

// MARK: - Detection & Sizing Types
/// High-level furniture size estimate surfaced to SharpRoom / viewers.
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
    case segmentSelected
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
            let strokeColor: UIColor = item.isSelected ? .systemYellow : UIColor.white.withAlphaComponent(0.88)
            let fillColor = UIColor.black.withAlphaComponent(item.isSelected ? 0.55 : 0.38)
            let lineWidth: CGFloat = item.isSelected ? 2.5 : 1.2
            let boxPath = UIBezierPath(roundedRect: item.rectInView, cornerRadius: 6)
            fillColor.setFill()
            strokeColor.setStroke()
            boxPath.lineWidth = lineWidth
            boxPath.stroke()

            let scoreText = String(format: "%.2f", item.confidence)
            let text = "\(item.label) \(scoreText)"
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
