// FurnitureFitUIKitExtensions.swift
// Small UIKit / geometry helpers used by Furniture Fit (extracted from FurnitureFitView).

import UIKit

extension UIView {
    var parentViewController: UIViewController? {
        var responder: UIResponder? = self
        while let r = responder {
            if let vc = r as? UIViewController { return vc }
            responder = r.next
        }
        return nil
    }
}

extension CGRect {
    var area: CGFloat { width * height }
}
