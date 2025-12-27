import UIKit
import PhotosUI
import AVFoundation

final class VideoPicker: NSObject, PHPickerViewControllerDelegate {

    var onPickedURL: ((URL) -> Void)?

    func present(from vc: UIViewController) {
        var config = PHPickerConfiguration(photoLibrary: .shared())
        config.filter = .videos
        config.selectionLimit = 1

        let picker = PHPickerViewController(configuration: config)
        picker.delegate = self
        vc.present(picker, animated: true)
    }

    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)

        guard let itemProvider = results.first?.itemProvider,
              itemProvider.hasItemConformingToTypeIdentifier("public.movie") else {
            return
        }

        itemProvider.loadFileRepresentation(forTypeIdentifier: "public.movie") { url, error in
            guard let url, error == nil else { return }

            let fm = FileManager.default
            let dst = fm.temporaryDirectory
                .appendingPathComponent("test_video_\(UUID().uuidString).mov")

            do {
                if fm.fileExists(atPath: dst.path) {
                    try fm.removeItem(at: dst)
                }
                try fm.copyItem(at: url, to: dst)
                DispatchQueue.main.async {
                    self.onPickedURL?(dst)
                }
            } catch {
                print("❌ VideoPicker copy failed:", error)
            }
        }
    }
}
