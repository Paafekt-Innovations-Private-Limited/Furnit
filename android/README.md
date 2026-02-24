# Furnit Android

Android app for Furnit (3D room models, SHARP inference, etc.).

## Removed components

### CameraClassifierActivity (SmartyPants)

**Removed:** The live camera classifier feature and its entry point have been removed from the app.

**What it did:**

- **CameraClassifierActivity** was a full-screen activity that showed a live camera preview with an on-device object-detection overlay (SmartyPants). It used **CameraX** for preview and **ExecuTorch** with a **MobileNetV3** classification model to run inference on each frame (~10 FPS), showing top-5 class labels and an FPS counter.
- It was launched from a camera icon (📷) in the home screen top bar (ContentActivity). That icon has been removed.
- Supporting code that was removed together with it:
  - **ExecutorchClassifier** – loaded and ran the ExecuTorch MobileNetV3 `.pte` model and returned classification results.
  - **FrameAnalyzer** – CameraX `ImageAnalysis.Analyzer` that throttled frames and called the classifier, then passed results to the UI.
  - **ImagePreprocessor** – converted CameraX `ImageProxy` (YUV) to bitmap and preprocessed for MobileNetV3 (224×224, ImageNet normalization, NCHW `FloatArray`).

Room creation from a **single photo** (take or pick image → AI or manual 3D room) is unchanged and is still available via the gallery/image icon (🖼) on the home screen; that flow uses `SinglePhotoRoomActivity`, not the removed camera classifier.
