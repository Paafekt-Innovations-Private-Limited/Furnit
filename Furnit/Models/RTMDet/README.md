# RTMDet Model Drop Zone

Place the iOS Core ML export for `RTMDet-Ins-m` here.

Preferred filenames:

- `rtmdet-ins-m.mlpackage`
- `rtmdet-ins-m.mlmodelc`

Also accepted by the Swift loader:

- `rtmdet_ins_m.mlpackage`
- `rtmdet_ins_m.mlmodelc`
- `rtmdet-ins-m-coreml.mlpackage`
- `rtmdet-ins-m-coreml.mlmodelc`

Important:

- The model must be added to the `Furnit` app target resources in Xcode.
- For App Store/TestFlight distribution, `rtmdet-ins-m.mlpackage` at this path is tagged with the On-Demand Resources tag `RTMDetModel` by `Furnit.xcodeproj`.
- This folder is only a stable in-repo landing zone so the model is not left at repo root.
- Use `scripts/install_rtmdet_ios_model.sh` to copy a local model here quickly.
- Current Swift postprocess expects raw RTMDet heads:
  - `cls_80`, `bbox_80`, `kernel_80`
  - `cls_40`, `bbox_40`, `kernel_40`
  - `cls_20`, `bbox_20`, `kernel_20`
  - `mask_feat`
- Current preferred export accepts an image input and performs BGR mean/std normalization inside the Core ML graph.
- See `docs/RTMDET_IOS_SWIFT_SPIKE.md`, `Furnit/Views/FurnitureFit/README.md`, and `Furnit/diagrams/rtmdet-swift-flow.svg` for the live/still-image pipeline.
