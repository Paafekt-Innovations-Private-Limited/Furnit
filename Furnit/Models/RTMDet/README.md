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
- This folder is only a stable in-repo landing zone so the model is not left at repo root.
- Use `scripts/install_rtmdet_ios_model.sh` to copy a local model here quickly.
