# RTMDet iOS Swift Spike

This repo now contains a first-pass Swift test path for `RTMDet-Ins-m` in the iOS app.

## What is already wired

- Developer image-scan screen now supports a backend switch:
  - `RTMDet-Ins-m`
  - `YOLOE`
- Swift-side RTMDet loader:
  - `Furnit/Services/OnDevice/RTMDetModelService.swift`
- Swift-side RTMDet still-image inference adapter:
  - `Furnit/Services/OnDevice/RTMDetImageInference.swift`
- Existing live room segmentation remains on the YOLOE path for now.

## What is still external

The actual `RTMDet-Ins-m` Core ML model package is not in this repo.

You still need to add one of these bundled model names to the iOS target:

- `rtmdet-ins-m.mlpackage`
- `rtmdet-ins-m.mlmodelc`
- `rtmdet_ins_m.mlpackage`
- `rtmdet_ins_m.mlmodelc`
- `rtmdet-ins-m-coreml.mlpackage`
- `rtmdet-ins-m-coreml.mlmodelc`

Recommended in-repo location:

- `Furnit/Models/RTMDet/`

Helper:

- `scripts/install_rtmdet_ios_model.sh /path/to/rtmdet-ins-m.mlpackage`

The loader tries those names with this compute-unit fallback chain:

1. `.all`
2. `.cpuAndNeuralEngine`
3. `.cpuAndGPU`
4. `.cpuOnly`

## How to test in the iOS app

1. Add the RTMDet Core ML package to the `Furnit` target.
   Preferred location: `Furnit/Models/RTMDet/`
2. Launch the app.
3. Open:
   - `Settings`
   - `Image scan`
4. Switch backend to `RTMDet-Ins-m`.
5. Pick a furniture photo.

What you should see:

- detection boxes over the image
- a basic mask overlay if the Core ML export exposes mask tensors
- a small debug card with the first few output tensor names and shapes

## Local output inspection

There is also a local probe script:

- `scripts/probe_rtmdet_coreml.py`

Example:

```bash
python3 scripts/probe_rtmdet_coreml.py \
  --model /path/to/rtmdet-ins-m.mlpackage \
  --image /path/to/test.jpg \
  --preprocess stretch
```

This script:

- loads the model with `coremltools`
- prints output tensor names, shapes, and ranges
- heuristically identifies box / label / mask outputs
- prints the top scored box rows

Use this first if the Swift overlay looks wrong. It will tell you whether the export format matches the current Swift assumptions.

## Current limits

- This is a still-image spike only.
- The mask parser is heuristic because the exact Core ML export format depends on how RTMDet was converted.
- Live camera segmentation should not be switched to RTMDet until the output tensors are verified on a real exported model.
- License review still needs to cover the exact checkpoint you export from, not just the OpenMMLab code license.
