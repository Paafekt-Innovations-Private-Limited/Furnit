# Python–Kotlin Parity Test for ExecuTorch SHARP

Validates that Python (mobile-like) and Android Kotlin produce the same Part1 output for the same room image.

## 1. Python (mobile-like)

Simulates mobile hardware with 4 threads. Run on Mac/Linux:

```bash
cd android
python test_sharp_split_mobile.py \
  --image /Users/al/Downloads/PXL_20260209_032207120.jpg \
  --output app/src/androidTest/assets/python_part1_baseline.json
```

Or use the bundled test image:

```bash
python test_sharp_split_mobile.py \
  --image app/src/androidTest/assets/PXL_room.jpg \
  -o app/src/androidTest/assets/python_part1_baseline.json
```

**Output:** `load_ms`, `forward_ms`, `tokens_checksum`, `block5_checksum` in JSON.

## 2. Kotlin (Android)

Requires:
- `sharp_split_part1.pte` pushed to device
- `python_part1_baseline.json` in androidTest assets (from step 1)

```bash
./push_sharp_executorch_models.sh executorch_models
./gradlew connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.furnit.android.ExecutorchSharpParityTest
```

**Note:** Build may fail if BlasNeonTest (native) is included. Run only this test class.

## Mapping: Python → Android

| Python                    | Kotlin / Android                          |
|---------------------------|-------------------------------------------|
| PIL Image → resize 1536   | Bitmap.createScaledBitmap(1536, 1536)     |
| Extract patch [0:384,0:384] | Bitmap.createBitmap(scaled, 0, 0, 384, 384) |
| HWC→CHW, /255             | preprocessPatch (RGB channels, /255f)     |
| executorch Runtime        | Module.load(path)                         |
| forward_method.execute([tensor]) | module.forward(EValue.from(tensor)) |
| outputs[0], outputs[1]    | outputs[0].toTensor(), outputs[1].toTensor() |
