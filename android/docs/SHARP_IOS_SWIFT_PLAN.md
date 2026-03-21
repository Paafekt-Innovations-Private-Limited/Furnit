# SHARP iOS Swift Plan

This document is based on the current iOS project at `/Users/al/Documents/tries01/Furnit`.

## Goal

Keep the current iOS SHARP output quality, especially the fact that it does **not** show the clean square-like edge seen on Android, while improving runtime carefully so the current `< 1 minute` room-creation flow does not regress.

## Current iOS state

Current Swift/iOS SHARP path:

- Model package: `SHARP_fp32_1536.mlpackage`
- Xcode project wires it as ODR with tag `SHARPModel`
- Service: `Furnit/Services/OnDevice/SHARPService.swift`
- Input path: single full-image Core ML model, `1536 x 1536`
- Compute units: `MLModelConfiguration.computeUnits = .cpuOnly`
- Preprocess: direct stretch of the full image to `1536 x 1536` using `CGContext.draw`
- Output path: Core ML returns 5 arrays, Swift interleaves them into `[N * 14]`, then writes PLY
- Filtering: effectively disabled; all Gaussians are kept

Important implication:

- The current iOS SHARP path is **not** using Metal for inference right now.
- The code imports `Metal`, `MetalKit`, and `MetalPerformanceShaders`, and creates an `MPSImageBilinearScale`, but inference is still forced to CPU because `computeUnits = .cpuOnly`.

## What we learned on Android that should transfer

These are the Android lessons that are useful for iOS:

1. Do **not** introduce gray/constant letterbox padding.
   Android testing showed that letterbox-style padding can produce jagged or square-edged geometry.
   iOS currently stretches to square and should keep that as the baseline.

2. Do **not** re-enable aggressive Gaussian filtering.
   Both codebases ended up preserving more of the raw SHARP output because filtering removed valid wall and ceiling splats.

3. Do **not** jump straight to tiled/split decoder work on iOS.
   Android needed split/tiled routing mostly for memory/runtime constraints on that stack.
   iOS currently has a good-quality single-model FP32 path; changing model topology is a higher-risk move.

4. First chase acceleration that does **not** change model behavior.
   On iOS that means changing Core ML execution backend before changing model structure.

## Answer to the Metal question

Yes, iOS can use Metal-backed execution, but the right first move is **not** to hand-write SHARP inference in Metal.

Recommended approach:

- Keep inference in `CoreML`
- Let `CoreML` use the best backend by changing `computeUnits`
- Use Metal or MPS manually only for preprocessing or postprocessing **if profiling proves that is worth it**

In other words:

- First choice: `CoreML + computeUnits`
- Second choice: optional Metal/MPS preprocessing optimization
- Last choice: custom Metal inference code

## Recommended plan

### Phase 0: Baseline and guardrails

Before changing runtime behavior:

- Pick 10 fixed test photos
- Save current outputs and timings
- Record:
  - total time
  - model load time
  - inference time
  - PLY write time
  - Gaussian count
  - 3 viewer screenshots per room

Success rule:

- no obvious new square-edge artifacts
- no major room-shape regression
- no major Gaussian-count collapse
- median runtime improves or stays close

### Phase 1: Low-risk performance win via Core ML compute units

Current code forces:

- `config.computeUnits = .cpuOnly`

First experiment:

- try `.all`

If `.all` is unstable or unsupported on some devices:

- fallback chain:
  - `.all`
  - `.cpuAndNeuralEngine`
  - `.cpuAndGPU`
  - `.cpuOnly`

Why this is the best first step:

- same model
- same preprocessing
- same output parsing
- same PLY writing
- much lower risk than changing model topology

What to watch:

- load failures
- prediction failures
- output dtype/layout changes
- visual quality drift

### Phase 2: Keep current preprocessing as baseline, only A/B alternatives

Current iOS preprocessing is:

- full-image stretch to `1536 x 1536`

Recommendation:

- keep this as the production baseline for now
- do **not** switch to letterbox
- if needed, A/B only against center-crop, behind a developer-only switch

Reason:

- Android evidence says letterbox is risky for SHARP quality
- iOS current stretch path is already producing the better-looking result

### Phase 3: Reduce non-inference overhead

If inference stays the main cost, leave this alone.
If profiling shows tail latency after inference matters, optimize these next:

- avoid writing 3 PLY variants on the critical path
  - write the viewer PLY first
  - generate the others lazily only when needed
- preallocate `Data` buffers more aggressively
- vectorize parts of postprocessing with `Accelerate`
- only use `MPSImageBilinearScale` for resize if resize time is material

This phase is lower risk than model conversion and should not change quality.

### Phase 4: Optional model experiments, only after Phase 1 succeeds

Only do this if Phase 1 is not enough.

Possible experiments:

- export an FP16 Core ML version of the same single-model SHARP path
- compare FP32 vs FP16 on the fixed test set
- keep the single-model path before considering any split/tiled port

Do **not** make split/tiled iOS the first optimization attempt.
That is the most likely way to reintroduce Android-style edge artifacts or seam behavior.

## Implementation order

Recommended order:

1. Add timing/profiling logs in `SHARPService.swift`
2. Change Core ML backend selection from fixed `.cpuOnly` to a fallback chain
3. Benchmark on the fixed photo set
4. If needed, optimize PLY writing and postprocessing
5. Only then consider FP16 model export
6. Leave split/tiled SHARP for last

## Concrete files to touch first

Primary iOS files:

- `Furnit/Services/OnDevice/SHARPService.swift`
- `Furnit/Views/SettingsView.swift` if a hidden developer toggle is needed
- `FurnitTests/` for a small benchmark/parity harness

Most likely first code change:

- replace the fixed `.cpuOnly` configuration with a safe compute-unit fallback strategy

## Recommendation

For iOS, the safest plan is:

- keep the current single `SHARP_fp32_1536` flow
- keep the current no-letterbox preprocessing baseline
- keep filtering disabled
- try Core ML hardware acceleration first

That should give the best chance of improving performance without losing the current iOS quality advantage.
