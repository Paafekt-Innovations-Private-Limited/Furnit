# 3D math findings — what is native, what is borrowed, what to build

Recorded 2026-08-06. Sources: three in-run riders to gemini-3.6-flash, asked inside real
spectral-flip runs. Verbatim answers archived in the paafekt repo under
`docs/recordings/gemini-qa-spectral-2026-08-04.md`.

**Read this before proposing anything "quantum" for the app again.**

---

## The line

Every quantum-flavoured idea this project killed died the same way: a formalism whose
native domain is a complex Hilbert space, applied to real unstructured vectors —
embeddings, mask kernels, feature maps — where it provably reduces to cosine similarity.

3D is different, but **not because quantum applies**:

> Quantum mechanics is still borrowed and invalid here. No Planck constant, no complex
> probability amplitudes, no collapse, no superposition of macro-states. Calling spatial
> correlation "entanglement" remains metaphor inflation.
>
> **HOWEVER** — the group theory behind quantum angular momentum (SO(3), SU(2), SE(3),
> harmonic analysis on compact Lie groups) **is not borrowed. It is the literal, native
> differential geometry of 3D space.**
>
> *"When your math fails in 3D, it is usually not because you lacked quantum mechanics;
> it is because you treated SO(3) or SE(3) as if it were ℝ³ or ℝ⁴."*

## Native vs metaphor, per pipeline stage

| stage | verdict | the actual native math |
|---|---|---|
| Single-photo metric depth | **metaphor** | projective geometry; Hilbert spaces add zero |
| Camera pose / GeoCalib | **NATIVE** | calculus on SE(3) via 𝔰𝔢(3) |
| Plane fitting | **NATIVE** | Grassmannian Gr(2,3) / spherical statistics on S² |
| Gaussian splatting | **NATIVE geometry**, metaphor physics | radiance on S² via SO(3) irreps |
| Furniture pose | **NATIVE** | Bingham / Matrix-Fisher distributions on SO(3) |
| Collision & fit-checking | **NATIVE** | SE(3) action on distance fields |
| AR anchor drift | **NATIVE** | EKF / pose-graph optimisation on the SE(3) manifold |

---

## Ranked by information-per-effort (small team, Apple Silicon)

| # | item | cost | return |
|---|---|---|---|
| 1 | **𝔰𝔬(3)/𝔰𝔢(3) residuals + local tangent parameterisation** | very low (~100 lines) | **critical** |
| 2 | Tensor-Train / MPS compression of 3D fields | medium | high — Apple Silicon is bandwidth-bound |
| 3 | Bingham / Matrix-Fisher orientation distributions | medium (normaliser) | high |
| 4 | Wigner-D rotation of 3DGS SH features | low | medium |
| 5 | Invariant Point Attention for fit-checking | high | low-to-medium |
| 6 | e3nn / steerable Clebsch-Gordan networks | extreme | **near-zero** |

### #1 — why it is critical

Treating rotations as Euclidean produces three concrete failures:

1. **Gimbal lock** at pitch ±90° with Euler angles.
2. **Double cover.** `q` and `−q` are the *same* rotation, but a Gaussian in ℝ⁴ centred at
   `q` treats `−q` as infinitely far away — splitting one symmetric orientation into **two
   fake modes**. Furniture is largely symmetric (tables, boxes, chairs), so this is a live
   bug generator, not a theoretical one.
3. **Volume distortion.** The Euclidean metric does not match the invariant Haar measure
   on SO(3).

Plus: gradient descent on rotation matrices drifts off SO(3) losing orthogonality; on
quaternions it leaves S³; renormalising inside the loss explodes gradients near `q → 0`.

Keep **classical 𝔰𝔢(3) Gauss-Newton in the 60 FPS loop** — measured at <0.1 ms.

### #3 — what it buys

Models a chair whose yaw is known to ±2° but pitch/roll only to ±15°, and satisfies
`P(q) = P(−q)` natively, so symmetric objects do not collapse into antipodal modes.
Composes with **placement abstention**: refuse to auto-place and prompt a re-scan when
orientation entropy is high, rather than placing confidently into geometry we do not have.

---

## Explicitly rejected, with the reason

| idea | why |
|---|---|
| **e3nn / steerable CG nets in the render loop** | 20–100 ms/frame; variable tensor shapes fail ANE static-graph execution. Never at 60 FPS. |
| **IPA for fit-checking** | Feasible at 1–3 ms for N<128, but beaten by plain SDFs. Offline scene understanding only. |
| **"Quantum" framing of spherical harmonics** | The mechanism is real — exact rotation `c'_l = D^l(R) c_l`, and truncating degree 3→1 cuts 48 floats/splat to 12, a **75% bandwidth saving**. But it is **renaming**: Precomputed Radiance Transfer (Sloan et al., SIGGRAPH 2002) has used Wigner-D for 24 years, and SH are the Laplace–Beltrami eigenbasis — **Laplace, 1782**, 144 years before Schrödinger. Do it for LOD; do not call it quantum. |
| **Complex / phase networks for 2D masks** | Killed by our own 111.8M complex LM, whose phase channel sat at 0.5000 ± 0.0222 — exactly random init — after 98M tokens. Use **Lovász-Softmax / boundary-aware Dice** for mask quality instead. |
| **Quantum fidelity for object matching** | Provably identical to cosine (verified to 8.88e-16). Use cosine or Mahalanobis + Hungarian matching. |
| **Phase / DC-notch filtering for plane fitting** | Visual feature grids have no periodic sequence structure. Use RANSAC/MSAC or PointNet++. |
| **"GHZ" for cluster grouping** | Our own proposal, and wrong: it collapses to 2-body cosine in ℝ¹⁶⁹, the same trap `ghzResidual` fell into. The underlying idea survives as **Dynamic Kernel Aggregation** (DyCo3D CVPR 2021, DKNet ECCV 2022) — *not* Matrix NMS, which works on 2D mask overlap and never inspects kernels. |

---

## Segmentation: what the same riders said

- **Do not** gate NMS by kernel similarity — two identical chairs side by side have
  near-identical kernels, so you would delete valid duplicate objects. Spatial IoU is
  required to separate spatially distinct instances of identical appearance.
- **Do not** track by kernel cosine — kernels drift with camera motion and lighting. Use it
  as a secondary affinity weight at most.
- **Do** consider gating *cluster grouping*: `IoMin ≥ 0.12` **AND** `cos(kᵢ,kⱼ) ≥ θ`.
  Cosine in ℝ¹⁶⁹ is not transitively reachable, which is what blocks a bridging fragment
  chaining two adjacent objects into one cluster.
- **Precondition test, an afternoon, before any code:** compute cross-object kernel cosines
  on frames with two touching distinct objects. **If `cos(k_sofa, k_table) > 0.82`**, the
  dynamic head is leaning on positional encoding rather than semantic separation and the
  idea is dead.
- Transitive closure currently merges objects at **different depth planes**, so the median
  depth of a merged sofa+table cluster lands in the empty space between them — corrupting
  the AR z-ordering that cluster depth ranking depends on.

---

## iOS Metal delegate — verified against source

Asked as part of the same session, then checked independently:

- `TFLGpuDelegateOptions` has **exactly three fields** — `allow_precision_loss`,
  `wait_type`, `enable_quantization`. **No op allowlist, no denylist, no
  max-delegated-partitions, no way to pin nodes to CPU.** Verified by fetching the header.
- The **Android** GPU delegate (`TfLiteGpuDelegateOptionsV2`) *does* expose partition
  control (`max_delegated_partitions`, `FirstNLargestPartitions`). So "Android tolerates
  partial delegation, iOS does not" is **not a policy difference in our code — the two
  delegates expose different control surfaces.**
- Unverified, from model weights only, treat as hypotheses: the rank-1-vs-rank-4 broadcast
  defect; that `RELU_0_TO_1` is unmapped on Metal through 2.17.x; and that Metal uses
  native `half` accumulators where OpenCL keeps fp32 — the last of which would explain the
  observed *finite but saturated* logits exactly.
- It rated the **Core ML delegate** the cleanest iOS route, which is where the app already
  ended up.
