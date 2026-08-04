# Furnit documentation

Docs-as-code golden path:
[`READ_FIRST.md`](READ_FIRST.md) · [`overview.md`](overview.md) ·
[`architecture.md`](architecture.md) · [`adr/`](adr/) ·
[`runbooks/`](runbooks/) · [`deferred/`](deferred/)

Detailed platform documentation remains beside its owning code. A file's lifecycle
classification below matters: dated audits and research are evidence, not automatic
descriptions of the current app.

## Golden-path documents

| Document | Lifecycle | Purpose |
|---|---|---|
| [`READ_FIRST.md`](READ_FIRST.md) | Active | Current facts, settled production identity, login behavior, architecture summary, and validation rules |
| [`overview.md`](overview.md) | Active | Product/platform overview and repository map |
| [`architecture.md`](architecture.md) | Active | Architecture entry point and invariants |
| [`architecture/CODE_MAP.md`](architecture/CODE_MAP.md) | Active | Product behavior → owning code map |
| [`adr/README.md`](adr/README.md) | Active index | Architecture decision records |
| [`adr/2026-08-03-docs-as-code-organization.md`](adr/2026-08-03-docs-as-code-organization.md) | Accepted ADR | Central spine, code-local platform docs, lifecycle classification, and confirmed-only deferred ledger |
| [`runbooks/README.md`](runbooks/README.md) | Active index | Build, release, auth, memory, and review operations |
| [`deferred/README.md`](deferred/README.md) | Active index | Historical/deferred lifecycle policy |
| [`deferred/CONFIRMED_DEFERRED.md`](deferred/CONFIRMED_DEFERRED.md) | Active ledger | Evidence-backed stubs, compatibility branches, cleanup candidates, and tooling follow-ups |

## iOS implementation documents

| Document | Lifecycle | Purpose |
|---|---|---|
| [`../Furnit/docs/README.md`](../Furnit/docs/README.md) | Active index | iOS room/Furniture Fit docs and diagrams |
| [`../Furnit/docs/mask-head-accel.md`](../Furnit/docs/mask-head-accel.md) | Active reference | RTMDet mask-head performance and profiling |
| [`../Furnit/Views/FurnitureFit/README.md`](../Furnit/Views/FurnitureFit/README.md) | Active, code-local | RTMDet/Furniture Fit implementation |
| [`../Furnit/Models/RTMDet/README.md`](../Furnit/Models/RTMDet/README.md) | Active, code-local | RTMDet model package notes |
| [`IOS_FURNITURE_FIT_ONNX_STYLE_PIPELINE.md`](IOS_FURNITURE_FIT_ONNX_STYLE_PIPELINE.md) | Active reference | iOS RTMDet Core ML pipeline |
| [`IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md`](IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md) | Active reference | Room/furniture sizing and overlay math |
| [`RTMDET_IOS_SWIFT_SPIKE.md`](RTMDET_IOS_SWIFT_SPIKE.md) | Active implementation reference | Loader/export expectations and Swift postprocess status |
| [`ON_DEMAND_RESOURCES.md`](ON_DEMAND_RESOURCES.md) | Active operations | iOS RTMDet ODR and bundled-model behavior |
| [`../Furnit/diagrams/README.md`](../Furnit/diagrams/README.md) | Active diagrams | iOS room and RTMDet flows |

## Android implementation documents

| Document | Lifecycle | Purpose |
|---|---|---|
| [`../android/docs/README.md`](../android/docs/README.md) | Active index | Android code-local documentation |
| [`../android/docs/AUTHENTICATION.md`](../android/docs/AUTHENTICATION.md) | Active operations | Firebase/Play signing, locale country defaults, OTP autofill, and production smoke test |
| [`../android/docs/ANDROID_ROOM_CREATION.md`](../android/docs/ANDROID_ROOM_CREATION.md) | Active reference | Room-generation and GLB flow |
| [`../android/docs/FURNITURE_FIT_PERFORMANCE.md`](../android/docs/FURNITURE_FIT_PERFORMANCE.md) | Active reference | Android RTMDet LiteRT execution, Swift-parity frame ownership, and profiling |
| [`../android/docs/ANDROID_STUDIO_RUN.md`](../android/docs/ANDROID_STUDIO_RUN.md) | Active runbook | Android Studio/device run instructions |
| [`../android/docs/TEST_AND_SETTINGS.md`](../android/docs/TEST_AND_SETTINGS.md) | Active runbook | Room, Furniture Fit, assets, and auth smoke tests |
| [`../android/README.md`](../android/README.md) | Active overview | Compact Android architecture |
| [`../android/README_ANDROID.md`](../android/README_ANDROID.md) | Active operations | Detailed build, signing, models, and viewers |
| [`../android/diagrams/README.md`](../android/diagrams/README.md) | Active diagrams | Android room and RTMDet flows |
| [`../android/SUBMISSION_POLICY_AUDIT.md`](../android/SUBMISSION_POLICY_AUDIT.md) | Historical audit + current addendum | 2026-07-25 submission audit; 2026-08-03 production correction |

## Product, policy, release, and operations

| Document | Lifecycle | Purpose |
|---|---|---|
| [`PAAFEKT_DESIGN_SYSTEM.md`](PAAFEKT_DESIGN_SYSTEM.md) | Active reference | Paafekt UI tokens, brand assets, components, and copy |
| [`privacy.html`](privacy.html) | Active source | Repository privacy-policy source; website deployment is separate |
| [`firebase-sms-regions.md`](firebase-sms-regions.md) | Active runbook | Firebase Phone Auth SMS policy and reviewer test numbers |
| [`store-submission-status.md`](store-submission-status.md) | Active status | Last verified store/Firebase state, processed uploads, and publication boundary |
| [`apple-review-checklist.md`](apple-review-checklist.md) | Reusable runbook | Apple reviewer path, privacy labels, offline legal disclosures, and QA |
| [`android-play-review-checklist.md`](android-play-review-checklist.md) | Reusable runbook | Google Play reviewer path, Data Safety, offline legal disclosures, and notes |
| [`CHECK_APP_MEMORY.md`](CHECK_APP_MEMORY.md) | Active runbook | iOS memory checks and comparison procedure |
| [`MODEL_LICENSE_AUDIT.md`](MODEL_LICENSE_AUDIT.md) | Dated compliance evidence | Model/weights inventory, redistribution packaging, and lawyer-review triggers; re-audit when models change |

## Research and historical evidence

| Document | Lifecycle | Purpose |
|---|---|---|
| [`ROOM_3D_APPROACHES.md`](ROOM_3D_APPROACHES.md) | Research chronology | Approaches tried; includes current iOS production summary but is not the architecture entry point |
| [`UI_AUDIT.md`](UI_AUDIT.md) | Historical (2026-07-10) | Pre-design-system read-only UI snapshot |
| [`DEAD_CODE_CLEANUP.md`](DEAD_CODE_CLEANUP.md) | Historical audit | iOS removals and Periphery baseline |
| [`DEAD_CODE_CLEANUP_ANDROID.md`](DEAD_CODE_CLEANUP_ANDROID.md) | Historical audit | Android cleanup evidence and follow-ups |
| [`dev/UNUSED_CODE_PERIPHERY.md`](dev/UNUSED_CODE_PERIPHERY.md) | Tooling reference | How to run and interpret Periphery |
| [`release-status-2026-07-20.md`](release-status-2026-07-20.md) | Historical release snapshot | July build/submission evidence; superseded for current Android state |

The historical classification and reasons are mirrored in
[`deferred/README.md`](deferred/README.md). Files remain at their existing paths to
preserve inbound links; git history remains the provenance record.

## Repository convention files

| Document | Lifecycle | Purpose |
|---|---|---|
| [`../README.md`](../README.md) | Active entry point | Project summary and golden-path links |
| [`../AGENTS.md`](../AGENTS.md) | Active instructions | Agent/contributor reading order and hard rules |
| [`../CONTEXT.md`](../CONTEXT.md) | Compatibility pointer | Preserves old links while directing readers to this spine |
| [`../CLAUDE.md`](../CLAUDE.md) | Active tool-specific instructions | Compile and environment expectations |

## Documentation update rule

When behavior changes:

1. Update the nearest platform/code-local document.
2. Update `READ_FIRST.md`, architecture, status, or runbook content only if its claim
   changed.
3. Add a dated ADR for a durable architectural/documentation decision.
4. Put confirmed future cleanup/stubs in `deferred/CONFIRMED_DEFERRED.md`; do not call
   something unused without source evidence.
5. Validate local Markdown/HTML links before committing.
