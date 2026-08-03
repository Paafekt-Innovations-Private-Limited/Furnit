# Android documentation

Android-specific documentation stays beside the Android code. Start with the
repository-wide context in [`../../docs/READ_FIRST.md`](../../docs/READ_FIRST.md),
then use this index for implementation and operations.

| Document | Purpose |
|---|---|
| [`AUTHENTICATION.md`](AUTHENTICATION.md) | Firebase Phone Auth identity, Play signing certificates, country preselection, OTP autofill, and production verification. |
| [`ANDROID_ROOM_CREATION.md`](ANDROID_ROOM_CREATION.md) | Photo/manual room-generation flow, GLB output, packaged models, and viewer behavior. |
| [`ANDROID_STUDIO_RUN.md`](ANDROID_STUDIO_RUN.md) | Open, build, run, device requirements, and log filters. |
| [`TEST_AND_SETTINGS.md`](TEST_AND_SETTINGS.md) | Build and on-device smoke tests for room creation and Furniture Fit. |
| [`../README.md`](../README.md) | Compact Android architecture overview. |
| [`../README_ANDROID.md`](../README_ANDROID.md) | Detailed build, signing, assets, and release notes. |
| [`../SUBMISSION_POLICY_AUDIT.md`](../SUBMISSION_POLICY_AUDIT.md) | Dated Play submission audit with a current production addendum. |
| [`../diagrams/README.md`](../diagrams/README.md) | Android flow diagrams. |

When Android behavior changes, update the nearest document above and any affected
repository-wide index. Code and verified artifacts remain the source of truth when a
dated audit disagrees with the current implementation.
