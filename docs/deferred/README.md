# Deferred and historical documentation

This directory indexes work that is confirmed but not part of the active product path,
plus dated documents that remain useful as evidence. Start with
[`../READ_FIRST.md`](../READ_FIRST.md) and [`../architecture.md`](../architecture.md)
for current behavior.

## Deferred implementation ledger

[`CONFIRMED_DEFERRED.md`](CONFIRMED_DEFERRED.md) records code-backed stubs,
compatibility-only parameters, unreachable branches, cleanup candidates, and tooling
follow-ups. An entry is not permission to delete or revive code; re-check call sites,
persistence contracts, tests, and product intent first.

## Historical files retained in place

These documents are not moved because existing code/docs link to them.

| Document | Why it is historical/reference-only |
|---|---|
| [`../release-status-2026-07-20.md`](../release-status-2026-07-20.md) | July 2026 build/submission snapshot; Android is now released and its Play Firebase signing fix is documented elsewhere |
| [`../UI_AUDIT.md`](../UI_AUDIT.md) | 2026-07-10 pre-design-system snapshot; it predates current Paafekt theme/assets |
| [`../DEAD_CODE_CLEANUP.md`](../DEAD_CODE_CLEANUP.md) | Dated iOS removal and Periphery evidence, not a live unused-code inventory |
| [`../DEAD_CODE_CLEANUP_ANDROID.md`](../DEAD_CODE_CLEANUP_ANDROID.md) | Dated Android deletion audit; unresolved tooling items are copied into the confirmed ledger |
| [`../ROOM_3D_APPROACHES.md`](../ROOM_3D_APPROACHES.md) | Research chronology containing retired experiments; only the explicitly labeled production section describes a current path |
| [`../../android/SUBMISSION_POLICY_AUDIT.md`](../../android/SUBMISSION_POLICY_AUDIT.md) | Pre-publication audit with a 2026-08-03 production addendum |
| [`.cursor/rules/furnit-ml.mdc`](../../.cursor/rules/furnit-ml.mdc) | Explicitly archived SHARP/ExecuTorch/Vulkan guidance; not the current room or RTMDet path |

## Admission rule

Add an item only when code or a verified artifact demonstrates one of these states:

- a callable stub explicitly returns a placeholder result;
- a branch cannot currently receive its required result/intent;
- a parameter is documented and suppressed as compatibility-only;
- a computed value is intentionally discarded;
- a tool/test gate is known and reproducible but intentionally postponed;
- a document describes a removed, dormant, or superseded path.

Do not classify a fallback, compatibility decoder, persisted enum, reflection target,
test-only symbol, or defensive branch as unused based only on a text search.
