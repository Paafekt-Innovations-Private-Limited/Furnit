# Unused Swift code (Periphery)

## How to run

1. Install [Periphery](https://github.com/peripheryapp/periphery): `brew install peripheryapp/periphery/periphery`
2. From the repo root: `./scripts/periphery-scan.sh` or `periphery scan` (uses `.periphery.yml` at the repo root).

Build requires Xcode and may take a few minutes on first run.

## Last triage (baseline)

| Result | Notes |
|--------|--------|
| **Periphery 2.x** | `periphery scan --project Furnit.xcodeproj --schemes Furnit --targets Furnit` reported **no unused declarations** in the **Furnit** app target. |
| **Manual follow-up** | Removed dead **`pickPrimaryIndex`** (+ helpers) from `FurnitureFitOnnxStylePipeline.swift` — it had no call sites; primary selection uses **`selectPrimaryIndexCoreFlow`** only. |

## Interpreting results

- **False positives** are common: `@objc`, protocol witnesses, `#Preview`, AppIntents, XCTest-only symbols. Use Periphery’s `// periphery:ignore` or config `report_exclude` / `retain_*` flags as needed.
- **FurnitTests** is a separate target; scanning **Furnit** only may flag types that exist only for tests—triage before deleting.
- **Public APIs** in app targets are often retained; use `--retain-public` only for frameworks.

## CI

Optional workflow: `.github/workflows/periphery-unused-code.yml` (`workflow_dispatch` only—macOS + full Xcode build is slow).
