# Paafekt — Design System & Rebrand Reference

**Brand:** Paafekt (user-facing name). **"Furnit" is the internal codename only — it must never appear on any screen, launch, or store listing.** **Direction:** Unify both apps around the app icon's identity — **gold-on-charcoal, premium, minimal.** Kill the old rainbow/utility look.

---

## 1. Color tokens

| Role              | Hex       | Use                            |
| ----------------- | --------- | ------------------------------ |
| Background        | `#0E0F12` | App base (warm near-black)     |
| Surface           | `#1A1C20` | Cards                          |
| Surface Hi        | `#24272D` | Elevated rows, tiles           |
| Hairline          | white 8%  | Borders (not heavy strokes)    |
| Text Primary      | `#F4F3EF` | Titles, body                   |
| Text Secondary    | `#9BA0A8` | Metadata, captions             |
| **Accent (gold)** | `#C9A24B` | Primary actions, active states |
| Accent Pressed    | `#A9853A` | Pressed gold                   |
| Success           | `#3E9E6E` | State only, sparingly          |
| Danger            | `#C85A54` | Destructive                    |

One accent (gold). Semantic colors only for true state — never decoration.

## 2. Typography (System / SF · Roboto)

Display 34 Bold · Title 24 Semibold · Headline 17 Semibold · Body 15 Regular · Caption 12 · Tag 11 Semibold mono. Support Dynamic Type — add `@ScaledMetric` (iOS) to fixed-size controls (viewer pills, big icons).

## 3. Spacing / Radii / Elevation

- Spacing: 4 / 8 / 12 / 16 / 24 / 32 only.
- Radii: **12** (controls/cards), **20** (sheets), **capsule** (pills). Retire 8/10/16/25 sprawl.
- Elevation: two subtle shadow levels.

## 4. Components

- **Primary button:** gold fill, dark text, radius 12, press scale 0.98.
- **Secondary button:** ghost/outline, hairline border, text primary.
- **Card / list row:** surface + hairline; monoline icon in a `surfaceHi` tile; name primary; one metadata line secondary; file type as small uppercase mono tag (no color coding).
- **Viewer toolbar:** ONE shared blurred-glass capsule, gold active state — reused across all four viewers.
- **Helper hints:** glass chip (default), bottom scrim, or first-run coach mark — see §4.1.

Token files live in repo:

- iOS: `Furnit/Theme/Theme.swift`
- Android: `android/app/src/main/java/com/furnit/android/theme/PaafektTheme.kt` + `android/app/src/main/res/values/paafekt_colors.xml`

Values above are the source of truth.

### 4.1 Helper text / hints (viewer)

Replace floating grey caption boxes with on-brand hint chrome. Reuses the same glass treatment as `PaafektViewerToolbarCapsule` — no new visual language.

**Default — glass chip (`PaafektHintChip`):**

- Blurred-glass capsule backing (`.ultraThinMaterial` + `viewerCapsuleFill` + hairline stroke)
- Gold monoline SF Symbol gesture icon (left)
- One short caption line (`Theme.Typo.caption`, `textPrimary`)
- Anchored just above the viewer toolbar / measurement pill
- Auto-dismiss after ~3s (existing viewer hint timers unchanged)

**Variants:**

| Variant | Component | When to use |
| ------- | --------- | ----------- |
| Glass chip | `PaafektHintChip` | Transient gesture helpers (pinch, brain, ruler, full-video tap) |
| Bottom scrim | `PaafektBottomScrimHint` | Optional: float chip over a soft bottom gradient when extra legibility is needed |
| First-run coach mark | `PaafektHintCoachMark` | On-demand hint replay (`?` toolbar): chip + gold **Got it** button |

**Implementation:** `Furnit/Views/Components/PaafektViewerHint.swift`  
**Reference mockups:** `docs/design-assets/paafekt/hints/hint_in_context.png`, `hint_component_spec.png`

**Do not:** rainbow/cyan hint boxes, color-cycling tap hints, or raw `Color.black.opacity(0.78)` rounded rects.

**Dual-platform rule:** Every design-system change ships on **iOS and Android together**, screen-for-screen. iOS: `PaafektViewerHint.swift` / `Theme.swift`. Android: `PaafektHintViews.kt` + `PaafektTheme.kt` (`PaafektHintController` for View-based viewers such as `GLBRoomActivity`).

## 5. Asset inventory

### Repo paths (`docs/design-assets/paafekt/`)

**Brand (PNG + SVG):**

| File | Path |
| ---- | ---- |
| Wordmark PNG | `brand/paafekt_wordmark.png` |
| Mark PNG | `brand/paafekt_mark.png` |
| Wordmark SVG | `brand/paafekt_wordmark.svg` |
| Mark SVG | `brand/paafekt_mark.svg` |

**Icons — 16 transparent PNGs (512×512):**

`add`, `settings`, `help`, `room`, `ai`, `measure`, `camera`, `gallery`, `back`, `snapshot`, `delete`, `home`, `share`, `edit`, `close`, `layers`

- PNGs: `icons/png/<name>.png` — includes **`ai.png`**, **`snapshot.png`** (512×512 transparent, gold monoline; also exported at 96/192 for Android density buckets)
- Android drawables: `ic_ai`, `ic_snapshot` in `drawable-mdpi` (96), `drawable-xhdpi` (192), `drawable-xxxhdpi` (512)
- iOS imagesets: `PaafektIconAI`, `PaafektIconSnapshot` (template rendering, tint `#F4F3EF` / `#C9A24B` when active)
- SVGs (algorithmically traced): `icons/svg/<name>.svg` (partial set in repo; use PNGs as source of truth for SF Symbols / vector drawables)
- QA contact sheet: `icons/png/qa_contact_sheet.png`

**Hint mockups:**

| File | Path |
| ---- | ---- |
| In-context chip | `hints/hint_in_context.png` |
| Component spec (3 variants) | `hints/hint_component_spec.png` |

**iOS runtime assets:**

- `Furnit/Assets.xcassets/PaafektMark.imageset/` — login / launch mark
- `AccentColor.colorset` — gold `#C9A24B`

**Reference room renders & mockups:** design canvas (Nordic Light, Urban Loft, Home / Creation / Viewer boards).

**Rejected / delete:** old Furnit wordmark, Furnit launch screen, first-pass junk icon slices.

Icon SVGs are algorithmically traced (usable templates); for a pixel-perfect logo, redraw the wordmark in a vector tool. Use icon PNGs/SVGs as the guide to build true SF Symbols / vector drawables.

## 6. Rollout directive (for the coding agent)

Visual restyle only — **no logic/navigation/data changes.** Branch `cleanup/design-system`, one screen per commit, build + test + smoke-test after each.

1. Add token files as single source of truth.
2. Buttons → shared primary (gold) / secondary (ghost). Remove inline styling.
3. Home list → drop rainbow file colors + `💡` emoji; neutral monoline icon tile, one metadata line, mono file tag.
4. Room creation → gold primary (AI) + ghost secondary (manual).
5. Unify the four viewer toolbars into one glass capsule component (functions unchanged).
6. Settings → neutral monoline icons, gold section accents.
7. Login → dark-gold reskin, real Paafekt artwork (drop `cube.fill`), remove blue→purple gradient.
8. Global → replace `systemGroupedBackground`/`systemBackground` with tokens; standardize radii.

## 7. Rebrand rule

- **Displayed name = "Paafekt"** everywhere: `CFBundleDisplayName`, Android `app_name`, all user-facing strings, login, launch, store listing.
- **Do NOT rename** the Xcode target, source folders, or bundle id — "Furnit" stays as the internal codename only.

**Guardrails:** keep all accessibility labels; flag-don't-break anything dynamic or public; never touch the measurement/capture/export pipelines except surface styling.

## 8. Viewer controls & helper copy

**Action hierarchy (all four viewers):**

- **Primary actions — Fit Furniture, Capture:** solid gold `#C9A24B` button with dark glyph **and a short label**. Larger, clearly the focal actions.
- **Secondary nav — pan / rotate / zoom / layers:** small monoline off-white icons in the blurred-glass capsule toolbar. Quiet.
- Icon files: `ic_ai` (sparkle → Fit/AI) and `ic_snapshot` (shutter → Capture) at 512/192/96 px. Monoline, off-white default, gold when active.

**Hint chip:** frosted-glass capsule + gold monoline gesture icon + one caption line. Auto-dismiss ~3–4s.

**First-run coach mark:** frosted rounded-20 card over dimmed scrim + gold "Got it".

**Approved microcopy:** navigation "Drag to look · pinch to zoom · two fingers to move"; Fit/Capture purpose strings; first-open coach copy.

## 9. Voice & microcopy

**Voice:** calm, human, confident, concise. **Sentence case.** Verbs on buttons. No "successfully," no jargon. **Success is quiet** — transient snackbar, not blocking modal.

**Rules:**

- Sentence-case titles and body; short and warm.
- Buttons: **Save, Done, Cancel, Delete.**
- Confirmations/success → **snackbar** (reuse `PaafektRoomSavedSnackbar` / `PaafektSnackbar`), auto-dismiss. Modals for decisions and destructive actions only.
- All dialogs use Paafekt tokens: surface `#1A1C20`, radius 20, hairline border, **gold primary + ghost secondary**, monoline gold icons.

**Approved copy:**

- **Generating:** title "Building your room", subtext "This usually takes a few seconds." — gold progress + gold monoline icon (not green download).
- **Save success:** snackbar "'{name}' saved to your rooms" (auto-dismiss).
- **Name room:** title "Name your room", placeholder "e.g. Living Room", **Cancel** (ghost) · **Save** (gold).
- **Delete:** title "Delete this room?", body "This can't be undone." / **Delete** (danger) · **Cancel**.

**Landscape:** dialogs keyboard-aware and compact — field + buttons stay above IME; iOS sheet detents; Android `adjustResize` + `ScrollView`.

**UI mockups (implementation targets):** hint chip, hero actions + coach mark, immersive resting/summoned, restyled dialogs (Building / saved snackbar / landscape Name).
