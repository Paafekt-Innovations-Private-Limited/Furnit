# Android localization

Paafekt Android has a complete default catalog plus 13 localized catalogs. Every
localized catalog must contain the same resource keys as
[`values/strings.xml`](../app/src/main/res/values/strings.xml); release builds do not
rely on silent English fallback for a declared language.

## Supported languages

| Language | Android locale | Resource directory |
|---|---|---|
| English | `en` | `values/` |
| Arabic | `ar` | `values-ar/` |
| Bengali | `bn` | `values-bn/` |
| German | `de` | `values-de/` |
| Spanish | `es` | `values-es/` |
| Spanish (Mexico) | `es-MX` | `values-es-rMX/` |
| French | `fr` | `values-fr/` |
| Hindi | `hi` | `values-hi/` |
| Kannada | `kn` | `values-kn/` |
| Malayalam | `ml` | `values-ml/` |
| Tamil | `ta` | `values-ta/` |
| Telugu | `te` | `values-te/` |
| Simplified Chinese | `zh-CN` | `values-zh-rCN/` |
| Traditional Chinese | `zh-TW` | `values-zh-rTW/` |

[`locales_config.xml`](../app/src/main/res/xml/locales_config.xml) publishes this list
to Android. On Android 13 and newer, users can choose a Paafekt language under the
system app-language settings (the exact OEM path varies; commonly **Settings → Apps →
Paafekt → Language**). Choosing **System default** follows the device language list.

## Runtime rules

- User-visible copy belongs in `strings.xml`; counts use `<plurals>`.
- Date and time in generated room names use Android locale-aware date/time formats.
- Room measurements use localized formatted resources, not `Locale.US` UI output.
- The country picker asks Android for localized country names. Login preselection
  prefers the current mobile-network country, then SIM country, then app/device locale;
  users can always override it.
- Detector labels remain stable English internally because measurement/style logic uses
  those identifiers. UI surfaces translate supported detector class IDs through
  `FurnitureClassNames`.
- Arabic uses mirrored back/forward controls and auto-mirrored vector arrows.
- Technical logs, metadata keys, model identifiers, filenames, and crash stack traces
  remain stable implementation data and are not UI translations.

## Verification

Run before committing any string or locale change:

```bash
python3 scripts/verify_i18n.py
./gradlew :app:lintDebug :app:testDebugUnitTest :app:assembleDebug
```

The i18n verifier fails on missing/extra/duplicate resources, bad format arguments,
incorrect plural sets, leaked translation tokens, declared-locale drift, disabled
`MissingTranslation` lint, and direct visible English literals in common Android UI
APIs. Gradle lint remains the platform-level backstop.

When adding a resource, add it to every declared locale in the same change. When adding
a language, update the resource directory, `locales_config.xml`, the verifier locale
map, this table, and the on-device language/RTL smoke tests.

## Device smoke test

1. Select each language from Android's Paafekt language setting and relaunch the app.
2. Exercise login, country picker, OTP/manual entry, home, photo-to-room, room viewer,
   Furniture Fit, Settings, Help, Credits, Licenses, sharing, and error dialogs.
3. For Arabic, confirm navigation direction, toolbar placement, and mixed phone-number
   text remain usable.
4. Confirm localized dates, decimal separators, plurals, country names, generated room
   names, detector labels, and share/email chooser titles.
5. Keep manual OTP entry working regardless of keyboard/autofill behavior.
