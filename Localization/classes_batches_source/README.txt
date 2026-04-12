Human-translated YOLO class labels (batch files).

batchN/<lang>.txt (N = 0, 1, 2, ...)
  - Exactly 100 lines. Line i (0-based) maps to class key str(N*100 + i).
  - batch0 -> keys "0".."99" -> emitted as Localization/classes_batches/<lang>/000.json
  - batch1 -> keys "100".."199" -> 001.json, etc.
  - One file per locale when that batch is translated (batch 0 requires all locales).

After editing line files, from repo root:
  python3 scripts/emit_classes_batch_txt_to_json.py
  python3 scripts/emit_classes_batch_txt_to_json.py --batch 1
  python3 scripts/merge_classes_i18n_batches.py

merge_classes_i18n_batches.py starts from en.lproj/classes.json and overlays each
Localization/classes_batches/<lang>/*.json fragment (sorted by filename).
Later fragments override earlier keys. Untranslated keys stay English.

For batch >= 1, locales without batchN/<lang>.txt are skipped until you add them.
