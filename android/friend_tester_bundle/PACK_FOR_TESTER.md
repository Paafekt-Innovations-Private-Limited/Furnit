# What to zip for a tester (developer checklist)

Your friend should **not** clone the repo. Give them a single zip (or Drive folder) with:

| Include | Notes |
|--------|--------|
| `friend_tester_bundle/install_furnit_apk.sh` | from this repo folder |
| `friend_tester_bundle/push_furnit_sharp_models.sh` | same |
| `friend_tester_bundle/adb_common.sh` | **required** (device pick: phone vs emulator) |
| `friend_tester_bundle/README_FRIEND.md` | same (rename to `README.txt` if they prefer) |
| `models_cpuvulkan_hybrid/` | full hybrid `.pte` set (same as `android/models_cpuvulkan_hybrid/` on your machine) |
| `*.apk` | e.g. `friend-nomodels-*-app-etVulkan-arm64-v8a-debug.apk` from `assemble_friend_apk_without_models.sh` |

Suggested layout **inside the zip**:

```
Furnit_tester/
  README_FRIEND.md
  adb_common.sh
  install_furnit_apk.sh
  push_furnit_sharp_models.sh
  furnit.apk
  models_cpuvulkan_hybrid/
    sharp_split_part*.pte
    ...
```

They unzip, `chmod +x *.sh`, run `./install_furnit_apk.sh`, then `./push_furnit_sharp_models.sh`.
