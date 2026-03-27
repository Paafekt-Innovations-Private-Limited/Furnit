#!/usr/bin/env bash
# Shared by install_furnit_apk.sh and push_furnit_sharp_models.sh.
# Picks adb target: physical USB/wireless phone over emulator.
#
# Override: export ANDROID_SERIAL=<id>   # from: adb devices
#
pick_physical_serial() {
  local s
  local -a ids=()
  while read -r s; do
    [[ -n "$s" ]] && ids+=("$s")
  done < <(adb devices | awk 'NR>1 && $2=="device" {print $1}')

  if [[ ${#ids[@]} -eq 0 ]]; then
    echo "No authorized devices. Run: adb devices" >&2
    return 1
  fi

  if [[ -n "${ANDROID_SERIAL:-}" ]]; then
    for s in "${ids[@]}"; do
      if [[ "$s" == "$ANDROID_SERIAL" ]]; then
        echo "$s"
        return 0
      fi
    done
    echo "ANDROID_SERIAL=$ANDROID_SERIAL is not in 'adb devices' (or not authorized)." >&2
    return 1
  fi

  local -a physical=()
  for s in "${ids[@]}"; do
    if [[ "$s" != emulator-* ]]; then
      physical+=("$s")
    fi
  done

  if [[ ${#physical[@]} -eq 1 ]]; then
    echo "${physical[0]}"
    return 0
  fi

  if [[ ${#physical[@]} -gt 1 ]]; then
    echo "Multiple physical devices. Unplug one, or pick explicitly:" >&2
    for s in "${physical[@]}"; do
      echo "  export ANDROID_SERIAL=$s  # then re-run this script" >&2
    done
    return 1
  fi

  echo "Only an emulator is connected; Furnit scripts expect a physical phone." >&2
  echo "Quit the emulator (or disconnect it from adb) and plug in the phone, then retry." >&2
  echo "Current: adb devices" >&2
  adb devices >&2
  return 1
}
