# Friend / sideload APK copies (separate from Android Studio)

**Android Studio “Run”** still installs from the normal Gradle output:

`app/build/outputs/apk/<flavor>/<buildType>/`

**Friend builds** should use **`../assemble_friend_apk_with_models.sh`**. After a successful build, that script **copies** the built APK(s) **here** with a timestamp prefix so:

- Studio debug runs do **not** get confused with friend artifacts.
- Repeated friend builds **do not overwrite** each other (new timestamp each run).

Example files after a run:

- `friend-20260322-143022-app-etVulkan-arm64-v8a-debug.apk`

`*.apk` files are gitignored (see repo `.gitignore`). This folder is only on your machine.
