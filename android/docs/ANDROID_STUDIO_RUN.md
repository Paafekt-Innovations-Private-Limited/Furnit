# Running From Android Studio

## Open And Run

1. Open the `android` folder as the Android Studio project.
2. Wait for Gradle sync.
3. Select an arm64 Android device.
4. Run the `app` configuration.

## Terminal Build

```bash
./gradlew :app:assembleDebug
```

The project builds a single app variant. If the Build Variants tool window is visible in Android Studio, that is normal IDE UI; there are no old room-generation product flavors to select.

## Device Notes

The APK is configured for `arm64-v8a`. Use a physical Android phone or an ARM64 emulator image. A default x86 emulator will not be a good target for this project.

AR features require a real device with ARCore support.

## Useful Logs

```bash
adb logcat -s SinglePhotoRoom:D PhotoRoomGeneration:D RoomGenerationAssets:D GLBRoomActivity:D FurnitureFitManager:D FurnitureFitOverlay:I -v time
```

For full-video segmentation debugging, filter `GLBRoomActivity` and `FurnitureFitOverlay` together.
