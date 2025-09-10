# Camera Permissions Setup

## Overview
The AR feature requires camera access to capture furniture images. This needs to be configured in the project settings.

## Configuration Steps

### Option 1: Using Xcode Project Settings
1. Open Furnit.xcodeproj in Xcode
2. Select the Furnit target
3. Go to the "Info" tab
4. Add the following keys:
   - **NSCameraUsageDescription**: "This app uses the camera to capture furniture objects for AR placement in the 3D room viewer."
   - **NSPhotoLibraryUsageDescription**: "This app may access your photo library to process furniture images for AR placement."

### Option 2: Using Build Settings
1. Open project in Xcode
2. Select the Furnit target
3. Go to "Build Settings"
4. Search for "Info.plist"
5. Add privacy descriptions in the build settings

### Option 3: Manual Info.plist (if needed)
If the project requires a manual Info.plist file, create `Furnit/Info.plist` with:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSCameraUsageDescription</key>
    <string>This app uses the camera to capture furniture objects for AR placement in the 3D room viewer.</string>
    <key>NSPhotoLibraryUsageDescription</key>
    <string>This app may access your photo library to process furniture images for AR placement.</string>
</dict>
</plist>
```

## Testing Camera Permissions
- The `ARCameraManager` class handles permission requests automatically
- Camera access is requested when AR mode is activated
- Users will see a system dialog requesting camera permission

## Troubleshooting
- If camera doesn't work, check that permissions are properly configured
- Test on actual device (camera not available in simulator)
- Check console logs for permission-related errors