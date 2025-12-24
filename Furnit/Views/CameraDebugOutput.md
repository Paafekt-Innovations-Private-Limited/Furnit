# Camera Debug Output Analysis

## What to Look For in Console Logs

### 1. **App Launch & View Appearance**
```
📱 SegmentExamine view appeared - requesting camera access
🔐 Camera permission result: GRANTED/DENIED
✅ Permission granted - starting camera session
```

**❌ If you see "DENIED"**: Check Settings > Privacy > Camera and enable for your app.

### 2. **Camera Setup Phase**
```
🎥 === CAMERA SETUP STARTING ===
📋 Session configuration began
📊 Initial session state:
   - isRunning: false
   - canSetSessionPreset: true
📐 Session preset set to HD 1280x720
🗑️ Cleared X inputs and Y outputs
🔍 Searching for back camera...
📱 Available camera devices:
   - Back Camera (position: 1)
✅ Found back camera: Back Camera
📊 Camera device info:
   - uniqueID: [device ID]
   - modelID: [model]
   - isConnected: true
   - isSuspended: false
🔌 Creating camera input...
✅ Camera input added successfully
📹 Configuring video output...
📊 Video output settings:
   - Pixel format: kCVPixelFormatType_32BGRA
   - Sample buffer delegate queue: sample.buffer
   - Detection queue: segmentExamineQueue
✅ Video output added successfully
🔗 Configuring video connection...
   - isActive: true
   - isEnabled: true
   - isVideoMirroringSupported: true
   - Video mirroring disabled
   - Video rotation set to 90°
✅ Video connection configured successfully
💾 Committing session configuration...
✅ Camera configured successfully [FastSAM-X PRIMARY]
🎥 === CAMERA SETUP COMPLETED ===
```

**❌ Problem Indicators:**
- "No back camera available" → Running on simulator or no camera
- "Cannot add camera input" → Device busy or permissions issue
- "Cannot add video output" → Session configuration problem

### 3. **Session Start Phase**
```
🚀 === START SESSION REQUESTED ===
📊 Pre-start session status:
   - isRunning: false
   - inputs count: 1
   - outputs count: 1
▶️ Session not running - attempting to start...
🔄 Starting session on background queue...
📊 Post-start session status:
   - isRunning: true
✅ Camera session started successfully [FastSAM-X PRIMARY]
🚀 === START SESSION REQUEST COMPLETED ===
```

**❌ Problem Indicators:**
- "isRunning: false" after start attempt → Critical failure
- "inputs count: 0" → Input not added properly
- "outputs count: 0" → Output not added properly

### 4. **Camera Preview Creation**
```
🖥️ Creating camera preview view...
   - Session running: true
   - Session inputs: 1
   - Session outputs: 1
✅ Preview layer created and added to view
📐 Preview layer frame updated: (0.0, 0.0, 375.0, 812.0)
```

**❌ Problem Indicators:**
- "Session running: false" when preview created → Session failed to start
- "Could not find preview layer in view!" → UI issue

### 5. **Video Frames Arriving**
```
📹 Frame 1 received from camera
✅ CAMERA IS WORKING - receiving video frames!
📊 Pixel buffer info: 1280x720, format: 875704422
📹 Frame 2 received from camera
📹 Frame 3 received from camera
📹 Frame 4 received from camera
📹 Frame 5 received from camera
📹 Camera working normally - suppressing frame logs...
```

**✅ Success Indicator:** If you see frames 1-5, your camera is working!

**❌ Problem Indicators:**
- No "Frame X received" messages → Delegate not being called
- "Failed to extract pixel buffer" → Sample buffer corruption

### 6. **Debug Status (Tap 🐛 button)**
```
🔍 === CAMERA DEBUG STATUS ===
📊 Session Status:
   - isRunning: true
   - sessionPreset: AVCaptureSessionPresetHD1280x720
   - inputs count: 1
   - outputs count: 1
📋 Detailed Input Information:
   Input 0: AVCaptureDeviceInput
      Device: Back Camera
      Connected: true
      Position: 1
📤 Detailed Output Information:
   Output 0: AVCaptureVideoDataOutput
      Connection active: true
      Connection enabled: true
      Video rotation: 90°
🔧 Session Capabilities:
   - canSetSessionPreset(hd1280x720): true
   - canSetSessionPreset(high): true
   - canSetSessionPreset(medium): true
   - canAddInput (test): false
   - canAddOutput (test): true
🔍 === DEBUG STATUS COMPLETE ===
```

## Quick Troubleshooting Steps

### If Camera Not Opening:

1. **Check Permissions First**
   - Look for "🔐 Camera permission result: DENIED"
   - Fix: Settings > Privacy > Camera > Enable for your app

2. **Check Device Availability**
   - Look for "❌ No back camera available"
   - Fix: Test on physical device, not simulator

3. **Check Session Status**
   - Tap the 🐛 debug button in app
   - Look for "isRunning: false" in debug output
   - Look for "inputs count: 0" or "outputs count: 0"

4. **Check for Video Frames**
   - Should see "📹 Frame 1 received from camera"
   - If not, delegate isn't being called

5. **Check Console for Errors**
   - Look for "❌ AVCaptureSession RUNTIME ERROR"
   - Look for "⚠️ SESSION INTERRUPTED"

### Most Common Issues:
- **Simulator**: Camera doesn't work on simulator
- **Permissions**: User denied camera access
- **Device Busy**: Another app is using camera
- **Background**: App was backgrounded and lost camera access