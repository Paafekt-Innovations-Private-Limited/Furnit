import SwiftUI
import RoomPlan

// MARK: - Room Capture View
@available(iOS 16.0, *)
struct RoomCaptureView: View {
    @StateObject private var captureManager = RoomCaptureManager()
    @Environment(\.dismiss) private var dismiss
    
    @State private var showInstructions = true
    @State private var showSaveDialog = false
    @State private var roomName = ""
    @State private var showSaveAlert = false
    @State private var saveAlertMessage = ""
    @State private var scanningStarted = false
    
    var body: some View {
        ZStack {
            // RoomPlan Capture Session View
            if scanningStarted {
                RoomCaptureViewRepresentable(captureManager: captureManager)
                    .ignoresSafeArea()
                    .transition(.opacity)
            } else {
                // Loading/Setup view
                Color.black
                    .ignoresSafeArea()
                    .overlay(
                        VStack {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(1.5)
                            Text("Initializing camera...")
                                .foregroundColor(.white)
                                .padding(.top)
                        }
                    )
            }
            
            // Live Feedback Overlay (shows during scanning)
            if captureManager.isSessionRunning && !showInstructions && !captureManager.isProcessing {
                VStack {
                    // Top instruction bar
                    VStack(spacing: 8) {
                        Text(captureManager.statusMessage)
                            .font(.headline)
                            .foregroundColor(.white)
                        
                        if !captureManager.instruction.isEmpty {
                            Text(captureManager.instruction)
                                .font(.subheadline)
                                .foregroundColor(.white.opacity(0.8))
                        }
                        
                        if !captureManager.feedbackText.isEmpty {
                            Text(captureManager.feedbackText)
                                .font(.caption)
                                .foregroundColor(.green)
                        }
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(12)
                    .padding()
                    
                    Spacer()
                }
            }
            
            // Instructions Overlay
            if showInstructions && scanningStarted {
                instructionsOverlay
            }
            
            // Processing Overlay
            if captureManager.isProcessing {
                processingOverlay
            }
            
            // Controls at bottom
            if !captureManager.isProcessing && scanningStarted {
                VStack {
                    Spacer()
                    controlsView
                }
            }
        }
        .navigationTitle("3D Room Scan")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Done") {
                    // Stop session first
                    captureManager.stopCaptureSession()
                    // If no room was captured, just go back
                    if captureManager.capturedRoom == nil && captureManager.finalScene == nil {
                        dismiss()
                    }
                    // Otherwise the save dialog will show via onChange
                }
                .disabled(captureManager.isProcessing || !scanningStarted || captureManager.captureSession == nil)
            }
        }
        .sheet(isPresented: $showSaveDialog) {
            saveRoomSheet
        }
        .alert("Room Save", isPresented: $showSaveAlert) {
            Button("OK", role: .cancel) {
                if saveAlertMessage.contains("successfully") {
                    dismiss()
                }
            }
        } message: {
            Text(saveAlertMessage)
        }
        .onAppear {
            // Delay to ensure view is ready
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                withAnimation {
                    scanningStarted = true
                }
            }
        }
        .onChange(of: captureManager.finalScene) { newValue in
            // Show save dialog when room processing is complete
            if newValue != nil {
                showSaveDialog = true
            }
        }
    }
    
    // MARK: - Instructions Overlay
    private var instructionsOverlay: some View {
        VStack(spacing: 20) {
            Spacer()
            
            VStack(spacing: 16) {
                Image(systemName: "camera.metering.center.weighted")
                    .font(.system(size: 60))
                    .foregroundColor(.white)
                
                Text("How to Scan Your Room")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                
                VStack(alignment: .leading, spacing: 12) {
                    InstructionRow(icon: "1.circle.fill", text: "Point your camera at the floor")
                    InstructionRow(icon: "2.circle.fill", text: "Slowly pan across the room")
                    InstructionRow(icon: "3.circle.fill", text: "Capture walls, ceiling, and furniture")
                    InstructionRow(icon: "4.circle.fill", text: "Tap 'Done' when complete")
                }
                .padding(.horizontal, 24)
                
                Button(action: {
                    withAnimation {
                        showInstructions = false
                        // Start the capture session when user clicks Start Scanning
                        if captureManager.captureSession == nil {
                            _ = captureManager.startCaptureSession()
                        }
                    }
                }) {
                    Text("Start Scanning")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(12)
                }
                .padding(.horizontal, 40)
                .padding(.top, 8)
            }
            .padding(32)
            .background(Color.black.opacity(0.85))
            .cornerRadius(20)
            .padding()
            
            Spacer()
        }
    }
    
    // MARK: - Processing Overlay
    private var processingOverlay: some View {
        ZStack {
            Color.black.opacity(0.8)
                .ignoresSafeArea()
            
            VStack(spacing: 24) {
                ProgressView(value: captureManager.exportProgress, total: 1.0)
                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    .scaleEffect(2)
                
                Text(captureManager.statusMessage)
                    .font(.headline)
                    .foregroundColor(.white)
                
                if captureManager.exportProgress > 0 {
                    Text("\(Int(captureManager.exportProgress * 100))%")
                        .font(.title2)
                        .foregroundColor(.white)
                }
            }
            .padding(40)
            .background(Color.black.opacity(0.9))
            .cornerRadius(20)
        }
    }
    
    // MARK: - Controls View
    private var controlsView: some View {
        VStack(spacing: 12) {
            if let room = captureManager.capturedRoom {
                // Show room statistics
                VStack(spacing: 8) {
                    HStack {
                        StatBadge(icon: "cube", count: room.walls.count, label: "Walls")
                        StatBadge(icon: "sofa", count: room.objects.count, label: "Objects")
                        StatBadge(icon: "door.left.hand.open", count: room.doors.count, label: "Doors")
                        StatBadge(icon: "window.vertical.open", count: room.windows.count, label: "Windows")
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical, 8)
                .background(.ultraThinMaterial)
                .cornerRadius(12)
            } else if captureManager.isSessionRunning {
                // Show scanning indicator
                HStack {
                    Image(systemName: "dot.radiowaves.left.and.right")
                        .foregroundColor(.green)
                        .symbolEffect(.pulse)
                    Text("Scanning in progress...")
                        .font(.caption)
                        .foregroundColor(.white)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.7))
                .cornerRadius(20)
            }
            
            HStack(spacing: 16) {
                Button(action: {
                    showInstructions = true
                }) {
                    Label("Help", systemImage: "questionmark.circle")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.blue.opacity(0.8))
                        .cornerRadius(12)
                }
            }
            .padding(.horizontal)
        }
        .padding(.bottom, 20)
    }
    
    // MARK: - Save Room Sheet
    private var saveRoomSheet: some View {
        NavigationView {
            VStack(spacing: 24) {
                Image(systemName: "cube.transparent")
                    .font(.system(size: 80))
                    .foregroundColor(.blue)
                    .padding(.top, 40)
                
                Text("Save Your Room Scan")
                    .font(.title2)
                    .fontWeight(.bold)
                
                if let _ = captureManager.capturedRoom {
                    Text(captureManager.getRoomStatistics())
                        .font(.body)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Room Name")
                        .font(.headline)
                    
                    TextField("e.g., Living Room", text: $roomName)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .autocapitalization(.words)
                }
                .padding(.horizontal, 32)
                
                Button(action: saveRoom) {
                    if roomName.isEmpty {
                        Text("Enter a name to save")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.gray)
                            .cornerRadius(12)
                    } else {
                        Text("Save Room")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(12)
                    }
                }
                .disabled(roomName.isEmpty)
                .padding(.horizontal, 32)
                
                Spacer()
            }
            .navigationTitle("Save Room")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Skip") {
                        showSaveDialog = false
                        roomName = ""
                        // Go back to the list screen without saving
                        dismiss()
                    }
                }
            }
        }
    }
    
    // MARK: - Save Room
    private func saveRoom() {
        print("💾 [RoomCaptureView] Saving room: \(roomName)")
        showSaveDialog = false
        
        captureManager.saveRoomToLibrary(name: roomName) { success, error in
            if success {
                saveAlertMessage = "Room '\(roomName)' saved successfully!"
                showSaveAlert = true
                print("✅ [RoomCaptureView] Room saved successfully")
            } else {
                saveAlertMessage = "Failed to save room: \(error ?? "Unknown error")"
                showSaveAlert = true
                print("❌ [RoomCaptureView] Save failed: \(error ?? "unknown")")
            }
            roomName = ""
        }
    }
}

// MARK: - Instruction Row
struct InstructionRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(.blue)
            Text(text)
                .foregroundColor(.white)
            Spacer()
        }
    }
}

// MARK: - Stat Badge
struct StatBadge: View {
    let icon: String
    let count: Int
    let label: String
    
    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption)
                Text("\(count)")
                    .font(.headline)
            }
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - RoomPlan View Representable
@available(iOS 16.0, *)
struct RoomCaptureViewRepresentable: UIViewRepresentable {
    @ObservedObject var captureManager: RoomCaptureManager
    
    func makeUIView(context: Context) -> RoomPlan.RoomCaptureView {
        let captureView = RoomPlan.RoomCaptureView()
        captureView.delegate = context.coordinator
        
        // Don't start session here - wait for user to click "Start Scanning"
        
        return captureView
    }
    
    func updateUIView(_ uiView: RoomPlan.RoomCaptureView, context: Context) {
        // No updates needed
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(captureManager: captureManager)
    }
    
    @objc(RoomCaptureViewCoordinator)
    class Coordinator: NSObject, RoomCaptureViewDelegate, RoomCaptureSessionDelegate {
        let captureManager: RoomCaptureManager
        
        init(captureManager: RoomCaptureManager) {
            self.captureManager = captureManager
            super.init()
            
            // Set ourselves as the session delegate if session exists
            if let session = captureManager.captureSession {
                session.delegate = self
            }
        }
        
        // NSCoding conformance (required but not used in our case)
        required init?(coder: NSCoder) {
            fatalError("init(coder:) has not been implemented - Coordinator is not meant to be archived")
        }
        
        @objc func encode(with coder: NSCoder) {
            // Not implemented - Coordinator is not meant to be archived
        }
        
        // MARK: - RoomCaptureViewDelegate
        
        func captureView(shouldPresent roomDataForProcessing: CapturedRoomData, error: Error?) -> Bool {
            print("📊 [RoomCaptureView] Room data ready for processing")
            if let error = error {
                print("❌ Error: \(error.localizedDescription)")
                captureManager.updateInstruction("Error: \(error.localizedDescription)")
                return false
            }
            return true
        }
        
        func captureView(didPresent processedResult: CapturedRoom, error: Error?) {
            print("✅ [RoomCaptureView] Room capture complete")
            if let error = error {
                print("❌ Error: \(error.localizedDescription)")
                captureManager.updateInstruction("Error: \(error.localizedDescription)")
            } else {
                Task {
                    await captureManager.processCapturedRoom(processedResult)
                }
            }
        }
        
        // MARK: - RoomCaptureSessionDelegate
        
        func captureSession(_ session: RoomCaptureSession, didUpdate room: CapturedRoom) {
            // Update live feedback
            DispatchQueue.main.async {
                let wallCount = room.walls.count
                let objectCount = room.objects.count
                
                if wallCount > 0 || objectCount > 0 {
                    self.captureManager.feedbackText = "Detected: \(wallCount) walls, \(objectCount) objects"
                }
                
                // Update instructions based on what's been captured
                if wallCount == 0 {
                    self.captureManager.instruction = "Point at walls to detect them"
                } else if wallCount < 3 {
                    self.captureManager.instruction = "Continue scanning walls"
                } else {
                    self.captureManager.instruction = "Good! Keep scanning for more details"
                }
            }
        }
        
        func captureSession(_ session: RoomCaptureSession, didProvide instruction: RoomCaptureSession.Instruction) {
            // Update with RoomPlan's built-in instructions
            var instructionText = ""
            
            switch instruction {
            case .moveCloseToWall:
                instructionText = "Move closer to the wall"
            case .moveAwayFromWall:
                instructionText = "Move away from the wall"
            case .slowDown:
                instructionText = "Slow down your movement"
            case .turnOnLight:
                instructionText = "Turn on more lights"
            case .normal:
                instructionText = "Continue scanning"
            case .lowTexture:
                instructionText = "Point at areas with more detail"
            @unknown default:
                instructionText = "Continue scanning"
            }
            
            captureManager.updateInstruction(instructionText)
        }
    }
}

// MARK: - Preview (for devices with RoomPlan support)
@available(iOS 16.0, *)
struct RoomCaptureView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            RoomCaptureView()
        }
    }
}
