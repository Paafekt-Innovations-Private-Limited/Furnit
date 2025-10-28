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
    @State private var waitingForDelegate = false
    @State private var sessionReady = false  // ✅ Track when session is ready
    
    var body: some View {
        ZStack {
            // RoomPlan Capture Session View
            if scanningStarted {
                RoomCaptureViewRepresentable(
                    captureManager: captureManager,
                    onSessionReady: { // ✅ Callback when session is ready
                        sessionReady = true
                        print("📹 [RoomCaptureView] Session is ready!")
                    }
                )
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
            
            // Instructions Overlay - shows after session is ready
            if showInstructions && sessionReady {  // ✅ Only show when session is ready
                instructionsOverlay
            }
            
            // Processing Overlay
            if captureManager.isProcessing {
                processingOverlay
            }
            
            // Controls at bottom
            if !captureManager.isProcessing && sessionReady {  // ✅ Only show controls when ready
                VStack {
                    Spacer()
                    controlsView
                }
            }
        }
        .navigationTitle("3D Room Scan")
        .navigationBarTitleDisplayMode(.inline)
        .interactiveDismissDisabled(waitingForDelegate || captureManager.isProcessing)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Done") {
                    // Mark that we're waiting for delegate
                    waitingForDelegate = true
                    
                    // Stop the session - delegate will receive final data
                    captureManager.stopCaptureSession()
                    
                    // Set a timeout in case delegate never responds
                    DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
                        if captureManager.capturedRoom == nil && captureManager.finalScene == nil {
                            print("⏱️ [RoomCaptureView] Timeout - no room data received")
                            waitingForDelegate = false
                            captureManager.cleanupSession()
                            dismiss()
                        }
                    }
                }
                .disabled(captureManager.isProcessing || !captureManager.isSessionRunning)
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
            print("🎬 [RoomCaptureView] View appeared, starting capture setup...")
            // Start immediately - no delay needed
            scanningStarted = true
        }
        .onDisappear {
            if !waitingForDelegate {
                captureManager.cleanupSession()
                print("🧹 [RoomCaptureView] Cleaned up resources on disappear")
            } else {
                print("⏸️ [RoomCaptureView] Skipping cleanup - waiting for delegate")
            }
        }
        .onChange(of: captureManager.finalScene) { oldValue, newValue in
            if newValue != nil {
                waitingForDelegate = false
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
                        // Signal that we want to start scanning
                        _ = captureManager.startCaptureSession()
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
            Color.black.opacity(0.9)
                .ignoresSafeArea()
            
            VStack(spacing: 30) {
                // Processing animation
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    .scaleEffect(2)
                
                VStack(spacing: 12) {
                    Text(captureManager.statusMessage)
                        .font(.title2)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                    
                    if captureManager.exportProgress > 0 {
                        ProgressView(value: captureManager.exportProgress, total: 1.0)
                            .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                            .frame(width: 200)
                        
                        Text("\(Int(captureManager.exportProgress * 100))%")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                }
                
                // Stats if available
                if let room = captureManager.capturedRoom {
                    VStack(spacing: 16) {
                        HStack(spacing: 32) {
                            StatBadge(icon: "rectangle.split.3x3", count: room.walls.count, label: "Walls")
                            StatBadge(icon: "cube", count: room.objects.count, label: "Objects")
                            StatBadge(icon: "door.left.hand.closed", count: room.doors.count, label: "Doors")
                        }
                        .padding(.horizontal)
                        
                        if let dims = captureManager.getRoomDimensions() {
                            Text("Room Size: \(String(format: "%.1f", dims.width))m × \(String(format: "%.1f", dims.depth))m")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    }
                    .padding()
                    .background(Color.white.opacity(0.1))
                    .cornerRadius(12)
                }
            }
            .padding()
        }
        .transition(.opacity)
    }
    
    // MARK: - Controls View
    private var controlsView: some View {
        HStack(spacing: 40) {
            // Info button
            Button(action: {
                withAnimation {
                    showInstructions.toggle()
                }
            }) {
                Image(systemName: "info.circle.fill")
                    .font(.title)
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.black.opacity(0.5))
                    .clipShape(Circle())
            }
            
            // Stats display
            if captureManager.isSessionRunning {
                Text(captureManager.feedbackText.isEmpty ? "Scanning..." : captureManager.feedbackText)
                    .font(.caption)
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(Color.black.opacity(0.5))
                    .cornerRadius(20)
            }
        }
        .padding(.bottom, 30)
    }
    
    // MARK: - Save Room Sheet
    private var saveRoomSheet: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Success indicator
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 80))
                    .foregroundColor(.green)
                    .padding(.top, 40)
                
                Text("Room Scanned Successfully!")
                    .font(.title2)
                    .fontWeight(.bold)
                
                // Room statistics - FIX: Don't use optional binding since it returns non-optional String
                let stats = captureManager.getRoomStatistics()
                Text(stats)
                    .font(.body)
                    .multilineTextAlignment(.leading)
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(12)
                    .padding(.horizontal)
                
                // Name input
                VStack(alignment: .leading, spacing: 8) {
                    Text("Room Name")
                        .font(.headline)
                    TextField("Enter room name", text: $roomName)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                .padding(.horizontal)
                
                Spacer()
                
                // Action buttons
                HStack(spacing: 20) {
                    Button("Cancel") {
                        showSaveDialog = false
                        captureManager.cleanupSession()
                        dismiss()
                    }
                    .foregroundColor(.red)
                    
                    Button("Save Room") {
                        saveRoom()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(roomName.isEmpty)
                }
                .padding(.bottom, 30)
            }
            .navigationTitle("Save Room")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
    
    // MARK: - Save Room
    private func saveRoom() {
        guard !roomName.isEmpty else { return }
        
        showSaveDialog = false
        
        captureManager.saveRoomToLibrary(name: roomName) { success, error in
            if success {
                saveAlertMessage = "Room '\(roomName)' saved successfully!"
                showSaveAlert = true
                print("✅ [RoomCaptureView] Room saved successfully")
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self.captureManager.cleanupSession()
                }
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
    var onSessionReady: (() -> Void)?  // ✅ Callback when ready
    
    func makeUIView(context: Context) -> RoomPlan.RoomCaptureView {
        print("🎥 [RoomCaptureViewRepresentable] Creating capture view...")
        
        let captureView = RoomPlan.RoomCaptureView()
        captureView.delegate = context.coordinator
        
        // Store the session in the manager
        captureManager.initializeSession(from: captureView.captureSession)
        
        // Set the session delegate
        captureView.captureSession.delegate = context.coordinator
        
        print("📹 [RoomCaptureViewRepresentable] Capture view created and delegates set")
        
        // Notify that session is ready
        DispatchQueue.main.async {
            onSessionReady?()
        }
        
        return captureView
    }
    
    func updateUIView(_ uiView: RoomPlan.RoomCaptureView, context: Context) {
        // Start the session only when manager says to and we haven't started yet
        if captureManager.isSessionRunning && !context.coordinator.hasStartedSession {
            print("🚀 [RoomCaptureViewRepresentable] Starting capture session...")
            
            var configuration = RoomCaptureSession.Configuration()
            configuration.isCoachingEnabled = true
            
            // Run the session
            uiView.captureSession.run(configuration: configuration)
            context.coordinator.hasStartedSession = true
            
            print("✅ [RoomCaptureViewRepresentable] Session started successfully")
        }
    }
    
    static func dismantleUIView(_ uiView: RoomPlan.RoomCaptureView, coordinator: Coordinator) {
        print("🧹 [RoomCaptureViewRepresentable] Dismantling view...")
        uiView.captureSession.delegate = nil
        uiView.delegate = nil
        uiView.captureSession.stop()
        print("✅ [RoomCaptureViewRepresentable] View dismantled")
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(captureManager: captureManager)
    }
    
    // MARK: - Coordinator
    // ✅ FIX: Add @objc attribute with stable name for NSCoding archiving
    @objc(RoomCaptureViewCoordinator)
    class Coordinator: NSObject, RoomCaptureViewDelegate, RoomCaptureSessionDelegate {
        let captureManager: RoomCaptureManager
        var hasStartedSession = false
        
        init(captureManager: RoomCaptureManager) {
            self.captureManager = captureManager
            super.init()
            print("🎯 [Coordinator] Initialized")
        }
        
        // ✅ FIX: Add required NSCoding methods (even though we don't use them)
        required init?(coder: NSCoder) {
            fatalError("init(coder:) has not been implemented - Coordinator is not meant to be archived")
        }
        
        @objc func encode(with coder: NSCoder) {
            // Not implemented - Coordinator is not meant to be archived
        }
        
        deinit {
            print("💀 [Coordinator] Deinitializing...")
            captureManager.captureSession?.delegate = nil
        }
        
        // MARK: - RoomCaptureViewDelegate
        
        func captureView(shouldPresent roomDataForProcessing: CapturedRoomData, error: Error?) -> Bool {
            print("📊 [Coordinator] Room data ready for processing")
            if let error = error {
                print("❌ [Coordinator] Error: \(error.localizedDescription)")
                return false
            }
            return true
        }
        
        func captureView(didPresent processedResult: CapturedRoom, error: Error?) {
            print("✅ [Coordinator] Room capture complete - received processed result")
            if let error = error {
                print("❌ [Coordinator] Processing error: \(error.localizedDescription)")
                captureManager.updateInstruction("Error: \(error.localizedDescription)")
            } else {
                print("🎉 [Coordinator] Successfully captured room:")
                print("   - Walls: \(processedResult.walls.count)")
                print("   - Objects: \(processedResult.objects.count)")
                
                Task {
                    await captureManager.processCapturedRoom(processedResult)
                }
            }
        }
        
        // MARK: - RoomCaptureSessionDelegate
        
        func captureSession(_ session: RoomCaptureSession, didUpdate room: CapturedRoom) {
            DispatchQueue.main.async {
                let wallCount = room.walls.count
                let objectCount = room.objects.count
                
                if wallCount > 0 || objectCount > 0 {
                    self.captureManager.feedbackText = "Detected: \(wallCount) walls, \(objectCount) objects"
                }
                
                // Dynamic instructions based on capture progress
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
        
        func captureSession(_ session: RoomCaptureSession, didStartWith configuration: RoomCaptureSession.Configuration) {
            print("🎬 [Coordinator] Session started with configuration")
            DispatchQueue.main.async {
                self.captureManager.statusMessage = "Scanning room..."
            }
        }
        
        func captureSession(_ session: RoomCaptureSession, didEndWith data: CapturedRoomData?, error: Error?) {
            print("🏁 [Coordinator] Session ended")
            if let error = error {
                print("❌ [Coordinator] Session ended with error: \(error)")
            }
        }
    }
}

// MARK: - Preview
@available(iOS 16.0, *)
struct RoomCaptureView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            RoomCaptureView()
        }
    }
}
