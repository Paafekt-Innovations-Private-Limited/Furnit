import SwiftUI
import RoomPlan
import Combine

// MARK: - Room Capture View (Complete Version with Navigation Loop Fix)
@available(iOS 16.0, *)
struct RoomCaptureView: View {
    @StateObject private var captureManager = RoomCaptureManager()
    @Environment(\.dismiss) private var dismiss
    @Environment(\.presentationMode) var presentationMode
    
    // Optional completion handler for parent view
    var onSaveComplete: (() -> Void)?
    
    @State private var viewState = RoomCaptureViewState()
    @State private var roomCaptureObserver: AnyCancellable?
    
    var body: some View {
        ZStack {
            // Main content
            mainContentView
            
            // Overlays
            overlayContent
        }
        .navigationTitle("3D Room Scan")
        .navigationBarTitleDisplayMode(.inline)
        .interactiveDismissDisabled(viewState.waitingForDelegate || captureManager.isProcessing || viewState.isProcessingDone)
        .toolbar {
            toolbarContent
        }
        .sheet(isPresented: $viewState.showSaveDialog) {
            SaveRoomSheet(
                captureManager: captureManager,
                roomName: $viewState.roomName,
                onSave: handleSaveRoom,
                onCancel: handleCancelSave
            )
            .presentationDetents([.medium, .large])  // ✅ Add stable size options
            .presentationDragIndicator(.hidden)       // ✅ Hide drag indicator
            .interactiveDismissDisabled(true)         // ✅ Prevent accidental dismissal
        }
        .alert("Room Save", isPresented: $viewState.showSaveAlert) {
            Button("OK", role: .cancel) {
                handleSaveAlertDismiss()
            }
        } message: {
            Text(viewState.saveAlertMessage)
        }
        .alert("Scanning Error", isPresented: $viewState.showErrorAlert) {
            Button("Try Again") {
                resetAndRetry()
            }
            Button("Cancel", role: .cancel) {
                performCompleteDismiss()
            }
        } message: {
            Text(viewState.errorMessage)
        }
        .onAppear {
            handleOnAppear()
            setupObservers()
        }
        .onDisappear {
            handleOnDisappear()
        }
        .onChange(of: captureManager.finalScene) { _, newValue in
            handleFinalSceneChange(newValue)
        }
    }
    
    // MARK: - Setup Observers
    private func setupObservers() {
        // Observe capturedRoom changes using Combine (fixes Equatable issue)
        roomCaptureObserver = captureManager.$capturedRoom
            .sink { newValue in
                if newValue != nil {
                    print("🏠 [RoomCaptureView] Room captured successfully")
                    viewState.hasScannedContent = true
                }
            }
    }
    
    // MARK: - Main Content View
    @ViewBuilder
    private var mainContentView: some View {
        if viewState.scanningStarted && !viewState.isProcessingDone {
            RoomCaptureViewRepresentable(
                captureManager: captureManager,
                onSessionReady: {
                    viewState.sessionReady = true
                    print("📹 [RoomCaptureView] Session is ready!")
                },
                onContentDetected: {
                    viewState.hasScannedContent = true
                },
                onError: { error in
                    viewState.errorMessage = error
                    viewState.showErrorAlert = true
                    print("❌ [RoomCaptureView] Error: \(error)")
                }
            )
            .ignoresSafeArea()
            .transition(.opacity)
        } else if !viewState.scanningStarted {
            LoadingView()
        }
    }
    
    // MARK: - Overlay Content
    @ViewBuilder
    private var overlayContent: some View {
        if captureManager.isSessionRunning && !viewState.showInstructions && !captureManager.isProcessing && !viewState.isProcessingDone {
            LiveFeedbackOverlay(captureManager: captureManager)
        }
        
        if viewState.showInstructions && viewState.sessionReady && !viewState.isProcessingDone {
            InstructionsOverlay(
                hasScannedContent: viewState.hasScannedContent,
                onStartScanning: {
                    withAnimation {
                        viewState.showInstructions = false
                        _ = captureManager.startCaptureSession()
                        print("🚀 [RoomCaptureView] User started scanning")
                    }
                }
            )
        }
        
        if captureManager.isProcessing || viewState.isProcessingDone {
            ProcessingOverlay(
                captureManager: captureManager,
                isProcessingDone: viewState.isProcessingDone
            )
        }
        
        if !captureManager.isProcessing && viewState.sessionReady && !viewState.showInstructions && !viewState.isProcessingDone {
            ControlsOverlay(
                captureManager: captureManager,
                hasScannedContent: viewState.hasScannedContent,
                onToggleInstructions: {
                    withAnimation {
                        viewState.showInstructions.toggle()
                    }
                }
            )
        }
    }
    
    // MARK: - Toolbar Content
    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .navigationBarTrailing) {
            // Show Done button if we're not in the middle of processing
            // OR if user cancelled save (they can press Done again)
            if !viewState.isProcessingDone || (viewState.isProcessingDone && !viewState.showSaveDialog) {
                Button("Done") {
                    handleDonePressed()
                }
                .disabled(
                    captureManager.isProcessing ||
                    (!captureManager.isSessionRunning && !viewState.isProcessingDone) ||
                    !viewState.hasScannedContent
                )
            }
        }
    }
    
    // MARK: - Action Handlers
    private func handleOnAppear() {
        print("🎬 [RoomCaptureView] View appeared, starting capture setup...")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            viewState.scanningStarted = true
        }
    }
    
    private func handleOnDisappear() {
        if !viewState.waitingForDelegate && !viewState.isProcessingDone {
            captureManager.cleanupSession()
            print("🧹 [RoomCaptureView] Cleaned up resources on disappear")
        }
        roomCaptureObserver?.cancel()
    }
    
    private func handleFinalSceneChange(_ newValue: URL?) {
        print("📄 [RoomCaptureView] finalScene changed: \(newValue != nil ? "Scene ready" : "nil")")
        if newValue != nil {
            viewState.waitingForDelegate = false
            // ✅ CRITICAL FIX: Keep isProcessingDone as true to prevent view reset
            viewState.isProcessingDone = true
            
            // ✅ Add small delay to ensure state is stable before showing dialog
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                viewState.showSaveDialog = true
            }
        }
    }
    
    // MARK: - Handle Done Pressed (with Extended Timeout and Force Processing)
    private func handleDonePressed() {
        print("✅ [RoomCaptureView] Done button pressed")
        
        // If we already have a finalScene and user cancelled save, just show the dialog again
        if viewState.isProcessingDone && captureManager.finalScene != nil && !viewState.showSaveDialog {
            print("📝 [RoomCaptureView] Showing save dialog again after cancel")
            viewState.showSaveDialog = true
            return
        }
        
        guard viewState.hasScannedContent else {
            print("⚠️ [RoomCaptureView] No content scanned yet")
            viewState.errorMessage = "Please scan your room before pressing Done"
            viewState.showErrorAlert = true
            return
        }
        
        viewState.isProcessingDone = true
        viewState.waitingForDelegate = true
        
        // Stop the capture session
        captureManager.stopCaptureSession()
        print("⏸️ [RoomCaptureView] Capture session stopped, waiting for processing...")
        
        // First check: Quick check after 2 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            if captureManager.capturedRoom != nil {
                print("✅ [RoomCaptureView] Room data received quickly")
            } else {
                print("⏳ [RoomCaptureView] Still waiting for room data...")
            }
        }
        
        // Second check: Try to force process after 5 seconds if we have any data
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
            if captureManager.finalScene == nil && captureManager.capturedRoom != nil {
                print("⚠️ [RoomCaptureView] Have room data but no scene, forcing processing...")
                captureManager.forceProcessCurrentData()
            }
        }
        
        // Final timeout: Extended to 15 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 15.0) {
            // Check if we still don't have processed data
            if captureManager.finalScene == nil {
                print("⏱️ [RoomCaptureView] Timeout after 15 seconds")
                
                // Check if we at least have captured room data
                if captureManager.capturedRoom != nil {
                    print("📦 [RoomCaptureView] Have room data, attempting force processing...")
                    captureManager.forceProcessCurrentData()
                    
                    // Give it 2 more seconds to process
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                        if captureManager.finalScene == nil {
                            // Still failed
                            viewState.errorMessage = "Room processing failed. The scan was captured but couldn't be converted to 3D. Please try again."
                            viewState.showErrorAlert = true
                            
                            viewState.waitingForDelegate = false
                            viewState.isProcessingDone = false
                            viewState.hasScannedContent = false
                            viewState.showInstructions = true
                            
                            captureManager.cleanupSession()
                        }
                    }
                } else {
                    // No room data at all
                    print("❌ [RoomCaptureView] No room data received at all")
                    
                    viewState.errorMessage = "Room scanning failed to capture data. Please ensure:\n• Good lighting in the room\n• Slow, steady movements\n• Capturing multiple walls"
                    viewState.showErrorAlert = true
                    
                    viewState.waitingForDelegate = false
                    viewState.isProcessingDone = false
                    viewState.hasScannedContent = false
                    viewState.showInstructions = true
                    
                    captureManager.cleanupSession()
                }
            } else {
                print("✅ [RoomCaptureView] Processing completed successfully")
            }
        }
    }
    
    private func resetAndRetry() {
        print("🔄 [RoomCaptureView] Resetting for retry...")
        
        viewState = RoomCaptureViewState()
        captureManager.cleanupSession()
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            viewState.scanningStarted = true
            setupObservers()
        }
    }
    
    private func handleSaveRoom() {
        guard !viewState.roomName.isEmpty else { return }
        
        print("💾 [RoomCaptureView] Saving room: \(viewState.roomName)")
        let savedName = viewState.roomName
        
        viewState.showSaveDialog = false
        viewState.isProcessingDone = true
        captureManager.statusMessage = "Saving room..."
        
        captureManager.saveRoomToLibrary(name: savedName) { success, error in
            DispatchQueue.main.async {
                if success {
                    print("✅ [RoomCaptureView] Room saved successfully")
                    self.captureManager.cleanupSession()
                    
                    // ONLY use ONE dismissal method!
                    if self.onSaveComplete != nil {
                        print("📱 Using parent's onSaveComplete")
                        self.onSaveComplete!()
                        // DON'T call performCompleteDismiss here!
                    } else {
                        print("⚠️ No parent callback, using fallback")
                        self.dismiss()
                    }
                } else {
                    self.viewState.saveAlertMessage = "Failed: \(error ?? "Unknown")"
                    self.viewState.showSaveAlert = true
                    self.viewState.isProcessingDone = false
                }
                self.viewState.roomName = ""
            }
        }
    }
    
    // MARK: - Fixed Cancel Save Handler
    private func handleCancelSave() {
        print("❌ [RoomCaptureView] User cancelled save")
        
        // Close the save dialog
        viewState.showSaveDialog = false
        
        // ✅ FIXED: Keep isProcessingDone as true to maintain the completion state
        // This prevents the view from resetting back to scanning
        viewState.isProcessingDone = true
        viewState.waitingForDelegate = false
        
        // Keep showing the processing overlay with a "cancelled" message
        captureManager.statusMessage = "Save cancelled. Tap Done to try again."
        
        // User can still tap Done button again if they want to save
    }
    
    // MARK: - Save Alert Dismiss Handler
    private func handleSaveAlertDismiss() {
        if viewState.saveAlertMessage.contains("successfully") {
            performCompleteDismiss()
        } else {
            // Failed save - stay in current view
            viewState.isProcessingDone = false
            viewState.showInstructions = false
        }
    }
    
    private func performCompleteDismiss() {
        print("🚪 [RoomCaptureView] Performing complete dismiss")
        
        // Clean up first
        captureManager.cleanupSession()
        
        // Check if we have a parent callback FIRST
        if let complete = onSaveComplete {
            print("📱 [RoomCaptureView] Using parent callback for dismissal")
            complete()
            return  // Don't try other dismiss methods
        }
        
        // Only try these if no parent callback
        print("⚠️ [RoomCaptureView] No parent callback, using environment dismiss")
        
        // Try environment dismiss
        dismiss()
        
        // Backup: Use presentationMode
        if presentationMode.wrappedValue.isPresented {
            presentationMode.wrappedValue.dismiss()
        }
    }
}

// MARK: - View State
struct RoomCaptureViewState {
    var showInstructions = true
    var showSaveDialog = false
    var roomName = ""
    var showSaveAlert = false
    var saveAlertMessage = ""
    var scanningStarted = false
    var waitingForDelegate = false
    var sessionReady = false
    var showErrorAlert = false
    var errorMessage = ""
    var hasScannedContent = false
    var isProcessingDone = false
}

// MARK: - Loading View Component
struct LoadingView: View {
    var body: some View {
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
}

// MARK: - Live Feedback Overlay
struct LiveFeedbackOverlay: View {
    @ObservedObject var captureManager: RoomCaptureManager
    
    var body: some View {
        VStack {
            feedbackBar
            Spacer()
        }
    }
    
    private var feedbackBar: some View {
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
    }
}

// MARK: - Instructions Overlay
struct InstructionsOverlay: View {
    let hasScannedContent: Bool
    let onStartScanning: () -> Void
    
    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            instructionCard
            Spacer()
        }
    }
    
    private var instructionCard: some View {
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
            
            Text("The 'Done' button will enable once scanning begins")
                .font(.caption)
                .foregroundColor(.white.opacity(0.7))
                .padding(.top, 8)
            
            Button(action: onStartScanning) {
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
    }
}

// MARK: - Processing Overlay
struct ProcessingOverlay: View {
    @ObservedObject var captureManager: RoomCaptureManager
    let isProcessingDone: Bool
    
    var body: some View {
        ZStack {
            Color.black.opacity(0.9)
                .ignoresSafeArea()
            
            VStack(spacing: 30) {
                progressSection
                
                if let room = captureManager.capturedRoom {
                    statsSection(room: room)
                }
            }
            .padding()
        }
        .transition(.opacity)
    }
    
    private var progressSection: some View {
        VStack(spacing: 12) {
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                .scaleEffect(2)
            
            Text(captureManager.statusMessage)
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(.white)
            
            if isProcessingDone && captureManager.capturedRoom == nil {
                Text("Finalizing scan data...")
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            
            if captureManager.exportProgress > 0 {
                ProgressView(value: captureManager.exportProgress, total: 1.0)
                    .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                    .frame(width: 200)
                
                Text("\(Int(captureManager.exportProgress * 100))%")
                    .font(.caption)
                    .foregroundColor(.gray)
            }
        }
    }
    
    private func statsSection(room: CapturedRoom) -> some View {
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

// MARK: - Controls Overlay
struct ControlsOverlay: View {
    @ObservedObject var captureManager: RoomCaptureManager
    let hasScannedContent: Bool
    let onToggleInstructions: () -> Void
    
    var body: some View {
        VStack {
            Spacer()
            controlsBar
        }
    }
    
    private var controlsBar: some View {
        HStack(spacing: 40) {
            Button(action: onToggleInstructions) {
                Image(systemName: "info.circle.fill")
                    .font(.title)
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.black.opacity(0.5))
                    .clipShape(Circle())
            }
            
            if captureManager.isSessionRunning {
                statusBadge
            }
        }
        .padding(.bottom, 30)
    }
    
    private var statusBadge: some View {
        VStack(spacing: 4) {
            Text(captureManager.feedbackText.isEmpty ? "Scanning..." : captureManager.feedbackText)
                .font(.caption)
                .foregroundColor(.white)
            
            if hasScannedContent {
                HStack(spacing: 4) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.caption2)
                        .foregroundColor(.green)
                    Text("Content detected - 'Done' enabled")
                        .font(.caption2)
                        .foregroundColor(.green)
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.black.opacity(0.5))
        .cornerRadius(20)
    }
}

// MARK: - Save Room Sheet (Fixed Statistics Display)
struct SaveRoomSheet: View {
    @ObservedObject var captureManager: RoomCaptureManager
    @Binding var roomName: String
    let onSave: () -> Void
    let onCancel: () -> Void
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Success header
                VStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 80))
                        .foregroundColor(.green)
                        .padding(.top, 40)
                    
                    Text("Room Scanned Successfully!")
                        .font(.title2)
                        .fontWeight(.bold)
                }
                
                // Statistics section (Fixed to show actual data)
                Group {
                    if let room = captureManager.capturedRoom {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Scan Results")
                                .font(.headline)
                            
                            HStack(spacing: 30) {
                                VStack {
                                    Text("\(room.walls.count)")
                                        .font(.title2)
                                        .fontWeight(.bold)
                                    Text("Walls")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                
                                VStack {
                                    Text("\(room.objects.count)")
                                        .font(.title2)
                                        .fontWeight(.bold)
                                    Text("Objects")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                
                                if room.doors.count > 0 {
                                    VStack {
                                        Text("\(room.doors.count)")
                                            .font(.title2)
                                            .fontWeight(.bold)
                                        Text("Doors")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                }
                                
                                if room.windows.count > 0 {
                                    VStack {
                                        Text("\(room.windows.count)")
                                            .font(.title2)
                                            .fontWeight(.bold)
                                        Text("Windows")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                }
                            }
                            .frame(maxWidth: .infinity)
                            
                            if let dims = captureManager.getRoomDimensions() {
                                Divider()
                                Text("Size: \(String(format: "%.1f", dims.width))m × \(String(format: "%.1f", dims.depth))m × \(String(format: "%.1f", dims.height))m")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                
                                let area = dims.width * dims.depth
                                Text("Area: \(String(format: "%.1f", area))m²")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(12)
                        .padding(.horizontal)
                    } else if captureManager.finalScene != nil {
                        // We have a scene but no detailed room data
                        VStack(spacing: 8) {
                            Image(systemName: "cube.box.fill")
                                .font(.largeTitle)
                                .foregroundColor(.green)
                            Text("3D model created successfully")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(12)
                        .padding(.horizontal)
                    } else {
                        // Fallback
                        VStack(spacing: 8) {
                            Image(systemName: "checkmark.circle")
                                .font(.largeTitle)
                                .foregroundColor(.green)
                            Text("Room data captured")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(12)
                        .padding(.horizontal)
                    }
                }
                
                // Room name input
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
                        // Only call onCancel, let parent handle dismissal
                        onCancel()
                    }
                    .foregroundColor(.red)
                    
                    Button("Save Room") {
                        // Don't dismiss here - let the parent handle it
                        onSave()
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
}

// MARK: - Supporting Views
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

// MARK: - RoomPlan View Representable with Workaround Coordinator
@available(iOS 16.0, *)
struct RoomCaptureViewRepresentable: UIViewRepresentable {
    @ObservedObject var captureManager: RoomCaptureManager
    var onSessionReady: (() -> Void)?
    var onContentDetected: (() -> Void)?
    var onError: ((String) -> Void)?
    
    func makeUIView(context: Context) -> RoomPlan.RoomCaptureView {
        print("🎥 [RoomCaptureViewRepresentable] Creating capture view...")
        
        let captureView = RoomPlan.RoomCaptureView()
        captureView.delegate = context.coordinator
        
        captureManager.initializeSession(from: captureView.captureSession)
        captureView.captureSession.delegate = context.coordinator
        
        print("📹 [RoomCaptureViewRepresentable] Capture view created and delegates set")
        
        DispatchQueue.main.async {
            onSessionReady?()
        }
        
        return captureView
    }
    
    func updateUIView(_ uiView: RoomPlan.RoomCaptureView, context: Context) {
        if captureManager.isSessionRunning && !context.coordinator.hasStartedSession {
            print("🚀 [RoomCaptureViewRepresentable] Starting capture session...")
            
            var configuration = RoomCaptureSession.Configuration()
            configuration.isCoachingEnabled = true
            
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
        Coordinator(
            captureManager: captureManager,
            onContentDetected: onContentDetected,
            onError: onError
        )
    }
    
    // MARK: - Coordinator with Workaround for Delegate Issues
    @objc(RoomCaptureViewCoordinator)
    class Coordinator: NSObject, RoomCaptureViewDelegate, RoomCaptureSessionDelegate {
        let captureManager: RoomCaptureManager
        var hasStartedSession = false
        var onContentDetected: (() -> Void)?
        var onError: ((String) -> Void)?
        private var hasDetectedContent = false
        
        // WORKAROUND: Save the last good room data
        private var lastCapturedRoom: CapturedRoom?
        private var updateCount = 0
        private var didReceiveFinalData = false
        
        init(captureManager: RoomCaptureManager,
             onContentDetected: (() -> Void)?,
             onError: ((String) -> Void)?) {
            self.captureManager = captureManager
            self.onContentDetected = onContentDetected
            self.onError = onError
            super.init()
            print("🎯 [Coordinator] Initialized with workaround")
        }
        
        required init?(coder: NSCoder) {
            fatalError("init(coder:) not implemented")
        }
        
        @objc func encode(with coder: NSCoder) {
            // Not implemented - not meant to be archived
        }
        
        deinit {
            print("💀 [Coordinator] Deinitializing...")
            captureManager.captureSession?.delegate = nil
        }
        
        // MARK: - RoomCaptureViewDelegate
        
        func captureView(shouldPresent roomDataForProcessing: CapturedRoomData, error: Error?) -> Bool {
            print("📊 [Coordinator] shouldPresent called - error: \(error?.localizedDescription ?? "none")")
            
            if let error = error {
                print("❌ [Coordinator] Processing error: \(error.localizedDescription)")
                onError?("Room processing error: \(error.localizedDescription)")
                return false
            }
            
            print("✅ [Coordinator] Room data accepted for processing")
            return true
        }
        
        func captureView(didPresent processedResult: CapturedRoom, error: Error?) {
            print("🏁 [Coordinator] didPresent called - THIS IS GOOD!")
            print("   - Walls: \(processedResult.walls.count)")
            print("   - Objects: \(processedResult.objects.count)")
            print("   - Has error: \(error != nil)")
            
            didReceiveFinalData = true
            
            if let error = error {
                print("❌ [Coordinator] Final processing error: \(error.localizedDescription)")
                onError?("Failed to process room: \(error.localizedDescription)")
            } else {
                print("🎉 [Coordinator] Successfully captured room via delegate")
                
                // Save the final room
                lastCapturedRoom = processedResult
                
                Task { @MainActor in
                    await captureManager.processCapturedRoom(processedResult)
                    print("✅ [Coordinator] Room processing completed")
                }
            }
        }
        
        // MARK: - RoomCaptureSessionDelegate
        
        func captureSession(_ session: RoomCaptureSession, didUpdate room: CapturedRoom) {
            updateCount += 1
            
            // WORKAROUND: Save every good room update
            if room.walls.count > 0 || room.objects.count > 0 {
                lastCapturedRoom = room
                print("💾 [Update #\(updateCount)] Saved room: W:\(room.walls.count) O:\(room.objects.count)")
            }
            
            DispatchQueue.main.async {
                let wallCount = room.walls.count
                let objectCount = room.objects.count
                
                // Notify when content is first detected
                if !self.hasDetectedContent && (wallCount > 0 || objectCount > 0) {
                    self.hasDetectedContent = true
                    self.onContentDetected?()
                    print("🎯 [Coordinator] Content detected for the first time")
                }
                
                if wallCount > 0 || objectCount > 0 {
                    self.captureManager.feedbackText = "Detected: \(wallCount) walls, \(objectCount) objects"
                    
                    // Also update the manager's capturedRoom periodically
                    self.captureManager.capturedRoom = room
                }
                
                // Update instructions
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
            didReceiveFinalData = false
            updateCount = 0
            
            DispatchQueue.main.async {
                self.captureManager.statusMessage = "Scanning room..."
            }
        }
        
        func captureSession(_ session: RoomCaptureSession, didEndWith data: CapturedRoomData?, error: Error?) {
            print("🏁 [Coordinator] Session ended")
            print("   - Has data: \(data != nil)")
            print("   - Has error: \(error != nil)")
            print("   - Updates received: \(updateCount)")
            print("   - Last room saved: \(lastCapturedRoom != nil)")
            
            if let error = error {
                print("❌ [Coordinator] Session ended with error: \(error)")
                onError?("Session error: \(error.localizedDescription)")
            }
            
            // WORKAROUND: If didPresent wasn't called, use the saved room
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
                guard let self = self else { return }
                
                if !self.didReceiveFinalData {
                    print("⚠️ [Coordinator] didPresent was never called - using workaround")
                    
                    if let savedRoom = self.lastCapturedRoom {
                        print("🔧 [Coordinator] Using last saved room: W:\(savedRoom.walls.count) O:\(savedRoom.objects.count)")
                        
                        Task { @MainActor in
                            await self.captureManager.processCapturedRoom(savedRoom)
                            print("✅ [Coordinator] Processed saved room data")
                        }
                    } else {
                        print("❌ [Coordinator] No saved room data available")
                        self.onError?("No room data was captured. Please try scanning again.")
                    }
                } else {
                    print("✅ [Coordinator] didPresent was called, no workaround needed")
                }
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
