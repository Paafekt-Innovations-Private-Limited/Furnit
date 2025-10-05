import SwiftUI
import ARKit
import RealityKit

// MARK: - Simplified Scanning Methods
enum SimpleScanMethod: String, CaseIterable {
    case fourPhotos = "4-Photo Room Scan"
    case arQuickScan = "AR Quick Scan"
    case multiPhoto = "20-Photo Detailed"
    case hybrid25D = "2.5D Dollhouse"
    
    var description: String {
        switch self {
        case .fourPhotos:
            return "Take 4 photos from room corners"
        case .arQuickScan:
            return "Quick AR capture of room geometry"
        case .multiPhoto:
            return "20 photos for detailed 3D model"
        case .hybrid25D:
            return "6 photos for dollhouse view"
        }
    }
    
    var photoCount: Int {
        switch self {
        case .fourPhotos: return 4
        case .arQuickScan: return 0
        case .multiPhoto: return 20
        case .hybrid25D: return 6
        }
    }
}

// MARK: - Simple Multi-Method Scanner
struct SimpleMultiMethodScanner: View {
    @Binding var isPresented: Bool
    @ObservedObject var modelManager: USDZModelManager
    @State private var selectedMethod: SimpleScanMethod = .fourPhotos
    @State private var showMethodPicker = true
    @State private var showScanner = false
    @State private var capturedPhotos: [UIImage] = []
    @State private var isProcessing = false
    @State private var scanComplete = false
    
    var body: some View {
        NavigationView {
            ZStack {
                // Background
                LinearGradient(
                    gradient: Gradient(colors: [Color.black, Color.blue.opacity(0.3)]),
                    startPoint: .top,
                    endPoint: .bottom
                )
                .ignoresSafeArea()
                
                if showMethodPicker {
                    methodPickerView
                } else if showScanner {
                    scannerView
                } else if isProcessing {
                    processingView
                } else if scanComplete {
                    successView
                }
            }
            .navigationTitle("3D Room Scanner")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") {
                        isPresented = false
                    }
                    .foregroundColor(.white)
                }
            }
            .preferredColorScheme(.dark)
        }
    }
    
    // MARK: - Method Picker View
    private var methodPickerView: some View {
        VStack(spacing: 20) {
            Text("Choose Scanning Method")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .padding(.top, 40)
            
            VStack(spacing: 16) {
                ForEach(SimpleScanMethod.allCases, id: \.self) { method in
                    Button(action: {
                        selectedMethod = method
                    }) {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(method.rawValue)
                                    .font(.headline)
                                    .foregroundColor(.white)
                                
                                Text(method.description)
                                    .font(.caption)
                                    .foregroundColor(.white.opacity(0.7))
                            }
                            
                            Spacer()
                            
                            if selectedMethod == method {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                    .font(.title2)
                            }
                        }
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(selectedMethod == method ?
                                     Color.blue.opacity(0.3) :
                                     Color.white.opacity(0.1))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(selectedMethod == method ?
                                       Color.blue :
                                       Color.white.opacity(0.2), lineWidth: 2)
                        )
                    }
                }
            }
            .padding(.horizontal)
            
            Spacer()
            
            Button(action: {
                showMethodPicker = false
                showScanner = true
                if selectedMethod == .arQuickScan {
                    // For AR scan, go straight to processing
                    performARScan()
                }
            }) {
                HStack {
                    Image(systemName: "camera.fill")
                    Text("Start Scanning")
                }
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(
                    LinearGradient(
                        gradient: Gradient(colors: [.blue, .purple]),
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .cornerRadius(12)
            }
            .padding(.horizontal)
            .padding(.bottom, 30)
        }
    }
    
    // MARK: - Scanner View
    private var scannerView: some View {
        Group {
            if selectedMethod == .arQuickScan {
                // AR scanning view
                ARQuickScanView { success in
                    if success {
                        completeScanning()
                    }
                }
            } else {
                // Photo capture view
                SimplePhotoCaptureView(
                    requiredPhotos: selectedMethod.photoCount,
                    capturedPhotos: $capturedPhotos,
                    currentPhotoNumber: capturedPhotos.count + 1,
                    onPhotoTaken: { photo in
                        capturedPhotos.append(photo)
                        if capturedPhotos.count >= selectedMethod.photoCount {
                            showScanner = false
                            processPhotos()
                        }
                    },
                    onCancel: {
                        showScanner = false
                        showMethodPicker = true
                        capturedPhotos = []
                    }
                )
            }
        }
    }
    
    // MARK: - Processing View
    private var processingView: some View {
        VStack(spacing: 20) {
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                .scaleEffect(2)
            
            Text("Processing...")
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            Text("Creating 3D model from \(capturedPhotos.count) photos")
                .font(.body)
                .foregroundColor(.white.opacity(0.8))
        }
        .padding(40)
        .background(Color.black.opacity(0.8))
        .cornerRadius(20)
    }
    
    // MARK: - Success View
    private var successView: some View {
        VStack(spacing: 30) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 80))
                .foregroundColor(.green)
            
            Text("Scan Complete!")
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            Text("Room added to your collection")
                .font(.body)
                .foregroundColor(.white.opacity(0.8))
            
            HStack(spacing: 20) {
                Button(action: {
                    // Reset for another scan
                    resetScanner()
                }) {
                    Text("Scan Another")
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .frame(width: 150, height: 50)
                        .background(Color.blue)
                        .cornerRadius(25)
                }
                
                Button(action: {
                    isPresented = false
                }) {
                    Text("Done")
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .frame(width: 150, height: 50)
                        .background(Color.green)
                        .cornerRadius(25)
                }
            }
        }
        .padding()
    }
    
    // MARK: - Helper Methods
    private func performARScan() {
        showScanner = false
        isProcessing = true
        
        // Simulate AR scanning
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            completeScanning()
        }
    }
    
    private func processPhotos() {
        isProcessing = true
        
        // Simulate processing
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            completeScanning()
        }
    }
    
    private func completeScanning() {
        // Create new model
        let roomName = "\(selectedMethod.rawValue) - \(Date().formatted(date: .abbreviated, time: .shortened))"
        let newModel = USDZModel(
            name: roomName,
            fileName: modelManager.models.first?.fileName ?? "room.usdz"
        )
        
        // Add to collection
        modelManager.models.append(newModel)
        
        isProcessing = false
        scanComplete = true
    }
    
    private func resetScanner() {
        capturedPhotos = []
        scanComplete = false
        showMethodPicker = true
    }
}

// MARK: - Simple Photo Capture View
struct SimplePhotoCaptureView: View {
    let requiredPhotos: Int
    @Binding var capturedPhotos: [UIImage]
    let currentPhotoNumber: Int
    let onPhotoTaken: (UIImage) -> Void
    let onCancel: () -> Void
    
    @State private var showImagePicker = true
    @State private var currentImage: UIImage?
    
    var body: some View {
        VStack {
            // Header
            HStack {
                Text("Photo \(currentPhotoNumber) of \(requiredPhotos)")
                    .font(.headline)
                    .foregroundColor(.white)
                
                Spacer()
                
                // Progress dots
                HStack(spacing: 8) {
                    ForEach(1...requiredPhotos, id: \.self) { index in
                        Circle()
                            .fill(index <= capturedPhotos.count ? Color.green : Color.white.opacity(0.3))
                            .frame(width: 10, height: 10)
                    }
                }
            }
            .padding()
            .background(Color.black.opacity(0.7))
            
            Spacer()
            
            // Instructions
            VStack(spacing: 12) {
                Image(systemName: "camera.viewfinder")
                    .font(.system(size: 80))
                    .foregroundColor(.white.opacity(0.8))
                
                Text(getInstructionForPhoto(currentPhotoNumber))
                    .font(.title3)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
            
            Spacer()
            
            // Preview strip
            if !capturedPhotos.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(capturedPhotos.enumerated()), id: \.offset) { _, photo in
                            Image(uiImage: photo)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: 60, height: 60)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                    .padding(.horizontal)
                }
                .frame(height: 80)
                .background(Color.black.opacity(0.5))
            }
        }
        .sheet(isPresented: $showImagePicker, onDismiss: {
            if let image = currentImage {
                onPhotoTaken(image)
                currentImage = nil
                // Re-show picker for next photo
                if currentPhotoNumber < requiredPhotos {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        showImagePicker = true
                    }
                }
            } else if capturedPhotos.isEmpty {
                // User cancelled on first photo
                onCancel()
            }
        }) {
            ImagePicker(image: $currentImage, sourceType: .camera)
        }
    }
    
    private func getInstructionForPhoto(_ number: Int) -> String {
        if requiredPhotos == 4 {
            switch number {
            case 1: return "Stand in first corner\nCapture opposite wall"
            case 2: return "Move to next corner\nCapture adjacent wall"
            case 3: return "Third corner\nCapture another view"
            case 4: return "Final corner\nCapture last wall"
            default: return "Capture the room"
            }
        } else if requiredPhotos == 6 {
            switch number {
            case 1...4: return "Capture wall \(number)"
            case 5: return "Point camera at floor"
            case 6: return "Point camera at ceiling"
            default: return "Capture the room"
            }
        } else {
            let angle = (360.0 / Double(requiredPhotos)) * Double(number - 1)
            return "Rotate to \(Int(angle))° and capture"
        }
    }
}

// MARK: - AR Quick Scan View
struct ARQuickScanView: View {
    let onComplete: (Bool) -> Void
    @State private var scanProgress: Float = 0.0
    
    var body: some View {
        ZStack {
            // AR visualization placeholder
            Color.black.opacity(0.9)
                .ignoresSafeArea()
            
            VStack {
                Text("AR Quick Scan")
                    .font(.title)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .padding(.top, 50)
                
                Spacer()
                
                // AR visualization
                Image(systemName: "cube.transparent")
                    .font(.system(size: 150))
                    .foregroundColor(.white.opacity(0.5))
                    .rotationEffect(.degrees(Double(scanProgress * 360)))
                    .animation(.linear(duration: 2).repeatForever(autoreverses: false), value: scanProgress)
                
                Text("Move device slowly around room")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                
                Spacer()
                
                ProgressView(value: scanProgress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .white))
                    .frame(width: 250)
                    .padding(.bottom, 100)
            }
        }
        .onAppear {
            startARScan()
        }
    }
    
    private func startARScan() {
        // Simulate AR scanning progress
        withAnimation(.linear(duration: 3)) {
            scanProgress = 1.0
        }
        
        // Complete after 3 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            onComplete(true)
        }
    }
}
