import SwiftUI

// Specialized AR status view component for enhanced user experience
// Provides detailed status updates, progress tracking, and visual feedback during AR operations
struct ARStatusView: View {
    let processingState: ARProcessingState
    let elapsedTime: String
    
    var body: some View {
        VStack(spacing: 12) {
            // Status icon and message
            statusHeader
            
            // Progress indicators
            if processingState.showsProgress {
                progressSection
            }
            
            // Placement instructions
            if processingState.isReadyToPlace {
                placementInstructions
            }
            
            // Additional contextual information
            if let secondaryMessage = processingState.secondaryMessage {
                secondaryText(secondaryMessage)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
        .background(backgroundStyle)
        .animation(.easeInOut(duration: 0.3), value: processingState)
    }
    
    // MARK: - Status Header
    
    private var statusHeader: some View {
        HStack(spacing: 12) {
            // Status icon with animation
            Image(systemName: processingState.iconName)
                .font(.title2)
                .foregroundColor(.white)
                .rotationEffect(.degrees(processingState.isProcessing ? 360 : 0))
                .animation(
                    processingState.isProcessing ? 
                        .linear(duration: 2.0).repeatForever(autoreverses: false) : .default,
                    value: processingState.isProcessing
                )
            
            VStack(alignment: .leading, spacing: 2) {
                // Primary status message
                Text(processingState.displayMessage)
                    .font(.headline)
                    .foregroundColor(.white)
                    .multilineTextAlignment(.leading)
                
                // Processing time indicator
                if processingState.isProcessing && !elapsedTime.isEmpty {
                    HStack(spacing: 4) {
                        Image(systemName: "clock")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.7))
                        
                        Text(elapsedTime)
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.7))
                    }
                }
            }
            
            Spacer()
        }
    }
    
    // MARK: - Progress Section
    
    private var progressSection: some View {
        VStack(spacing: 8) {
            // Linear progress bar
            ProgressView(value: processingState.progressValue)
                .progressViewStyle(CustomLinearProgressViewStyle())
                .frame(height: 6)
            
            // Processing details
            if processingState.isProcessing {
                HStack {
                    // Spinning indicator for active processing
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .scaleEffect(0.6)
                    
                    // Percentage complete
                    Text("\(Int(processingState.progressValue * 100))%")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.8))
                    
                    Spacer()
                    
                    // Current processing stage indicator
                    processingStageIndicator
                }
            }
        }
    }
    
    // MARK: - Processing Stage Indicator
    
    private var processingStageIndicator: some View {
        HStack(spacing: 2) {
            ForEach(0..<4, id: \.self) { index in
                Circle()
                    .frame(width: 6, height: 6)
                    .foregroundColor(
                        processingState.progressValue >= Double(index + 1) / 4.0 ? 
                            .white : .white.opacity(0.3)
                    )
            }
        }
    }
    
    // MARK: - Placement Instructions
    
    private var placementInstructions: some View {
        HStack(spacing: 8) {
            // Animated tap gesture icon
            Image(systemName: "hand.tap")
                .foregroundColor(.white)
                .font(.title3)
                .scaleEffect(1.0)
                .animation(
                    .easeInOut(duration: 0.8).repeatForever(autoreverses: true),
                    value: processingState.isReadyToPlace
                )
            
            VStack(alignment: .leading, spacing: 2) {
                Text("Tap anywhere to place")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.white)
                
                Text("Position your 3D furniture model")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.8))
            }
            
            Spacer()
        }
        .padding(.top, 8)
    }
    
    // MARK: - Secondary Text
    
    private func secondaryText(_ message: String) -> some View {
        Text(message)
            .font(.caption)
            .foregroundColor(.white.opacity(0.8))
            .multilineTextAlignment(.center)
            .padding(.horizontal, 8)
    }
    
    // MARK: - Background Style
    
    private var backgroundStyle: some View {
        RoundedRectangle(cornerRadius: 16)
            .fill(
                processingState.isError ? 
                    Color.red.opacity(0.9) : 
                    processingState.statusColor.opacity(0.85)
            )
            .shadow(color: .black.opacity(0.3), radius: 8, x: 0, y: 4)
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.2), lineWidth: 1)
            )
    }
}

// MARK: - Custom Progress View Style

struct CustomLinearProgressViewStyle: ProgressViewStyle {
    func makeBody(configuration: Configuration) -> some View {
        ZStack(alignment: .leading) {
            // Background track
            RoundedRectangle(cornerRadius: 3)
                .fill(Color.white.opacity(0.3))
                .frame(height: 6)
            
            // Progress fill
            RoundedRectangle(cornerRadius: 3)
                .fill(
                    LinearGradient(
                        colors: [Color.white.opacity(0.8), Color.white],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .frame(width: CGFloat(configuration.fractionCompleted ?? 0) * 120, height: 6)
                .animation(.easeInOut(duration: 0.5), value: configuration.fractionCompleted)
        }
        .frame(width: 120)
    }
}

// MARK: - AR Status View with Processing State Manager

struct ARStatusViewContainer: View {
    @ObservedObject var stateManager: ARProcessingStateManager
    
    var body: some View {
        VStack {
            Spacer()
            
            HStack {
                Spacer()
                
                ARStatusView(
                    processingState: stateManager.currentState,
                    elapsedTime: stateManager.formattedElapsedTime
                )
                
                Spacer()
            }
            
            Spacer().frame(height: 120) // Space above bottom controls
        }
    }
}

// MARK: - Preview

#Preview {
    VStack(spacing: 20) {
        ARStatusView(
            processingState: .pointing,
            elapsedTime: ""
        )
        
        ARStatusView(
            processingState: .processing(progress: 0.6),
            elapsedTime: "12s"
        )
        
        ARStatusView(
            processingState: .ready,
            elapsedTime: ""
        )
        
        ARStatusView(
            processingState: .error("Network connection failed"),
            elapsedTime: ""
        )
    }
    .padding()
    .background(Color.gray)
}