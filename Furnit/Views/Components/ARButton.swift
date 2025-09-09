import SwiftUI

struct ARButton: View {
    @Binding var isARActive: Bool
    let isProcessing: Bool
    let onToggle: () -> Void
    
    @State private var isPressed = false
    @State private var pulseScale: CGFloat = 1.0
    @State private var processingRotation: Double = 0
    
    // Default initializer for backward compatibility
    init(isARActive: Binding<Bool>, onToggle: @escaping () -> Void) {
        self._isARActive = isARActive
        self.isProcessing = false
        self.onToggle = onToggle
    }
    
    // Full initializer with processing state
    init(isARActive: Binding<Bool>, isProcessing: Bool = false, onToggle: @escaping () -> Void) {
        self._isARActive = isARActive
        self.isProcessing = isProcessing
        self.onToggle = onToggle
    }
    
    var body: some View {
        Button(action: {
            onToggle()
            
            // Haptic feedback
            let impact = UIImpactFeedbackGenerator(style: .medium)
            impact.impactOccurred()
            
        }) {
            ZStack {
                // Background circle with glow effect
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                backgroundGradientColors.0,
                                backgroundGradientColors.1
                            ],
                            center: .center,
                            startRadius: 10,
                            endRadius: 30
                        )
                    )
                    .frame(width: 60, height: 60)
                    .scaleEffect(isPressed ? 0.9 : 1.0)
                    .scaleEffect(pulseScale)
                
                // Inner circle with glass effect
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [
                                Color.white.opacity(0.3),
                                Color.white.opacity(0.1)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 50, height: 50)
                    .scaleEffect(isPressed ? 0.9 : 1.0)
                
                // Icon with processing state
                if isProcessing {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .scaleEffect(1.2)
                        .rotationEffect(.degrees(processingRotation))
                } else {
                    Image(systemName: buttonIcon)
                        .font(.system(size: 20, weight: .semibold))
                        .foregroundColor(.white)
                        .scaleEffect(isPressed ? 0.85 : 1.0)
                }
            }
        }
        .buttonStyle(PlainButtonStyle()) // Remove default button styling
        .disabled(isProcessing) // Disable button during processing
        .onLongPressGesture(minimumDuration: 0, maximumDistance: .infinity, pressing: { pressing in
            withAnimation(.easeInOut(duration: 0.1)) {
                isPressed = pressing && !isProcessing // Don't show press effect when processing
            }
        }, perform: {})
        .onChange(of: isARActive) { _, newValue in
            // Animate when AR state changes
            withAnimation(.easeInOut(duration: 0.3)) {
                pulseScale = newValue ? 1.1 : 1.0
            }
            
            // Start pulsing animation when AR is active
            if newValue && !isProcessing {
                startPulseAnimation()
            } else {
                stopPulseAnimation()
            }
        }
        .onChange(of: isProcessing) { _, newValue in
            if newValue {
                startProcessingAnimation()
                stopPulseAnimation()
            } else {
                stopProcessingAnimation()
                if isARActive {
                    startPulseAnimation()
                }
            }
        }
        .shadow(
            color: shadowColor.opacity(0.4),
            radius: isARActive ? 8 : 4,
            x: 0,
            y: 2
        )
    }
    
    private func startPulseAnimation() {
        withAnimation(
            Animation.easeInOut(duration: 1.0)
                .repeatForever(autoreverses: true)
        ) {
            pulseScale = 1.2
        }
    }
    
    private func stopPulseAnimation() {
        withAnimation(.easeInOut(duration: 0.3)) {
            pulseScale = 1.0
        }
    }
    
    private func startProcessingAnimation() {
        withAnimation(
            Animation.linear(duration: 1.0)
                .repeatForever(autoreverses: false)
        ) {
            processingRotation = 360
        }
    }
    
    private func stopProcessingAnimation() {
        withAnimation(.easeInOut(duration: 0.3)) {
            processingRotation = 0
        }
    }
    
    // MARK: - Computed Properties
    
    private var buttonIcon: String {
        if isProcessing {
            return "hourglass"
        } else if isARActive {
            return "stop.fill"
        } else {
            return "camera.viewfinder"
        }
    }
    
    private var backgroundGradientColors: (Color, Color) {
        if isProcessing {
            return (Color.orange.opacity(0.8), Color.orange.opacity(0.6))
        } else if isARActive {
            return (Color.red.opacity(0.8), Color.red.opacity(0.6))
        } else {
            return (Color.blue.opacity(0.8), Color.blue.opacity(0.6))
        }
    }
    
    private var shadowColor: Color {
        if isProcessing {
            return .orange
        } else if isARActive {
            return .red
        } else {
            return .blue
        }
    }
}

// MARK: - Preview
#Preview {
    ZStack {
        Color.black.opacity(0.8)
            .ignoresSafeArea()
        
        VStack(spacing: 40) {
            ARButton(isARActive: .constant(false)) {
                print("AR Button tapped - inactive")
            }
            
            ARButton(isARActive: .constant(true)) {
                print("AR Button tapped - active")
            }
        }
    }
}