import SwiftUI

struct VirtualJoystick: View {
    // Joystick state and configuration
    @Binding var joystickOffset: CGSize
    @State private var knobPosition: CGSize = .zero
    @State private var isDragging: Bool = false
    
    // Joystick appearance properties
    let outerCircleRadius: CGFloat = 50
    let innerKnobRadius: CGFloat = 20
    let maxDistance: CGFloat = 30 // Maximum distance knob can move from center
    
    var body: some View {
        ZStack {
            // Outer joystick background circle
            Circle()
                .fill(Color.black.opacity(0.3))
                .frame(width: outerCircleRadius * 2, height: outerCircleRadius * 2)
                .overlay(
                    Circle()
                        .stroke(Color.white.opacity(0.5), lineWidth: 2)
                )
            
            // Inner draggable knob
            Circle()
                .fill(Color.white.opacity(0.8))
                .frame(width: innerKnobRadius * 2, height: innerKnobRadius * 2)
                .offset(knobPosition)
                .scaleEffect(isDragging ? 1.2 : 1.0) // Visual feedback when dragging
                .animation(.easeInOut(duration: 0.1), value: isDragging)
        }
        .gesture(
            DragGesture()
                .onChanged { dragValue in
                    // Update dragging state for visual feedback
                    if !isDragging {
                        isDragging = true
                    }
                    
                    // Calculate new knob position within circular bounds
                    let dragDistance = sqrt(dragValue.translation.width * dragValue.translation.width + 
                                          dragValue.translation.height * dragValue.translation.height)
                    
                    if dragDistance <= maxDistance {
                        // Within bounds - use actual drag position
                        knobPosition = dragValue.translation
                        joystickOffset = dragValue.translation
                    } else {
                        // Outside bounds - constrain to circle edge
                        let constrainingFactor = maxDistance / dragDistance
                        let constrainedX = dragValue.translation.width * constrainingFactor
                        let constrainedY = dragValue.translation.height * constrainingFactor
                        
                        knobPosition = CGSize(width: constrainedX, height: constrainedY)
                        joystickOffset = knobPosition
                    }
                }
                .onEnded { _ in
                    // Reset knob to center when drag ends
                    isDragging = false
                    withAnimation(.easeOut(duration: 0.3)) {
                        knobPosition = .zero
                        joystickOffset = .zero
                    }
                }
        )
    }
}

#Preview {
    VirtualJoystick(joystickOffset: .constant(.zero))
        .frame(width: 200, height: 200)
        .background(Color.gray.opacity(0.3))
}
