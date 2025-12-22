import SwiftUI

struct ARButton: View {
    @Binding var isARActive: Bool
    let onToggle: () -> Void
    
    var body: some View {
        Button(action: {
            onToggle()
        }) {
            Image(systemName: isARActive ? "stop.fill" : "camera.viewfinder")
                .font(.title2)
                .foregroundColor(.white)
                .frame(width: 44, height: 44)
                .background(isARActive ? Color.red : Color.blue)
                .clipShape(Circle())
                .shadow(color: .black.opacity(0.3), radius: 4, x: 0, y: 2)
        }
        .animation(.easeInOut(duration: 0.2), value: isARActive)
    }
}

#Preview {
    VStack {
        ARButton(isARActive: .constant(false), onToggle: {})
        ARButton(isARActive: .constant(true), onToggle: {})
    }
    .padding()
    .background(Color.gray)
}