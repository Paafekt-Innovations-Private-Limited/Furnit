import SwiftUI

struct HomeView: View {
    @StateObject private var modelManager = USDZModelManager()
    
    var body: some View {
        NavigationStack {
            VStack {
                if modelManager.models.isEmpty {
                    ContentUnavailableView(
                        "No 3D Models Found",
                        systemImage: "cube.transparent",
                        description: Text("Add USDZ files to your Assets catalog")
                    )
                } else {
                    List(modelManager.models) { model in
                        NavigationLink(destination: ModelViewerView(model: model)) {
                            ModelRowView(model: model)
                        }
                    }
                    .listStyle(PlainListStyle())
                }
            }
            .navigationTitle("3D Room Models")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

struct ModelRowView: View {
    let model: USDZModel
    
    var body: some View {
        HStack {
            Image(systemName: "cube.fill")
                .foregroundColor(.blue)
                .font(.title2)
                .frame(width: 40, height: 40)
                .background(Color.blue.opacity(0.1))
                .cornerRadius(8)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(model.displayName)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Text("3D Room Model")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Image(systemName: "chevron.right")
                .foregroundColor(.secondary)
                .font(.caption)
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    HomeView()
}