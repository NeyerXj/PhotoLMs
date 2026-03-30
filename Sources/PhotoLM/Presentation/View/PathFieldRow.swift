import SwiftUI

struct PathFieldRow: View {
    let title: String
    @Binding var value: String
    let browseTitle: String
    let onBrowse: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Text(title)
                .frame(width: 170, alignment: .leading)
            TextField("", text: $value)
                .textFieldStyle(.roundedBorder)
                .font(.system(.body, design: .monospaced))
            Button(browseTitle, action: onBrowse)
                .buttonStyle(.bordered)
        }
    }
}
