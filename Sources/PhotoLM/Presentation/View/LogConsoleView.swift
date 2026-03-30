import SwiftUI

struct LogConsoleView: View {
    @Binding var text: String

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    Text(text.isEmpty ? "Логи появятся здесь." : text)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(.white)
                        .textSelection(.enabled)
                    Color.clear
                        .frame(height: 1)
                        .id("log-end")
                }
                .padding(10)
            }
            .background(Color.black.opacity(0.9))
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .onChange(of: text) { _, _ in
                withAnimation(.easeOut(duration: 0.08)) {
                    proxy.scrollTo("log-end", anchor: .bottom)
                }
            }
        }
    }
}
