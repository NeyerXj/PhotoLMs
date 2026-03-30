import SwiftUI

struct MainWindowView: View {
    @StateObject private var viewModel = MainViewModel()

    var body: some View {
        VStack(spacing: 14) {
            header
            Form {
                pathsSection
                viewerSection
                logsSection
            }
            .formStyle(.grouped)
        }
        .padding(16)
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text("PhotoLM")
                    .font(.system(size: 29, weight: .bold, design: .rounded))
                Text("Viewer редактор с выбором input/output")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Label(viewModel.statusText, systemImage: viewModel.state.symbolName)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(viewModel.state.color.opacity(0.16))
                .foregroundStyle(viewModel.state.color)
                .clipShape(Capsule())
            if viewModel.isRunning {
                ProgressView()
                    .controlSize(.small)
            }
        }
    }

    private var pathsSection: some View {
        Section("Папки") {
            PathFieldRow(
                title: "Input directory",
                value: $viewModel.inputDirectoryPath,
                browseTitle: "Выбрать",
                onBrowse: viewModel.chooseInputDirectory
            )

            PathFieldRow(
                title: "Output directory",
                value: $viewModel.outputDirectoryPath,
                browseTitle: "Выбрать",
                onBrowse: viewModel.chooseOutputDirectory
            )

            HStack {
                Button("Открыть output", action: viewModel.openOutputInFinder)
                    .buttonStyle(.bordered)
                Button("Пути по умолчанию", action: viewModel.applyProjectDefaults)
                    .buttonStyle(.bordered)
            }
        }
        .disabled(viewModel.isRunning)
    }

    private var viewerSection: some View {
        Section("Viewer") {
            Toggle("Перезаписывать output файлами из input", isOn: $viewModel.overwriteOutputFiles)
                .disabled(viewModel.isRunning)
            HStack(spacing: 10) {
                Button("Открыть Viewer для редактирования", action: viewModel.runViewerOnly)
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.isRunning)
                Button("Стоп", action: viewModel.stopProcess)
                    .buttonStyle(.bordered)
                    .disabled(!viewModel.isRunning)
                Button("Очистить логи", action: viewModel.clearLogs)
                    .buttonStyle(.bordered)
            }
        }
    }

    private var logsSection: some View {
        Section("Логи") {
            LogConsoleView(text: $viewModel.logs)
                .frame(minHeight: 250)
                .listRowInsets(EdgeInsets())
        }
    }
}
