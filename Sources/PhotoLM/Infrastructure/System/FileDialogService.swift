import AppKit
import Foundation

protocol FileDialogProviding {
    @MainActor
    func chooseFile(initialPath: String) -> String?
}

final class FileDialogService: FileDialogProviding {
    @MainActor
    func chooseFile(initialPath: String) -> String? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Выбрать"

        let expanded = (initialPath as NSString).expandingTildeInPath
        if !expanded.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: expanded, isDirectory: false).deletingLastPathComponent()
        }

        let response = panel.runModal()
        return response == .OK ? panel.url?.path : nil
    }
}
