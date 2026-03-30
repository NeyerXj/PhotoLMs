import AppKit
import Foundation

protocol FolderDialogProviding {
    @MainActor
    func chooseDirectory(initialPath: String) -> String?
}

final class FolderDialogService: FolderDialogProviding {
    @MainActor
    func chooseDirectory(initialPath: String) -> String? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.prompt = "Выбрать"

        let expanded = (initialPath as NSString).expandingTildeInPath
        if !expanded.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: expanded, isDirectory: true)
        }

        let response = panel.runModal()
        return response == .OK ? panel.url?.path : nil
    }
}
