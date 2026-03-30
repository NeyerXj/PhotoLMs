import Foundation

protocol DefaultPathProviding {
    func defaultProjectDirectory() -> URL
    func defaultPythonExecutable(for projectDirectory: URL) -> URL
    func defaultInputDirectory(for projectDirectory: URL) -> URL
    func defaultOutputDirectory(for projectDirectory: URL) -> URL
    func defaultMaskDirectory(for projectDirectory: URL) -> URL
}

final class DefaultPathProvider: DefaultPathProviding {
    private let fileManager = FileManager.default

    func defaultProjectDirectory() -> URL {
        let workspace = AppBundlePaths.defaultWorkspaceDirectory()
        try? fileManager.createDirectory(at: workspace, withIntermediateDirectories: true)
        if hasScripts(in: workspace) || AppBundlePaths.bundledScriptURL(named: "ui_remove.py") != nil {
            return workspace
        }

        let currentDirectory = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
        if hasScripts(in: currentDirectory) {
            return currentDirectory
        }

        let bundleDirectory = Bundle.main.bundleURL.deletingLastPathComponent()
        if hasScripts(in: bundleDirectory) {
            return bundleDirectory
        }

        return workspace
    }

    func defaultPythonExecutable(for projectDirectory: URL) -> URL {
        if let bundledPythonURL = AppBundlePaths.bundledPythonExecutableURL() {
            return bundledPythonURL
        }

        let candidates = [
            projectDirectory.appendingPathComponent(".venv/bin/python3"),
            projectDirectory.appendingPathComponent(".venv/bin/python"),
            URL(fileURLWithPath: "/opt/homebrew/bin/python3"),
            URL(fileURLWithPath: "/usr/local/bin/python3"),
            URL(fileURLWithPath: "/usr/bin/python3")
        ]

        for candidate in candidates where fileManager.fileExists(atPath: candidate.path) {
            return candidate
        }

        return candidates[candidates.count - 1]
    }

    func defaultInputDirectory(for projectDirectory: URL) -> URL {
        projectDirectory.appendingPathComponent("input", isDirectory: true)
    }

    func defaultOutputDirectory(for projectDirectory: URL) -> URL {
        projectDirectory.appendingPathComponent("output", isDirectory: true)
    }

    func defaultMaskDirectory(for projectDirectory: URL) -> URL {
        projectDirectory.appendingPathComponent("output/masks", isDirectory: true)
    }

    private func hasScripts(in directory: URL) -> Bool {
        let removeScript = directory.appendingPathComponent("ui_remove.py").path
        let viewerScript = directory.appendingPathComponent("ui_viewer.py").path
        return fileManager.fileExists(atPath: removeScript) && fileManager.fileExists(atPath: viewerScript)
    }
}
