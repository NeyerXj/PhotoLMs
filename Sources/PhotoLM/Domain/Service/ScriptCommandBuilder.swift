import Foundation

enum CommandBuilderError: LocalizedError {
    case scriptMissing(String)

    var errorDescription: String? {
        switch self {
        case let .scriptMissing(path):
            return "Не найден скрипт: \(path)"
        }
    }
}

protocol ScriptCommandBuilding {
    func makeRemovalCommand(configuration: RunConfiguration, openViewer: Bool) throws -> ScriptCommand
    func makeViewerCommand(configuration: RunConfiguration) throws -> ScriptCommand
}

final class ScriptCommandBuilder: ScriptCommandBuilding {
    private let validator: ConfigurationValidating
    private let fileManager = FileManager.default

    init(validator: ConfigurationValidating) {
        self.validator = validator
    }

    func makeRemovalCommand(configuration: RunConfiguration, openViewer: Bool) throws -> ScriptCommand {
        try validator.validateForRemoval(configuration)
        let scriptURL = try scriptURL(name: "ui_remove.py", in: configuration.projectDirectory)

        var arguments: [String] = [
            scriptURL.path,
            "--input", configuration.inputDirectory.path,
            "--output", configuration.outputDirectory.path,
            "--mode", configuration.mode.rawValue,
            "--device", configuration.device.rawValue,
            "--enhance", configuration.enhancement.rawValue,
            "--max-side", String(configuration.maxSide)
        ]

        if let maskDirectory = configuration.maskDirectory {
            arguments.append(contentsOf: ["--mask-out", maskDirectory.path])
        }
        if configuration.tightTextMask {
            arguments.append("--tight-text-mask")
        }
        if configuration.poissonBlend {
            arguments.append("--poisson-blend")
        }
        if configuration.offline {
            arguments.append("--offline")
        }
        if openViewer {
            arguments.append("--open-viewer")
        }

        var environment = [
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1"
        ]
        environment.merge(embeddedPythonEnvironment(for: configuration.pythonExecutable)) { _, new in new }

        return ScriptCommand(
            executableURL: configuration.pythonExecutable,
            arguments: arguments,
            workingDirectoryURL: configuration.projectDirectory,
            environment: environment
        )
    }

    func makeViewerCommand(configuration: RunConfiguration) throws -> ScriptCommand {
        try validator.validateForViewer(configuration)
        let scriptURL = try scriptURL(name: "ui_viewer.py", in: configuration.projectDirectory)

        let arguments: [String] = [
            scriptURL.path,
            "--dir", configuration.outputDirectory.path,
            "--lama-device", "cpu",
            "--lama-roi-pad", "96",
            "--lama-max-side", "1280"
        ]

        var environment = [
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1"
        ]
        environment.merge(embeddedPythonEnvironment(for: configuration.pythonExecutable)) { _, new in new }

        return ScriptCommand(
            executableURL: configuration.pythonExecutable,
            arguments: arguments,
            workingDirectoryURL: configuration.projectDirectory,
            environment: environment
        )
    }

    private func scriptURL(name: String, in directory: URL) throws -> URL {
        if let bundledScriptURL = AppBundlePaths.bundledScriptURL(named: name) {
            return bundledScriptURL
        }
        let scriptURL = directory.appendingPathComponent(name)
        if !fileManager.fileExists(atPath: scriptURL.path) {
            throw CommandBuilderError.scriptMissing(scriptURL.path)
        }
        return scriptURL
    }

    private func embeddedPythonEnvironment(for executableURL: URL) -> [String: String] {
        guard
            let bundledPythonExecutable = AppBundlePaths.bundledPythonExecutableURL(),
            let bundledPythonHome = AppBundlePaths.bundledPythonHomeURL(),
            let bundledSitePackages = AppBundlePaths.bundledSitePackagesURL(for: RuntimeArchitecture.current),
            bundledPythonExecutable.standardizedFileURL == executableURL.standardizedFileURL
        else {
            return [:]
        }

        return [
            "PYTHONHOME": bundledPythonHome.path,
            "PYTHONPATH": bundledSitePackages.path,
            "PYTHONNOUSERSITE": "1"
        ]
    }
}
