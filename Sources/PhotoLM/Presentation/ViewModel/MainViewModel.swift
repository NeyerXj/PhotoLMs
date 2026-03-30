import Foundation

@MainActor
final class MainViewModel: ObservableObject {
    @Published var projectDirectoryPath: String
    @Published var pythonExecutablePath: String
    @Published var inputDirectoryPath: String
    @Published var outputDirectoryPath: String
    @Published var maskDirectoryPath: String

    @Published var mode: ProcessingMode = .ui
    @Published var device: DeviceOption = .auto
    @Published var enhancement: EnhancementMode = .none
    @Published var maxSide: Int = 1600

    @Published var saveMasks: Bool = true
    @Published var tightTextMask: Bool = false
    @Published var poissonBlend: Bool = false
    @Published var offline: Bool = false
    @Published var overwriteOutputFiles: Bool = false

    @Published var logs: String = ""
    @Published var statusText: String = "Готово"
    @Published var state: ExecutionState = .idle
    @Published private(set) var isRunning: Bool = false

    private let pathProvider: DefaultPathProviding
    private let commandBuilder: ScriptCommandBuilding
    private let processRunner: ProcessRunning
    private let folderDialog: FolderDialogProviding
    private let fileDialog: FileDialogProviding
    private let finderService: FinderOpening
    private let fileManager = FileManager.default

    init(
        pathProvider: DefaultPathProviding = DefaultPathProvider(),
        commandBuilder: ScriptCommandBuilding = ScriptCommandBuilder(validator: ConfigurationValidator()),
        processRunner: ProcessRunning = SystemProcessRunner(),
        folderDialog: FolderDialogProviding = FolderDialogService(),
        fileDialog: FileDialogProviding = FileDialogService(),
        finderService: FinderOpening = FinderService()
    ) {
        self.pathProvider = pathProvider
        self.commandBuilder = commandBuilder
        self.processRunner = processRunner
        self.folderDialog = folderDialog
        self.fileDialog = fileDialog
        self.finderService = finderService

        let projectDirectory = pathProvider.defaultProjectDirectory()
        self.projectDirectoryPath = projectDirectory.path
        self.pythonExecutablePath = pathProvider.defaultPythonExecutable(for: projectDirectory).path
        self.inputDirectoryPath = pathProvider.defaultInputDirectory(for: projectDirectory).path
        self.outputDirectoryPath = pathProvider.defaultOutputDirectory(for: projectDirectory).path
        self.maskDirectoryPath = pathProvider.defaultMaskDirectory(for: projectDirectory).path
        ensureDefaultDirectories()
    }

    func chooseProjectDirectory() {
        if let selectedPath = folderDialog.chooseDirectory(initialPath: projectDirectoryPath) {
            projectDirectoryPath = selectedPath
        }
    }

    func choosePythonExecutable() {
        if let selectedPath = fileDialog.chooseFile(initialPath: pythonExecutablePath) {
            pythonExecutablePath = selectedPath
        }
    }

    func chooseInputDirectory() {
        if let selectedPath = folderDialog.chooseDirectory(initialPath: inputDirectoryPath) {
            inputDirectoryPath = selectedPath
        }
    }

    func chooseOutputDirectory() {
        if let selectedPath = folderDialog.chooseDirectory(initialPath: outputDirectoryPath) {
            outputDirectoryPath = selectedPath
        }
    }

    func chooseMaskDirectory() {
        if let selectedPath = folderDialog.chooseDirectory(initialPath: maskDirectoryPath) {
            maskDirectoryPath = selectedPath
        }
    }

    func applyProjectDefaults() {
        let projectURL = normalizedDirectoryURL(from: projectDirectoryPath)
        inputDirectoryPath = pathProvider.defaultInputDirectory(for: projectURL).path
        outputDirectoryPath = pathProvider.defaultOutputDirectory(for: projectURL).path
        maskDirectoryPath = pathProvider.defaultMaskDirectory(for: projectURL).path
        pythonExecutablePath = pathProvider.defaultPythonExecutable(for: projectURL).path
        ensureDefaultDirectories()
    }

    func runRemoval() {
        runRemoval(openViewerAfter: false)
    }

    func runViewerOnly() {
        guard !isRunning else {
            statusText = "Дождитесь завершения текущего процесса"
            return
        }

        do {
            let configuration = makeConfiguration()
            let copyStats = try syncInputToOutput(
                from: configuration.inputDirectory,
                to: configuration.outputDirectory,
                overwrite: overwriteOutputFiles
            )
            appendLog("\nПодготовка Viewer: input images=\(copyStats.totalImages), copied=\(copyStats.copied), skipped=\(copyStats.skipped)\n")
            guard hasViewableImages(in: configuration.outputDirectory) else {
                let message = "В выбранной папке output нет изображений для Viewer"
                state = .failure
                statusText = message
                appendLog("\nОшибка: \(message)\n")
                return
            }
            let command = try commandBuilder.makeViewerCommand(configuration: configuration)
            start(command: command, actionName: "Viewer")
        } catch {
            handle(error)
        }
    }

    func runRemovalWithViewer() {
        runRemoval(openViewerAfter: true)
    }

    func stopProcess() {
        guard isRunning else {
            return
        }
        processRunner.stop()
        statusText = "Остановка запрошена"
        appendLog("\nОстановка процесса...\n")
    }

    func openOutputInFinder() {
        finderService.open(path: outputDirectoryPath)
    }

    func clearLogs() {
        logs = ""
    }

    private func runRemoval(openViewerAfter: Bool) {
        guard !isRunning else {
            statusText = "Дождитесь завершения текущего процесса"
            return
        }

        do {
            let configuration = makeConfiguration()
            let command = try commandBuilder.makeRemovalCommand(configuration: configuration, openViewer: openViewerAfter)
            let action = openViewerAfter ? "Удаление + Viewer" : "Удаление"
            start(command: command, actionName: action)
        } catch {
            handle(error)
        }
    }

    private func makeConfiguration() -> RunConfiguration {
        let projectURL = normalizedDirectoryURL(from: projectDirectoryPath)
        let pythonURL = normalizedFileURL(from: pythonExecutablePath)
        let inputURL = normalizedDirectoryURL(from: inputDirectoryPath)
        let outputURL = normalizedDirectoryURL(from: outputDirectoryPath)
        let maskURL = saveMasks ? normalizedDirectoryURL(from: maskDirectoryPath) : nil
        ensureDirectory(projectURL)
        ensureDirectory(outputURL)
        if saveMasks, let maskURL {
            ensureDirectory(maskURL)
        }

        return RunConfiguration(
            projectDirectory: projectURL,
            pythonExecutable: pythonURL,
            inputDirectory: inputURL,
            outputDirectory: outputURL,
            maskDirectory: maskURL,
            mode: mode,
            device: device,
            enhancement: enhancement,
            maxSide: maxSide,
            tightTextMask: tightTextMask,
            poissonBlend: poissonBlend,
            offline: offline
        )
    }

    private func start(command: ScriptCommand, actionName: String) {
        state = .running
        isRunning = true
        statusText = "\(actionName) запущено"

        appendLog("\n[\(timestamp())] \(actionName)\n")
        appendLog("$ \(command.displayLine)\n")

        do {
            try processRunner.run(
                command: command,
                onOutput: { [weak self] text in
                    Task { @MainActor in
                        self?.appendLog(text)
                    }
                },
                onTermination: { [weak self] statusCode in
                    Task { @MainActor in
                        self?.finish(statusCode: statusCode)
                    }
                }
            )
        } catch {
            isRunning = false
            state = .failure
            statusText = error.localizedDescription
            appendLog("\nОшибка запуска: \(error.localizedDescription)\n")
        }
    }

    private func finish(statusCode: Int32) {
        isRunning = false
        if statusCode == 0 {
            state = .success
            statusText = "Завершено успешно"
            appendLog("\n[\(timestamp())] Готово\n")
        } else {
            state = .failure
            statusText = "Процесс завершился с кодом \(statusCode)"
            appendLog("\n[\(timestamp())] Завершено с кодом \(statusCode)\n")
        }
    }

    private func handle(_ error: Error) {
        state = .failure
        statusText = error.localizedDescription
        appendLog("\nОшибка: \(error.localizedDescription)\n")
    }

    private func appendLog(_ text: String) {
        logs += text
        if logs.count > 240_000 {
            logs.removeFirst(logs.count - 160_000)
        }
    }

    private func normalizedDirectoryURL(from path: String) -> URL {
        let prepared = normalizedPath(path)
        return URL(fileURLWithPath: prepared, isDirectory: true)
    }

    private func normalizedFileURL(from path: String) -> URL {
        let prepared = normalizedPath(path)
        return URL(fileURLWithPath: prepared, isDirectory: false)
    }

    private func normalizedPath(_ raw: String) -> String {
        let expanded = (raw as NSString).expandingTildeInPath
        return expanded.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func timestamp() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return formatter.string(from: Date())
    }

    private func ensureDefaultDirectories() {
        ensureDirectory(normalizedDirectoryURL(from: projectDirectoryPath))
        ensureDirectory(normalizedDirectoryURL(from: inputDirectoryPath))
        ensureDirectory(normalizedDirectoryURL(from: outputDirectoryPath))
        if saveMasks {
            ensureDirectory(normalizedDirectoryURL(from: maskDirectoryPath))
        }
    }

    private func ensureDirectory(_ url: URL) {
        try? fileManager.createDirectory(at: url, withIntermediateDirectories: true)
    }

    private func syncInputToOutput(from inputDirectory: URL, to outputDirectory: URL, overwrite: Bool) throws -> (totalImages: Int, copied: Int, skipped: Int) {
        var isDirectory = ObjCBool(false)
        guard fileManager.fileExists(atPath: inputDirectory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw ConfigurationValidationError.directoryNotFound(inputDirectory.path)
        }

        ensureDirectory(outputDirectory)
        let inputRoot = inputDirectory.standardizedFileURL.path
        let rootPrefix = inputRoot.hasSuffix("/") ? inputRoot : inputRoot + "/"
        guard let enumerator = fileManager.enumerator(
            at: inputDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return (0, 0, 0)
        }

        var totalImages = 0
        var copied = 0
        var skipped = 0

        for case let sourceURL as URL in enumerator {
            if !isSupportedImageFile(sourceURL) {
                continue
            }
            totalImages += 1
            let sourcePath = sourceURL.standardizedFileURL.path
            guard sourcePath.hasPrefix(rootPrefix) else {
                continue
            }
            let relativePath = String(sourcePath.dropFirst(rootPrefix.count))
            let destinationURL = outputDirectory.appendingPathComponent(relativePath)
            ensureDirectory(destinationURL.deletingLastPathComponent())

            if fileManager.fileExists(atPath: destinationURL.path) {
                if overwrite {
                    try fileManager.removeItem(at: destinationURL)
                } else {
                    skipped += 1
                    continue
                }
            }
            try fileManager.copyItem(at: sourceURL, to: destinationURL)
            copied += 1
        }

        if totalImages == 0 {
            throw ConfigurationValidationError.invalidNumber("В input нет изображений (png/jpg/webp/...)")
        }

        return (totalImages, copied, skipped)
    }

    private func hasViewableImages(in directory: URL) -> Bool {
        guard let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }

        for case let itemURL as URL in enumerator {
            if !isSupportedImageFile(itemURL) {
                continue
            }
            let lowercasePath = itemURL.path.lowercased()
            if lowercasePath.contains("/masks/") || lowercasePath.contains("/_backup_originals/") || lowercasePath.contains("/_backups_originals/") {
                continue
            }
            if itemURL.deletingPathExtension().lastPathComponent.lowercased().hasSuffix("_mask") {
                continue
            }
            return true
        }
        return false
    }

    private func isSupportedImageFile(_ fileURL: URL) -> Bool {
        let supportedExtensions = Set(["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])
        return supportedExtensions.contains(fileURL.pathExtension.lowercased())
    }
}
