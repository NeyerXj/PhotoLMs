import Foundation

enum ProcessRunnerError: LocalizedError {
    case alreadyRunning
    case failedToStart(String)

    var errorDescription: String? {
        switch self {
        case .alreadyRunning:
            return "Уже выполняется другой процесс"
        case let .failedToStart(reason):
            return "Не удалось запустить процесс: \(reason)"
        }
    }
}

protocol ProcessRunning: AnyObject {
    var isRunning: Bool { get }
    func run(
        command: ScriptCommand,
        onOutput: @escaping @Sendable (String) -> Void,
        onTermination: @escaping @Sendable (Int32) -> Void
    ) throws
    func stop()
}

final class SystemProcessRunner: ProcessRunning, @unchecked Sendable {
    private let stateQueue = DispatchQueue(label: "photolm.process.runner")
    private var process: Process?
    private var stdoutHandle: FileHandle?
    private var stderrHandle: FileHandle?

    var isRunning: Bool {
        stateQueue.sync {
            process?.isRunning == true
        }
    }

    func run(
        command: ScriptCommand,
        onOutput: @escaping @Sendable (String) -> Void,
        onTermination: @escaping @Sendable (Int32) -> Void
    ) throws {
        try stateQueue.sync {
            if process != nil {
                throw ProcessRunnerError.alreadyRunning
            }

            let task = Process()
            task.executableURL = command.executableURL
            task.arguments = command.arguments
            task.currentDirectoryURL = command.workingDirectoryURL

            var environment = ProcessInfo.processInfo.environment
            command.environment.forEach { key, value in
                environment[key] = value
            }
            task.environment = environment

            let stdoutPipe = Pipe()
            let stderrPipe = Pipe()
            task.standardOutput = stdoutPipe
            task.standardError = stderrPipe

            let outHandle = stdoutPipe.fileHandleForReading
            let errHandle = stderrPipe.fileHandleForReading

            outHandle.readabilityHandler = { handle in
                let data = handle.availableData
                if data.isEmpty {
                    return
                }
                onOutput(String(decoding: data, as: UTF8.self))
            }

            errHandle.readabilityHandler = { handle in
                let data = handle.availableData
                if data.isEmpty {
                    return
                }
                onOutput(String(decoding: data, as: UTF8.self))
            }

            task.terminationHandler = { [weak self] finished in
                self?.cleanup()
                onTermination(finished.terminationStatus)
            }

            do {
                try task.run()
            } catch {
                outHandle.readabilityHandler = nil
                errHandle.readabilityHandler = nil
                throw ProcessRunnerError.failedToStart(error.localizedDescription)
            }

            process = task
            stdoutHandle = outHandle
            stderrHandle = errHandle
        }
    }

    func stop() {
        stateQueue.sync {
            guard let process else {
                return
            }
            if process.isRunning {
                process.interrupt()
                DispatchQueue.global().asyncAfter(deadline: .now() + 1.2) { [weak self] in
                    self?.forceTerminateIfNeeded()
                }
            }
        }
    }

    private func forceTerminateIfNeeded() {
        stateQueue.sync {
            guard let process, process.isRunning else {
                return
            }
            process.terminate()
        }
    }

    private func cleanup() {
        stateQueue.sync {
            stdoutHandle?.readabilityHandler = nil
            stderrHandle?.readabilityHandler = nil
            stdoutHandle = nil
            stderrHandle = nil
            process = nil
        }
    }
}
