import Foundation

enum ConfigurationValidationError: LocalizedError {
    case pathIsEmpty(String)
    case fileNotFound(String)
    case directoryNotFound(String)
    case invalidNumber(String)

    var errorDescription: String? {
        switch self {
        case let .pathIsEmpty(name):
            return "Поле \(name) пустое"
        case let .fileNotFound(path):
            return "Файл не найден: \(path)"
        case let .directoryNotFound(path):
            return "Папка не найдена: \(path)"
        case let .invalidNumber(text):
            return text
        }
    }
}

protocol ConfigurationValidating {
    func validateForRemoval(_ configuration: RunConfiguration) throws
    func validateForViewer(_ configuration: RunConfiguration) throws
}

final class ConfigurationValidator: ConfigurationValidating {
    private let fileManager = FileManager.default

    func validateForRemoval(_ configuration: RunConfiguration) throws {
        try validateCommon(configuration)
        try validatePath(configuration.inputDirectory.path, fieldName: "Input directory")
        var isDirectory = ObjCBool(false)
        if !fileManager.fileExists(atPath: configuration.inputDirectory.path, isDirectory: &isDirectory) || !isDirectory.boolValue {
            throw ConfigurationValidationError.directoryNotFound(configuration.inputDirectory.path)
        }
        if configuration.maxSide < 0 {
            throw ConfigurationValidationError.invalidNumber("Max side должен быть >= 0")
        }
    }

    func validateForViewer(_ configuration: RunConfiguration) throws {
        try validateCommon(configuration)
    }

    private func validateCommon(_ configuration: RunConfiguration) throws {
        try validatePath(configuration.projectDirectory.path, fieldName: "Project directory")
        try validatePath(configuration.pythonExecutable.path, fieldName: "Python executable")
        try validatePath(configuration.outputDirectory.path, fieldName: "Output directory")

        if let maskDirectory = configuration.maskDirectory {
            try validatePath(maskDirectory.path, fieldName: "Mask directory")
        }

        if !fileManager.fileExists(atPath: configuration.pythonExecutable.path) {
            throw ConfigurationValidationError.fileNotFound(configuration.pythonExecutable.path)
        }

        var isDirectory = ObjCBool(false)
        if !fileManager.fileExists(atPath: configuration.projectDirectory.path, isDirectory: &isDirectory) || !isDirectory.boolValue {
            throw ConfigurationValidationError.directoryNotFound(configuration.projectDirectory.path)
        }
    }

    private func validatePath(_ path: String, fieldName: String) throws {
        if path.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw ConfigurationValidationError.pathIsEmpty(fieldName)
        }
    }
}
