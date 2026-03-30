import Foundation

struct ScriptCommand {
    let executableURL: URL
    let arguments: [String]
    let workingDirectoryURL: URL
    let environment: [String: String]

    var displayLine: String {
        ([executableURL.path] + arguments).map(Self.quoted).joined(separator: " ")
    }

    private static func quoted(_ value: String) -> String {
        if value.isEmpty {
            return "\"\""
        }
        if value.contains(" ") || value.contains("\"") {
            let escaped = value.replacingOccurrences(of: "\"", with: "\\\"")
            return "\"\(escaped)\""
        }
        return value
    }
}
