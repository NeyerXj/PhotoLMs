import Foundation

enum RuntimeArchitecture: String {
    case arm64
    case x86_64

    static var current: RuntimeArchitecture {
        var systemInfo = utsname()
        uname(&systemInfo)
        let machine = withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { ptr in
                String(cString: ptr)
            }
        }
        if machine.contains("x86_64") {
            return .x86_64
        }
        return .arm64
    }
}

enum AppBundlePaths {
    static func resourceDirectory() -> URL? {
        Bundle.main.resourceURL
    }

    static func bundledScriptURL(named name: String) -> URL? {
        guard let resources = resourceDirectory() else {
            return nil
        }
        let scriptURL = resources.appendingPathComponent("PythonScripts/\(name)")
        guard FileManager.default.fileExists(atPath: scriptURL.path) else {
            return nil
        }
        return scriptURL
    }

    static func bundledPythonExecutableURL() -> URL? {
        guard let resources = resourceDirectory() else {
            return nil
        }
        let pythonURL = resources.appendingPathComponent("Python.framework/Versions/Current/bin/python3")
        guard FileManager.default.fileExists(atPath: pythonURL.path) else {
            return nil
        }
        return pythonURL
    }

    static func bundledPythonHomeURL() -> URL? {
        guard let resources = resourceDirectory() else {
            return nil
        }
        let homeURL = resources.appendingPathComponent("Python.framework/Versions/Current")
        guard FileManager.default.fileExists(atPath: homeURL.path) else {
            return nil
        }
        return homeURL
    }

    static func bundledSitePackagesURL(for architecture: RuntimeArchitecture) -> URL? {
        guard let resources = resourceDirectory() else {
            return nil
        }
        let sitePackagesURL = resources.appendingPathComponent("PythonSitePackages/\(architecture.rawValue)")
        guard FileManager.default.fileExists(atPath: sitePackagesURL.path) else {
            return nil
        }
        return sitePackagesURL
    }

    static func defaultWorkspaceDirectory() -> URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("PhotoLMData", isDirectory: true)
    }
}
