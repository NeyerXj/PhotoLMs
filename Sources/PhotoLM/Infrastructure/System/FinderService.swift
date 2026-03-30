import AppKit
import Foundation

protocol FinderOpening {
    @MainActor
    func open(path: String)
}

final class FinderService: FinderOpening {
    @MainActor
    func open(path: String) {
        let expanded = (path as NSString).expandingTildeInPath
        NSWorkspace.shared.open(URL(fileURLWithPath: expanded))
    }
}
