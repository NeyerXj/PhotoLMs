import Foundation

enum ProcessingMode: String, CaseIterable, Identifiable {
    case ui = "ui"
    case text = "text"
    case uiAndText = "ui+text"

    var id: String {
        rawValue
    }

    var title: String {
        switch self {
        case .ui:
            return "UI"
        case .text:
            return "Text"
        case .uiAndText:
            return "UI + Text"
        }
    }
}
