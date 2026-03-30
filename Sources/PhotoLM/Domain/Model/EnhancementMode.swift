import Foundation

enum EnhancementMode: String, CaseIterable, Identifiable {
    case none
    case sharp
    case sharp2x

    var id: String {
        rawValue
    }

    var title: String {
        switch self {
        case .none:
            return "None"
        case .sharp:
            return "Sharp"
        case .sharp2x:
            return "Sharp2x"
        }
    }
}
