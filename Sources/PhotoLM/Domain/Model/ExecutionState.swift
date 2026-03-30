import SwiftUI

enum ExecutionState {
    case idle
    case running
    case success
    case failure

    var title: String {
        switch self {
        case .idle:
            return "Готово"
        case .running:
            return "Выполняется"
        case .success:
            return "Успешно"
        case .failure:
            return "Ошибка"
        }
    }

    var symbolName: String {
        switch self {
        case .idle:
            return "circle"
        case .running:
            return "hourglass"
        case .success:
            return "checkmark.circle.fill"
        case .failure:
            return "xmark.circle.fill"
        }
    }

    var color: Color {
        switch self {
        case .idle:
            return .secondary
        case .running:
            return .orange
        case .success:
            return .green
        case .failure:
            return .red
        }
    }
}
