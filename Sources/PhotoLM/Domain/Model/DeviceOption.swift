import Foundation

enum DeviceOption: String, CaseIterable, Identifiable {
    case auto
    case cpu
    case mps
    case cuda

    var id: String {
        rawValue
    }

    var title: String {
        switch self {
        case .auto:
            return "Auto"
        case .cpu:
            return "CPU"
        case .mps:
            return "MPS"
        case .cuda:
            return "CUDA"
        }
    }
}
