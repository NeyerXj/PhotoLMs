import Foundation

struct RunConfiguration {
    let projectDirectory: URL
    let pythonExecutable: URL
    let inputDirectory: URL
    let outputDirectory: URL
    let maskDirectory: URL?
    let mode: ProcessingMode
    let device: DeviceOption
    let enhancement: EnhancementMode
    let maxSide: Int
    let tightTextMask: Bool
    let poissonBlend: Bool
    let offline: Bool
}
