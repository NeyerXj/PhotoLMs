// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "PhotoLM",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "PhotoLM", targets: ["PhotoLM"])
    ],
    targets: [
        .executableTarget(
            name: "PhotoLM",
            path: "Sources/PhotoLM"
        )
    ]
)
