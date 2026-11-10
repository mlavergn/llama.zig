// swift-tools-version:5.9
//
// llama.xcframework as a local SwiftPM package. Run `make -C xcframework`
// first: the xcframework is build output and is not checked in.
//
//     .package(path: "<repo>/xcframework")
//
// The product and module are `llama`, as upstream's framework's are.

import PackageDescription

let package = Package(
    name: "llama",
    platforms: [.iOS("26.0"), .macOS("26.0"), .tvOS("26.0"), .visionOS("26.0")],
    products: [
        .library(name: "llama", targets: ["llama"]),
    ],
    targets: [
        .binaryTarget(name: "llama", path: "build/llama.xcframework"),
    ]
)
