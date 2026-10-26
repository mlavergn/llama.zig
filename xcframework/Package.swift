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
    platforms: [.iOS("16.4"), .macOS("13.3"), .tvOS("16.4"), .visionOS("1.0")],
    products: [
        .library(name: "llama", targets: ["llama"]),
    ],
    targets: [
        .binaryTarget(name: "llama", path: "build/llama.xcframework"),
    ]
)
