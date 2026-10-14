// swift-tools-version:5.9
//
// llamazig as a local SwiftPM package. Run `make -C xcfwk` first: the
// xcframework is build output and is not checked in.
//
//     .package(path: "<repo>/xcfwk")

import PackageDescription

let package = Package(
    name: "llamazig",
    platforms: [.iOS("18.0"), .macOS("26.0")],
    products: [
        .library(name: "llamazig", targets: ["llamazig"]),
    ],
    targets: [
        .binaryTarget(name: "llamazig", path: "build/llamazig.xcframework"),
    ]
)
