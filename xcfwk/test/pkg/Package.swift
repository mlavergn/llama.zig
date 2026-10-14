// swift-tools-version:5.9
//
// A consumer of `xcfwk/Package.swift`, for `make pkg-test`. The library
// target builds for the iOS simulator through `xcodebuild`; the executable
// links and runs on macOS through `swift run`.

import PackageDescription

let package = Package(
    name: "PkgTest",
    platforms: [.iOS("18.0"), .macOS("26.0")],
    products: [
        .library(name: "PkgTest", targets: ["PkgTest"]),
    ],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .target(name: "PkgTest", dependencies: [.product(name: "llamazig", package: "xcfwk")]),
        .executableTarget(name: "pkgtest-run", dependencies: ["PkgTest"]),
    ]
)
