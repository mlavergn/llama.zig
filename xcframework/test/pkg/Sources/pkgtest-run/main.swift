import PkgTest

let info = systemInfo()
print("pkgtest: \(info)")
if !info.contains("CPU") { fatalError("unexpected system info") }
