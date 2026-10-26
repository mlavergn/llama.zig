#!/bin/bash
# One slice's llama.framework bundle, laid out exactly as upstream's
# build-xcframework.sh lays it out: its `setup_framework_structure`
# (llama.cpp/build-xcframework.sh:126 @c1d0e7a00), transcribed. The headers
# come from the pinned llama.cpp/, the module map and Info.plist are
# upstream's text, so `verify.sh compat` can require every file but the
# binary to be byte-identical to upstream's own xcframework.
#
#   bundle.sh <llama.cpp root> <framework-parent-dir> <min-os> <ios|macos|visionos|tvos>
#
# Writes <framework-parent-dir>/llama.framework without its binary.
set -euo pipefail

src="$1" build_dir="$2" min_os_version="$3" platform="$4"
framework_name="llama"

if [[ "$platform" == "macos" ]]; then
    # macOS versioned structure uses versioned directories
    mkdir -p ${build_dir}/${framework_name}.framework/Versions/A/Headers
    mkdir -p ${build_dir}/${framework_name}.framework/Versions/A/Modules
    mkdir -p ${build_dir}/${framework_name}.framework/Versions/A/Resources

    # Create symbolic links
    ln -sf A ${build_dir}/${framework_name}.framework/Versions/Current
    ln -sf Versions/Current/Headers ${build_dir}/${framework_name}.framework/Headers
    ln -sf Versions/Current/Modules ${build_dir}/${framework_name}.framework/Modules
    ln -sf Versions/Current/Resources ${build_dir}/${framework_name}.framework/Resources
    ln -sf Versions/Current/${framework_name} ${build_dir}/${framework_name}.framework/${framework_name}

    header_path=${build_dir}/${framework_name}.framework/Versions/A/Headers/
    module_path=${build_dir}/${framework_name}.framework/Versions/A/Modules/
else
    # iOS/VisionOS/tvOS use a flat structure
    mkdir -p ${build_dir}/${framework_name}.framework/Headers
    mkdir -p ${build_dir}/${framework_name}.framework/Modules
    rm -rf ${build_dir}/${framework_name}.framework/Versions

    header_path=${build_dir}/${framework_name}.framework/Headers/
    module_path=${build_dir}/${framework_name}.framework/Modules/
fi

cp "$src"/include/llama.h             ${header_path}
cp "$src"/ggml/include/ggml.h         ${header_path}
cp "$src"/ggml/include/ggml-opt.h     ${header_path}
cp "$src"/ggml/include/ggml-alloc.h   ${header_path}
cp "$src"/ggml/include/ggml-backend.h ${header_path}
cp "$src"/ggml/include/ggml-metal.h   ${header_path}
cp "$src"/ggml/include/ggml-cpu.h     ${header_path}
cp "$src"/ggml/include/ggml-blas.h    ${header_path}
cp "$src"/ggml/include/gguf.h         ${header_path}
cp "$src"/tools/mtmd/mtmd.h           ${header_path}
cp "$src"/tools/mtmd/mtmd-helper.h    ${header_path}

# Upstream's module map, verbatim. `link "c++"` is upstream's; this binary
# needs no C++ runtime, and an app autolinking libc++ through it loses
# nothing.
cat > ${module_path}module.modulemap << EOF
framework module llama {
    umbrella "Headers"

    link "c++"
    link framework "Accelerate"
    link framework "Metal"
    link framework "Foundation"

    export *
}
EOF

case "$platform" in
    "ios")
        platform_name="iphoneos"
        sdk_name="iphoneos${min_os_version}"
        supported_platform="iPhoneOS"
        plist_path="${build_dir}/${framework_name}.framework/Info.plist"
        device_family='    <key>UIDeviceFamily</key>
    <array>
        <integer>1</integer>
        <integer>2</integer>
    </array>'
        ;;
    "macos")
        platform_name="macosx"
        sdk_name="macosx${min_os_version}"
        supported_platform="MacOSX"
        plist_path="${build_dir}/${framework_name}.framework/Versions/A/Resources/Info.plist"
        device_family=""
        ;;
    "visionos")
        platform_name="xros"
        sdk_name="xros${min_os_version}"
        supported_platform="XRPlatform"
        plist_path="${build_dir}/${framework_name}.framework/Info.plist"
        device_family=""
        ;;
    "tvos")
        platform_name="appletvos"
        sdk_name="appletvos${min_os_version}"
        supported_platform="AppleTVOS"
        plist_path="${build_dir}/${framework_name}.framework/Info.plist"
        device_family='    <key>UIDeviceFamily</key>
    <array>
        <integer>3</integer>
    </array>'
        ;;
    *) echo "bundle.sh: unknown platform $platform" >&2; exit 2 ;;
esac

cat > ${plist_path} << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>llama</string>
    <key>CFBundleIdentifier</key>
    <string>org.ggml.llama</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>llama</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>${min_os_version}</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>${supported_platform}</string>
    </array>${device_family}
    <key>DTPlatformName</key>
    <string>${platform_name}</string>
    <key>DTSDKName</key>
    <string>${sdk_name}</string>
</dict>
</plist>
EOF
