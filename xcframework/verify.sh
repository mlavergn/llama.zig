#!/bin/bash
# The gates for llama.xcframework. Each subcommand is one gate: it prints
# what it checked and exits non-zero on failure. `make verify` runs them on the
# real build, `make compat` holds the result to upstream's own xcframework,
# and `make negative` runs them on deliberately broken inputs and requires
# each to fail -- a gate proves nothing until it has been made to.
#
#   verify.sh platform <platform-number> <minos> <archive-or-object>...
#   verify.sh symbols <object>
#   verify.sh link <sdk> <clang-target> <framework-parent-dir> <out>
#   verify.sh plist <Info.plist>
#   verify.sh dsym <binary> <dSYM>
#   verify.sh xcframework <path.xcframework>
#   verify.sh compat <ours.xcframework> <upstream.xcframework>
#
# Mach-O platform numbers: 1 macOS, 2 iOS, 3 tvOS, 7 iOS simulator, 8 tvOS
# simulator, 11 visionOS, 12 visionOS simulator.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

fail() { echo "FAIL: $*" >&2; exit 1; }

# Every member of every archive (or the object itself) carries an
# LC_BUILD_VERSION with this platform and minimum OS. A member with none fails
# too: an untagged object is how a host build slips into a slice.
gate_platform() {
    local plat="$1" minos="$2"; shift 2
    local f out
    for f in "$@"; do
        [ -f "$f" ] || fail "platform: missing $f"
        out="$(otool -l "$f" | awk -v P="$plat" -v M="$minos" '
            function check() {
                n++
                if (!has) { bad++; print "  no build version: " cur }
                else if (!ok) { bad++; print "  wrong platform/minos (" got "): " cur }
            }
            /^[^ \t].*:$/ { if (cur != "") check(); cur = $0; has = 0; ok = 1; got = ""; inb = 0; next }
            /cmd LC_BUILD_VERSION/ { inb = 1; next }
            inb && $1 == "platform" { has = 1; got = got $2; if ($2 != P) ok = 0; next }
            inb && $1 == "minos" { got = got "/" $2; if ($2 != M) ok = 0; inb = 0; next }
            END { if (cur != "") check(); printf "%d %d\n", n, bad }
        ')"
        local counts n bad
        counts="$(echo "$out" | tail -1)"
        n="${counts% *}"; bad="${counts#* }"
        echo "$out" | sed -e '$d' | sed -n '1,5p'
        [ "$n" -gt 0 ] || fail "platform: no objects in $f"
        [ "$bad" -eq 0 ] || fail "platform: $bad of $n objects in $(basename "$f") are not platform $plat minos $minos"
        echo "  platform $plat minos $minos: $n/$n objects in $(basename "$f")"
    done
}

# The prelinked object exports the C symbols exports.txt names -- llama_,
# ggml_, gguf_, mtmd_, and ggml-quants.c's quantize_/dequantize_/iq2xs_/iq3xs_
# that upstream's dylib exports too -- and nothing else: no private externs left for the dylib to export by accident, no
# libc++ visible in either direction, and the entry points of every library
# upstream's framework carries actually present.
gate_symbols() {
    local obj="$1" tmp
    [ -f "$obj" ] || fail "symbols: missing $obj"
    tmp="$(mktemp)"; trap 'rm -f "$tmp"' RETURN
    nm -m "$obj" > "$tmp"
    local private exported stray cxx
    private="$(grep 'private external' "$tmp" | grep -vc 'non-external' || true)"
    exported="$(grep -v '(undefined)' "$tmp" | grep ' external ' | grep -v 'non-external' | awk '{print $NF}' || true)"
    stray="$(echo "$exported" | grep -v '^$' | grep -Ev '^_(llama|ggml|gguf|mtmd|quantize|dequantize|iq2xs|iq3xs)_' || true)"
    cxx="$(grep ' external ' "$tmp" | grep -v 'non-external' | grep -c 'St3__1' || true)"
    [ "$private" -eq 0 ] || fail "symbols: $private symbols still private external (not localized)"
    [ -z "$stray" ] || fail "symbols: exported outside exports.txt's prefixes: $(echo "$stray" | head -5 | tr '\n' ' ')"
    [ "$cxx" -eq 0 ] || fail "symbols: $cxx std::__1 symbols external (defined or undefined)"
    local sym
    for sym in _llama_backend_init _llama_model_load_from_file _llama_decode _llama_sampler_init_greedy \
               _ggml_init _ggml_backend_metal_init _ggml_backend_blas_init _gguf_init_empty \
               _mtmd_init_from_file _mtmd_tokenize _mtmd_helper_eval_chunks _mtmd_helper_bitmap_init_from_file; do
        grep -qx "$sym" <<< "$exported" || fail "symbols: $sym is not exported"
    done
    # The math a `zig cc` link -- every bit-exact gate -- takes from compiler_rt
    # must be inside the object, or the dylib's link resolves it to Apple's
    # libm and the CPU path diverges from the port. (No `| grep -q` anywhere
    # here: under pipefail its early exit SIGPIPEs the writer and reads as a
    # miss.)
    for sym in _expf _logf _sinf _cosf _exp2f; do
        grep -Eq "^[0-9a-f]+ \([^)]*\) non-external.* $sym\$" "$tmp" \
            || fail "symbols: $sym is not defined inside the object (compiler_rt missing; libm would supply it)"
    done
    echo "  symbols: $(echo "$exported" | grep -c .) exported, all in exports.txt; 0 private external; 0 std::__1; compiler_rt math inside"
}

# `test/stub.m` links against the framework with nothing else named: the
# module map has to autolink the system frameworks. The framework's own
# binary must need no C++ runtime -- upstream's module map links libc++ for
# the app regardless, but nothing of ours depends on it.
gate_link() {
    local sdk="$1" target="$2" fwdir="$3" out="$4"
    mkdir -p "$(dirname "$out")"
    xcrun --sdk "$sdk" clang -target "$target" -fmodules -fmodules-cache-path="$(dirname "$out")/modules" \
        -F "$fwdir" -framework llama "$HERE/test/stub.m" -o "$out" \
        || fail "link: stub did not link for $target"
    local bin
    bin="$(find "$fwdir/llama.framework" -name llama -type f | head -1)"
    if otool -L "$bin" | grep 'libc++' > /dev/null; then fail "link: $bin depends on libc++.dylib"; fi
    echo "  link: stub links for $target; framework depends on $(otool -L "$bin" | grep -v ':$' | awk '{print $1}' | grep -v '^@rpath/llama' | xargs -n1 basename | sort -u | tr '\n' ' ')"
}

gate_plist() {
    plutil -lint "$1" >/dev/null || fail "plist: $1 does not lint"
    echo "  plist: $(basename "$(dirname "$1")")/$(basename "$1") lints"
}

# The dSYM belongs to the binary -- every architecture's UUID matches -- and
# carries DWARF: a dSYM of nothing is what an object compiled without debug
# info, or stripped before dsymutil ran, produces.
gate_dsym() {
    local bin="$1" dsym="$2" a b
    [ -d "$dsym" ] || fail "dsym: missing $dsym"
    a="$(xcrun dwarfdump --uuid "$bin" | awk '{print $2, $3}' | sort)"
    b="$(xcrun dwarfdump --uuid "$dsym" | awk '{print $2, $3}' | sort)"
    [ -n "$a" ] && [ "$a" = "$b" ] || fail "dsym: UUIDs differ: binary [$a] dSYM [$b]"
    local info
    info="$(xcrun size -m -arch arm64 "$dsym/Contents/Resources/DWARF/llama" 2>/dev/null \
        | awk '/__debug_info/ {print $(NF)}' | head -1)"
    [ -n "$info" ] && [ "$info" -gt 0 ] || fail "dsym: no __debug_info in $dsym"
    echo "  dsym: $(echo "$a" | wc -l | tr -d ' ') UUIDs match; __debug_info $info bytes (arm64)"
}

# Exactly upstream's seven libraries, each declaring the right platform,
# variant and architectures.
gate_xcframework() {
    local p="$1/Info.plist" got want
    [ -f "$p" ] || fail "xcframework: missing $p"
    got="$(xcf_libraries "$p")"
    want="ios-arm64:ios::arm64 ios-arm64_x86_64-simulator:ios:simulator:arm64,x86_64 macos-arm64_x86_64:macos::arm64,x86_64 tvos-arm64:tvos::arm64 tvos-arm64_x86_64-simulator:tvos:simulator:arm64,x86_64 xros-arm64:xros::arm64 xros-arm64_x86_64-simulator:xros:simulator:arm64,x86_64 "
    [ "$got" = "$want" ] || fail "xcframework: libraries are '$got', want '$want'"
    echo "  xcframework: $got"
}

# id:platform:variant:archs for every library in an xcframework's Info.plist,
# sorted.
xcf_libraries() {
    local p="$1" n i id plat variant archs got=""
    n="$(plutil -extract AvailableLibraries raw "$p")"
    for ((i = 0; i < n; i++)); do
        id="$(plutil -extract "AvailableLibraries.$i.LibraryIdentifier" raw "$p")"
        plat="$(plutil -extract "AvailableLibraries.$i.SupportedPlatform" raw "$p")"
        variant="$(plutil -extract "AvailableLibraries.$i.SupportedPlatformVariant" raw "$p" 2>/dev/null || true)"
        archs="$(plutil -extract "AvailableLibraries.$i.SupportedArchitectures" json -o - "$p" | tr -d '[]"' )"
        got="$got$id:$plat:$variant:$archs "
    done
    echo "$got" | tr ' ' '\n' | grep . | LC_ALL=C sort | tr '\n' ' '
}

# Load commands that identify a dylib, per architecture: its type, install
# name, platform, minimum OS and SDK.
binary_identity() {
    local bin="$1" a
    for a in $(lipo -archs "$bin"); do
        printf '%s: ' "$a"
        otool -arch "$a" -hlv "$bin" | awk '
            $1 ~ /^MH_MAGIC/ { print "type=" $5 }
            /cmd LC_ID_DYLIB/ { id = 1; next }
            id && $1 == "name" { print "id=" $2; id = 0 }
            /cmd LC_BUILD_VERSION/ { bv = 1; next }
            bv && $1 == "platform" { print "platform=" $2 }
            bv && $1 == "minos" { print "minos=" $2 }
            bv && $1 == "sdk" { print "sdk=" $2; bv = 0 }
        ' | tr '\n' ' '
        echo
    done
}

# Ours against upstream's: the same libraries; in every slice the same files
# and links, and every file but the binary byte-identical; each binary the
# same architectures, type, install name, platform, minimum OS and SDK, with
# no dependency upstream's lacks; and every function the shipped headers
# declare that upstream's binary exports, exported by ours too -- per slice,
# per architecture.
gate_compat() {
    local ours="$1" up="$2" lib o u f
    [ -d "$ours" ] && [ -d "$up" ] || fail "compat: missing $ours or $up"
    o="$(xcf_libraries "$ours/Info.plist")"; u="$(xcf_libraries "$up/Info.plist")"
    [ "$o" = "$u" ] || fail "compat: libraries differ: ours '$o' upstream '$u'"
    local nfiles=0 napi=0 nextra_total=0
    for lib in $(cd "$up" && ls -d */ | tr -d /); do
        local ofw="$ours/$lib/llama.framework" ufw="$up/$lib/llama.framework"
        o="$(cd "$ofw" && find . | LC_ALL=C sort | while read -r f; do
            if [ -L "$f" ]; then echo "$f -> $(readlink "$f")"; elif [ -d "$f" ]; then echo "$f/"; else echo "$f"; fi; done)"
        u="$(cd "$ufw" && find . | LC_ALL=C sort | while read -r f; do
            if [ -L "$f" ]; then echo "$f -> $(readlink "$f")"; elif [ -d "$f" ]; then echo "$f/"; else echo "$f"; fi; done)"
        [ "$o" = "$u" ] || fail "compat: $lib: bundle trees differ: $(diff <(echo "$u") <(echo "$o") | head -5 | tr '\n' ' ')"
        local bin
        bin="$(plutil -extract AvailableLibraries json -o - "$up/Info.plist" \
            | python3 -c 'import json,sys; print([l["BinaryPath"] for l in json.load(sys.stdin) if l["LibraryIdentifier"]==sys.argv[1]][0])' "$lib")"
        while read -r f; do
            [ "$f" = "./${bin#llama.framework/}" ] && continue
            cmp -s "$ofw/$f" "$ufw/$f" || fail "compat: $lib: $f is not upstream's"
            nfiles=$((nfiles + 1))
        done < <(cd "$ufw" && find . -type f | LC_ALL=C sort)
        [ -d "$ours/$lib/dSYMs/llama.dSYM" ] || fail "compat: $lib: no dSYMs/llama.dSYM"

        local ob="$ours/$lib/$bin" ub="$up/$lib/$bin"
        o="$(binary_identity "$ob")"; u="$(binary_identity "$ub")"
        [ "$o" = "$u" ] || fail "compat: $lib: binary identity differs:$(diff <(echo "$u") <(echo "$o") | grep '^[<>]' | tr '\n' ' ')"
        local odeps udeps extra
        odeps="$(otool -L "$ob" | grep -v ':$' | awk '{print $1}' | grep -v '^@rpath/llama' | sort -u)"
        udeps="$(otool -L "$ub" | grep -v ':$' | awk '{print $1}' | grep -v '^@rpath/llama' | sort -u)"
        extra="$(comm -23 <(echo "$odeps") <(echo "$udeps"))"
        [ -z "$extra" ] || fail "compat: $lib: ours links what upstream's does not: $extra"

        local idents a want have missing nup nours absent
        idents="$(cat "$ofw"/Headers/*.h | grep -ohE '\b[A-Za-z_][A-Za-z0-9_]*\b' | sed 's/^/_/' | LC_ALL=C sort -u)"
        for a in $(lipo -archs "$ub"); do
            want="$(nm -gU -arch "$a" "$ub" | awk '{print $3}' | LC_ALL=C sort -u)"
            have="$(nm -gU -arch "$a" "$ob" | awk '{print $3}' | LC_ALL=C sort -u)"
            nup="$(LC_ALL=C comm -12 <(echo "$want") <(echo "$idents"))"
            missing="$(LC_ALL=C comm -23 <(echo "$nup") <(echo "$have"))"
            [ -z "$missing" ] || fail "compat: $lib $a: $(echo "$missing" | wc -l | tr -d ' ') header-declared symbols upstream exports and ours does not: $(echo "$missing" | head -5 | tr '\n' ' ')"
            napi=$((napi + $(echo "$nup" | grep -c .)))
            absent="$(LC_ALL=C comm -23 <(echo "$want" | grep -v '^__Z') <(echo "$have") | grep -c . || true)"
            nextra_total=$((nextra_total + absent))
        done
        echo "  $lib: tree and files identical; $(lipo -archs "$ub" | tr ' ' ','): $(binary_identity "$ob" | head -1 | cut -d' ' -f2- | sed 's/ $//')"
    done
    echo "  compat: $nfiles files byte-identical; $napi header-declared exports present (slice x arch); upstream's internal C symbols not exported by ours: $nextra_total (slice x arch)"
}

cmd="${1:?usage: verify.sh platform|symbols|link|plist|dsym|xcframework|compat ...}"; shift
case "$cmd" in
    platform)    gate_platform "$@" ;;
    symbols)     gate_symbols "$@" ;;
    link)        gate_link "$@" ;;
    plist)       gate_plist "$@" ;;
    dsym)        gate_dsym "$@" ;;
    xcframework) gate_xcframework "$@" ;;
    compat)      gate_compat "$@" ;;
    *) fail "unknown gate $cmd" ;;
esac
