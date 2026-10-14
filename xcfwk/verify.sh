#!/bin/bash
# The gates for llamazig.xcframework. Each subcommand is one gate: it prints
# what it checked and exits non-zero on failure. `make verify` runs them on the
# real build; `make negative` runs them on deliberately broken inputs and
# requires each to fail -- a gate proves nothing until it has been made to.
#
#   verify.sh platform <platform-number> <minos> <archive-or-object>...
#   verify.sh symbols <object>
#   verify.sh link <sdk> <clang-target> <framework-parent-dir> <out>
#   verify.sh plist <Info.plist>
#   verify.sh xcframework <path.xcframework>
#
# Mach-O platform numbers: 1 macOS, 2 iOS, 7 iOS simulator.
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

# The prelinked object exports llama_/ggml_/gguf_ C symbols and nothing else:
# no private externs left for an app's link to bind to, no libc++ visible in
# either direction, and the entry points an app needs actually present.
gate_symbols() {
    local obj="$1" tmp
    [ -f "$obj" ] || fail "symbols: missing $obj"
    tmp="$(mktemp)"; trap 'rm -f "$tmp"' RETURN
    nm -m "$obj" > "$tmp"
    local private exported stray cxx
    private="$(grep 'private external' "$tmp" | grep -vc 'non-external' || true)"
    exported="$(grep -v '(undefined)' "$tmp" | grep ' external ' | grep -v 'non-external' | awk '{print $NF}' || true)"
    stray="$(echo "$exported" | grep -v '^$' | grep -Ev '^_(llama|ggml|gguf)_' || true)"
    cxx="$(grep ' external ' "$tmp" | grep -v 'non-external' | grep -c 'St3__1' || true)"
    [ "$private" -eq 0 ] || fail "symbols: $private symbols still private external (not localized)"
    [ -z "$stray" ] || fail "symbols: exported outside llama_/ggml_/gguf_: $(echo "$stray" | head -5 | tr '\n' ' ')"
    [ "$cxx" -eq 0 ] || fail "symbols: $cxx std::__1 symbols external (defined or undefined)"
    local sym
    for sym in _llama_backend_init _llama_model_load_from_file _llama_decode _llama_sampler_init_greedy \
               _ggml_init _ggml_backend_metal_init _gguf_init_empty; do
        grep -qx "$sym" <<< "$exported" || fail "symbols: $sym is not exported"
    done
    # The math a `zig cc` link -- every bit-exact gate -- takes from compiler_rt
    # must be inside the object, or an app's link resolves it to Apple's libm
    # and the CPU path diverges from the port. (No `| grep -q` anywhere here:
    # under pipefail its early exit SIGPIPEs the writer and reads as a miss.)
    for sym in _expf _logf _sinf _cosf _exp2f; do
        grep -Eq "^[0-9a-f]+ \([^)]*\) non-external.* $sym\$" "$tmp" \
            || fail "symbols: $sym is not defined inside the object (compiler_rt missing; libm would supply it)"
    done
    echo "  symbols: $(echo "$exported" | grep -c .) exported, all llama_/ggml_/gguf_; 0 private external; 0 std::__1; compiler_rt math inside"
}

# `test/stub.m` links against the framework with nothing else named: the
# module map has to autolink the system frameworks and the binary has to carry
# its own libc++. The result must not depend on libc++.dylib.
gate_link() {
    local sdk="$1" target="$2" fwdir="$3" out="$4"
    mkdir -p "$(dirname "$out")"
    xcrun --sdk "$sdk" clang -target "$target" -fmodules -fmodules-cache-path="$(dirname "$out")/modules" \
        -F "$fwdir" -framework llamazig "$HERE/test/stub.m" -o "$out" \
        || fail "link: stub did not link for $target"
    if otool -L "$out" | grep 'libc++' > /dev/null; then fail "link: stub depends on libc++.dylib"; fi
    echo "  link: stub links for $target; $(otool -L "$out" | tail -n +2 | awk '{print $1}' | xargs -n1 basename | tr '\n' ' ')"
}

gate_plist() {
    plutil -lint "$1" >/dev/null || fail "plist: $1 does not lint"
    echo "  plist: $(basename "$(dirname "$1")")/$(basename "$1") lints"
}

# Exactly the three slices, each declaring the right platform and variant.
gate_xcframework() {
    local p="$1/Info.plist" n i id plat variant got
    [ -f "$p" ] || fail "xcframework: missing $p"
    n="$(plutil -extract AvailableLibraries raw "$p")"
    got=""
    for ((i = 0; i < n; i++)); do
        id="$(plutil -extract "AvailableLibraries.$i.LibraryIdentifier" raw "$p")"
        plat="$(plutil -extract "AvailableLibraries.$i.SupportedPlatform" raw "$p")"
        variant="$(plutil -extract "AvailableLibraries.$i.SupportedPlatformVariant" raw "$p" 2>/dev/null || true)"
        got="$got$id:$plat:$variant "
    done
    got="$(echo "$got" | tr ' ' '\n' | grep . | LC_ALL=C sort | tr '\n' ' ')"
    local want="ios-arm64-simulator:ios:simulator ios-arm64:ios: macos-arm64:macos: "
    [ "$got" = "$want" ] || fail "xcframework: slices are '$got', want '$want'"
    echo "  xcframework: $got"
}

cmd="${1:?usage: verify.sh platform|symbols|link|plist|xcframework ...}"; shift
case "$cmd" in
    platform)    gate_platform "$@" ;;
    symbols)     gate_symbols "$@" ;;
    link)        gate_link "$@" ;;
    plist)       gate_plist "$@" ;;
    xcframework) gate_xcframework "$@" ;;
    *) fail "unknown gate $cmd" ;;
esac
