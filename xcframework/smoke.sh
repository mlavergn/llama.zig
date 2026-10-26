#!/bin/bash
# Output parity between the port's own libraries and llamazig.xcframework.
#
#   smoke.sh compare <label> <cpu|gpu> <outdir> <model> <npred> <ref-cmd> <test-cmd>
#   smoke.sh sim-udid <device-name>
#   smoke.sh device <label> <outdir> <model> <ref-cmd> <log>
#
# `compare` runs both commands -- `<cmd> <model> <prompt> <npred> <mode>` --
# over parity-port's six prompts and requires identical stdout. `ref-cmd` is
# `test/driver.c` linked against `zig build reference`; `test-cmd` is
# `test/smoke.swift` linked against the framework, on the host or through
# `xcrun simctl spawn`. The commands are word-split, so their paths must not
# contain spaces.
#
# `device` compares what the iOS app in `test/device` printed on a phone --
# one `SMOKE <mode> <n> <base64>` line per prompt and device, read from
# `log` -- with `ref-cmd` run here on the same prompts, CPU and Metal. The
# app predicts 32 tokens, so the reference does too.
#
# `sim-udid` prints the UDID of the first available simulator device with
# exactly that name.
set -uo pipefail

PROMPTS=(
    "The capital of France is"
    "Explain gravity in one sentence:"
    "def fibonacci(n):"
    "Write a haiku about winter:"
    "List three prime numbers:"
    "Translate to French: good morning"
)

compare() {
    local label="$1" mode="$2" out="$3" model="$4" npred="$5" ref="$6" test="$7"
    mkdir -p "$out"
    [ -f "$model" ] || { echo "FAIL: missing model $model" >&2; exit 2; }
    local pass=0 fail=0 i=0 p a b
    for p in "${PROMPTS[@]}"; do
        i=$((i + 1))
        a="$out/$label-$mode-$i.ref.txt"; b="$out/$label-$mode-$i.test.txt"
        $ref "$model" "$p" "$npred" "$mode" > "$a" 2> "$a.log"; local rc_ref=$?
        $test "$model" "$p" "$npred" "$mode" > "$b" 2> "$b.log"; local rc_test=$?
        printf '  %-8s %-4s %-36s ' "$label" "$mode" "$(echo "$p" | cut -c1-34)"
        if [ "$rc_ref" -ne 0 ] || [ "$rc_test" -ne 0 ]; then
            echo "FAIL (exit: ref=$rc_ref test=$rc_test; see $b.log)"; fail=$((fail + 1))
        elif [ "$(tr -d '[:space:]' < "$a" | wc -c)" -eq 0 ]; then
            echo "FAIL (reference produced no text)"; fail=$((fail + 1))
        elif cmp -s "$a" "$b"; then
            echo "identical"; pass=$((pass + 1))
        else
            echo "DIFFERS"; diff "$a" "$b" | head -6; fail=$((fail + 1))
        fi
    done
    if [ "$fail" -eq 0 ]; then
        echo "  PASS: $label $mode $pass/$pass prompts identical"
        return 0
    fi
    echo "FAIL: $label $mode $fail of $((pass + fail)) prompts differ" >&2
    return 1
}

device() {
    local label="$1" out="$2" model="$3" ref="$4" log="$5"
    mkdir -p "$out"
    [ -f "$model" ] || { echo "FAIL: missing model $model" >&2; exit 2; }
    grep -q '^SMOKE-DONE' "$log" || { echo "FAIL: the app did not finish -- $(grep -m1 -E '^SMOKE-FAIL|terminated' "$log")" >&2; exit 1; }
    echo "  on $(sed -n 's/^SMOKE-DEVICE //p' "$log")"
    local pass=0 fail=0 mode i p a b line
    for mode in cpu gpu; do
        i=0
        for p in "${PROMPTS[@]}"; do
            i=$((i + 1))
            a="$out/$label-$mode-$i.ref.txt"; b="$out/$label-$mode-$i.test.txt"
            $ref "$model" "$p" 32 "$mode" > "$a" 2> "$a.log"
            line=$(grep -m1 "^SMOKE $mode $i " "$log" | cut -d' ' -f4)
            printf '%s' "$line" | base64 -D > "$b"
            printf '  %-8s %-4s %-36s ' "$label" "$mode" "$(echo "$p" | cut -c1-34)"
            if [ -z "$line" ]; then
                echo "FAIL (no output from the device)"; fail=$((fail + 1))
            elif [ "$(tr -d '[:space:]' < "$a" | wc -c)" -eq 0 ]; then
                echo "FAIL (reference produced no text)"; fail=$((fail + 1))
            elif cmp -s "$a" "$b"; then
                echo "identical"; pass=$((pass + 1))
            else
                echo "DIFFERS"; diff "$a" "$b" | head -6; fail=$((fail + 1))
            fi
        done
    done
    if [ "$fail" -eq 0 ]; then
        echo "  PASS: $label $pass/$pass prompts identical, CPU and Metal"
        return 0
    fi
    echo "FAIL: $label $fail of $((pass + fail)) prompts differ" >&2
    return 1
}

sim_udid() {
    # The name is matched literally: "Apple TV 4K (3rd generation)" holds
    # parentheses, and its "(at 1080p)" sibling must not match.
    local esc
    esc="$(printf '%s' "$1" | sed 's/[][\.*^$()+?{}|/]/\\&/g')"
    xcrun simctl list devices available \
        | grep -E "^ +$esc \([0-9A-F-]{36}\)" | head -1 | sed -E 's/.*\(([0-9A-F-]{36})\).*/\1/'
}

cmd="${1:?usage: smoke.sh compare|device|sim-udid ...}"; shift
case "$cmd" in
    compare)  compare "$@" ;;
    device)   device "$@" ;;
    sim-udid) sim_udid "$@" ;;
    *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
