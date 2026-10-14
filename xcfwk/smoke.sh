#!/bin/bash
# Output parity between the port's own libraries and llamazig.xcframework.
#
#   smoke.sh compare <label> <cpu|gpu> <outdir> <model> <npred> <ref-cmd> <test-cmd>
#   smoke.sh sim-udid <device-name>
#
# `compare` runs both commands -- `<cmd> <model> <prompt> <npred> <mode>` --
# over parity-port's six prompts and requires identical stdout. `ref-cmd` is
# `test/driver.c` linked against `zig build reference`; `test-cmd` is
# `test/smoke.swift` linked against the framework, on the host or through
# `xcrun simctl spawn`. The commands are word-split, so their paths must not
# contain spaces.
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

sim_udid() {
    xcrun simctl list devices available \
        | grep -E "^ +$1 \([0-9A-F-]+\)" | head -1 | sed -E 's/.*\(([0-9A-F-]+)\).*/\1/'
}

cmd="${1:?usage: smoke.sh compare|sim-udid ...}"; shift
case "$cmd" in
    compare)  compare "$@" ;;
    sim-udid) sim_udid "$@" ;;
    *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
