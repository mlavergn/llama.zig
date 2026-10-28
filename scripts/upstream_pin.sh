# The upstream pin, for the bash generators. Source it.
#
# **Not a port.** `scripts/upstream_pin.py`'s counterpart: read from
# `build/upstream.zig`, the one place the pin is written down, so a
# regenerated golden says which commit it was captured at. The golden
# generators once printed "v0.3.0 (`c1d0e7a00`)" as a literal, which would
# have stamped every later capture with the wrong release.
#
# Sets UPSTREAM_TAG, UPSTREAM_COMMIT, UPSTREAM_VERSION and
# UPSTREAM_GGML_VERSION, and PIN_DEFS, the `-D`s that hand the first two to
# a C generator as PIN_TAG and PIN_COMMIT.

_pin_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build/upstream.zig"
_pin_field() {
    sed -n "s/^pub const $1 = \"\(.*\)\";\$/\1/p" "$_pin_file"
}
UPSTREAM_TAG="$(_pin_field tag)"
UPSTREAM_COMMIT="$(_pin_field commit)"
UPSTREAM_VERSION="$(_pin_field version)"
UPSTREAM_GGML_VERSION="$(_pin_field ggml_version)"
for _v in UPSTREAM_TAG UPSTREAM_COMMIT UPSTREAM_VERSION UPSTREAM_GGML_VERSION; do
    [ -n "${!_v}" ] || { echo "upstream_pin.sh: no $_v in $_pin_file" >&2; exit 2; }
done
PIN_DEFS=(-DPIN_TAG="\"$UPSTREAM_TAG\"" -DPIN_COMMIT="\"$UPSTREAM_COMMIT\"")
unset _v _pin_file
unset -f _pin_field
