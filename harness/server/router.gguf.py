#!/usr/bin/env python3
"""Copy a GGUF file with metadata keys set, for the router and hub cases
(harness/server/router.setup, hub.setup): a decision model's
"<arch>.decision.type" is what GET /models reads for its output modalities
(server-models.cpp update_caps, common_get_decision_type). Pure Python, so
no numpy: the tensor data is copied as it is, after the header is rewritten
and padded to the file's alignment.

Usage: router.gguf.py <in> <out> KEY=str:VALUE|KEY=u32:VALUE|KEY=arch-str:VALUE ...
  arch-str: the key's "{arch}" is replaced by the file's general.architecture.
"""

import struct
import sys

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_str(b, o):
    (n,) = struct.unpack_from("<Q", b, o)
    return b[o + 8:o + 8 + n], o + 8 + n


def skip_val(b, o, t):
    if t in SIZES:
        return o + SIZES[t]
    if t == 8:
        return read_str(b, o)[1]
    if t == 9:
        et, n = struct.unpack_from("<IQ", b, o)
        o += 12
        for _ in range(n):
            o = skip_val(b, o, et)
        return o
    raise SystemExit(f"bad type {t}")


def enc_str(s):
    return struct.pack("<Q", len(s)) + s


def main():
    src, dst, *sets = sys.argv[1:]
    b = open(src, "rb").read()
    assert b[:4] == b"GGUF"
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", b, 4)
    o = 24
    kvs = []  # (key, raw bytes of type+value)
    arch = b""
    alignment = 32
    for _ in range(n_kv):
        key, o = read_str(b, o)
        (t,) = struct.unpack_from("<I", b, o)
        end = skip_val(b, o + 4, t)
        kvs.append((key, b[o:end]))
        if key == b"general.architecture":
            arch = read_str(b, o + 4)[0]
        if key == b"general.alignment":
            (alignment,) = struct.unpack_from("<I", b, o + 4)
        o = end
    tinfo_start = o
    for _ in range(n_tensors):
        _, o = read_str(b, o)
        (nd,) = struct.unpack_from("<I", b, o)
        o += 4 + 8 * nd + 4 + 8
    tinfos = b[tinfo_start:o]
    data_start = (o + alignment - 1) // alignment * alignment
    data = b[data_start:]

    for s in sets:
        key, spec = s.split("=", 1)
        kind, value = spec.split(":", 1)
        if kind == "arch-str":
            key = key.replace("{arch}", arch.decode())
            kind = "str"
        if kind == "str":
            raw = struct.pack("<I", 8) + enc_str(value.encode())
        elif kind == "u32":
            raw = struct.pack("<II", 4, int(value))
        else:
            raise SystemExit(f"bad kind {kind}")
        kb = key.encode()
        kvs = [(k, v) for (k, v) in kvs if k != kb] + [(kb, raw)]

    out = bytearray(b"GGUF" + struct.pack("<IQQ", version, n_tensors, len(kvs)))
    for k, v in kvs:
        out += enc_str(k) + v
    out += tinfos
    out += b"\0" * ((-len(out)) % alignment)
    out += data
    open(dst, "wb").write(out)


if __name__ == "__main__":
    main()
