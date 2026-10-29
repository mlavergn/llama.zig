//! Code points, UTF-8, and the pre-tokenizer's regex split.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/src/unicode.cpp`
//! - `llama.cpp/src/unicode.h`
//! - `llama.cpp/src/unicode-data.h`
//!
//! All at v0.4.1 (`b29c606e2`). The tables of `llama.cpp/src/unicode-data.cpp`
//! are generated into `unicode_data.zig` by `scripts/gen-unicode-data`.
//!
//! # Contract
//!
//! Nothing here is exported. `llama-vocab.cpp` is the only caller of these
//! functions, all of them C++-linkage, so this file is gated in-process by
//! `scripts/unicode-diff` against the real `unicode.cpp` and swaps in with
//! the vocabulary.
//!
//! # Exceptions
//!
//! The C++ throws in four places, and each becomes an error here:
//! `unicode_cpt_from_utf8` on a malformed sequence and `unicode_cpt_to_utf8`
//! above U+10FFFF (`std::invalid_argument`), `unicode_utf8_to_byte` on a
//! string no byte maps to (`std::out_of_range`, from `map.at`), and
//! `unicode_regex_split` on a regex it cannot compile or run
//! (`std::runtime_error`). `unicode_cpts_from_utf8` catches the first and
//! substitutes U+FFFD, as there.
//!
//! The C's `assert`s are compiled out in the reference build and are not
//! reproduced.

const std = @import("std");
const impl = @import("impl.zig");
const regex = @import("regex.zig");
const data = @import("unicode_data.zig");

const allocator = impl.allocator;

fn oom() noreturn {
    impl.abort("unicode", 0, "out of memory");
}

/// Ports `MAX_CODEPOINTS` (unicode-data.h:14 @b29c606e2).
pub const max_codepoints: u32 = 0x110000;

/// Ports `unicode_cpt_flags` (src/unicode.h:8 @b29c606e2).
///
/// The C++ is a bit-field over a `uint16_t`, read and written through a
/// `reinterpret_cast` on little-endian targets; a packed struct puts its
/// first field in the low bit the same way, so `asUint` is the same value.
pub const CptFlags = packed struct(u16) {
    is_undefined: bool = false,
    is_number: bool = false,
    is_letter: bool = false,
    is_separator: bool = false,
    is_accent_mark: bool = false,
    is_punctuation: bool = false,
    is_symbol: bool = false,
    is_control: bool = false,
    is_whitespace: bool = false,
    is_lowercase: bool = false,
    is_uppercase: bool = false,
    is_nfd: bool = false,
    _pad: u4 = 0,

    pub const undefined_: u16 = 0x0001;
    pub const number: u16 = 0x0002;
    pub const letter: u16 = 0x0004;
    pub const separator: u16 = 0x0008;
    pub const accent_mark: u16 = 0x0010;
    pub const punctuation: u16 = 0x0020;
    pub const symbol: u16 = 0x0040;
    pub const control: u16 = 0x0080;
    pub const mask_categories: u16 = 0x00FF;

    /// The constructor from a `uint16_t`.
    pub fn fromUint(v: u16) CptFlags {
        return @bitCast(v);
    }

    /// `as_uint()`.
    pub fn asUint(self: CptFlags) u16 {
        return @bitCast(self);
    }

    /// `category_flag()`: the category bits alone.
    pub fn categoryFlag(self: CptFlags) u16 {
        return self.asUint() & mask_categories;
    }
};

// -----------------------------------------------------------------------------
// UTF-8

/// Ports `unicode_len_utf8` (src/unicode.cpp:16 @b29c606e2): a sequence's length
/// from its lead byte's high nibble. A continuation byte reads as 1.
///
/// Parameters:
/// - `src`: the lead byte.
///
/// Return: 1 to 4.
pub fn lenUtf8(src: u8) usize {
    const lookup = [_]usize{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    return lookup[src >> 4];
}

/// Ports `unicode_cpt_from_utf8` (src/unicode.cpp:30 @b29c606e2): decodes one
/// sequence and advances `offset` past it.
///
/// Lenient as the C++ is: overlong forms and surrogates decode, and a
/// four-byte lead reaches up to U+1FFFFF. `char` is signed on this target,
/// but every byte is masked before it is shifted, so reading it unsigned
/// gives the same value.
///
/// Parameters:
/// - `utf8`: the text; `offset.*` must be in bounds.
/// - `offset`: where to decode; moved past the sequence on success, left
///   alone on error.
///
/// Return: the code point, or `error.InvalidCharacter` where the C++ throws
/// `std::invalid_argument`.
pub fn cptFromUtf8(utf8: []const u8, offset: *usize) error{InvalidCharacter}!u32 {
    const o = offset.*;
    const b0: u32 = utf8[o];
    if ((b0 & 0x80) == 0) {
        offset.* += 1;
        return b0;
    }
    if ((b0 & 0x40) == 0) return error.InvalidCharacter;
    if ((b0 & 0x20) == 0) {
        if (o + 1 >= utf8.len or (utf8[o + 1] & 0xc0) != 0x80) return error.InvalidCharacter;
        const result = ((b0 & 0x1f) << 6) | (utf8[o + 1] & 0x3f);
        offset.* += 2;
        return result;
    }
    if ((b0 & 0x10) == 0) {
        if (o + 2 >= utf8.len or (utf8[o + 1] & 0xc0) != 0x80 or (utf8[o + 2] & 0xc0) != 0x80) return error.InvalidCharacter;
        const result = ((b0 & 0x0f) << 12) | (@as(u32, utf8[o + 1] & 0x3f) << 6) | (utf8[o + 2] & 0x3f);
        offset.* += 3;
        return result;
    }
    if ((b0 & 0x08) == 0) {
        if (o + 3 >= utf8.len or (utf8[o + 1] & 0xc0) != 0x80 or (utf8[o + 2] & 0xc0) != 0x80 or (utf8[o + 3] & 0xc0) != 0x80) return error.InvalidCharacter;
        const result = ((b0 & 0x07) << 18) | (@as(u32, utf8[o + 1] & 0x3f) << 12) | (@as(u32, utf8[o + 2] & 0x3f) << 6) | (utf8[o + 3] & 0x3f);
        offset.* += 4;
        return result;
    }
    // "failed to convert utf8 to codepoint"
    return error.InvalidCharacter;
}

/// Ports `unicode_cpt_to_utf8` (src/unicode.cpp:1088 @b29c606e2).
///
/// Parameters:
/// - `cpt`: the code point.
/// - `buf`: receives the bytes.
///
/// Return: the encoding, borrowed from `buf`, or `error.InvalidCodepoint`
/// above U+10FFFF, where the C++ throws `std::invalid_argument`.
pub fn cptToUtf8(cpt: u32, buf: *[4]u8) error{InvalidCodepoint}![]const u8 {
    if (cpt <= 0x7f) {
        buf[0] = @intCast(cpt);
        return buf[0..1];
    }
    if (0x80 <= cpt and cpt <= 0x7ff) {
        buf[0] = @intCast(0xc0 | ((cpt >> 6) & 0x1f));
        buf[1] = @intCast(0x80 | (cpt & 0x3f));
        return buf[0..2];
    }
    if (0x800 <= cpt and cpt <= 0xffff) {
        buf[0] = @intCast(0xe0 | ((cpt >> 12) & 0x0f));
        buf[1] = @intCast(0x80 | ((cpt >> 6) & 0x3f));
        buf[2] = @intCast(0x80 | (cpt & 0x3f));
        return buf[0..3];
    }
    if (0x10000 <= cpt and cpt <= 0x10ffff) {
        buf[0] = @intCast(0xf0 | ((cpt >> 18) & 0x07));
        buf[1] = @intCast(0x80 | ((cpt >> 12) & 0x3f));
        buf[2] = @intCast(0x80 | ((cpt >> 6) & 0x3f));
        buf[3] = @intCast(0x80 | (cpt & 0x3f));
        return buf[0..4];
    }
    return error.InvalidCodepoint;
}

/// `unicode_cpt_to_utf8` appended to `out`, the shape every caller uses.
///
/// Return: nothing, or `error.InvalidCodepoint`.
pub fn appendCptUtf8(out: *std.ArrayList(u8), cpt: u32) error{InvalidCodepoint}!void {
    var buf: [4]u8 = undefined;
    out.appendSlice(allocator, try cptToUtf8(cpt, &buf)) catch oom();
}

// `unicode_cpts_to_utf8` (src/unicode.cpp:22 @b29c606e2) is a `static`
// nothing calls, and is not ported.

/// Ports `unicode_cpts_from_utf8` (src/unicode.cpp:1130 @b29c606e2): every
/// sequence decoded, a malformed byte becoming U+FFFD and decoding resuming
/// at the next byte.
///
/// Parameters:
/// - `utf8`: the text.
///
/// Return: the code points, owned by the caller and freed with
/// `impl.allocator`.
pub fn cptsFromUtf8(utf8: []const u8) []u32 {
    var result: std.ArrayList(u32) = .empty;
    result.ensureTotalCapacity(allocator, utf8.len) catch oom();
    var offset: usize = 0;
    while (offset < utf8.len) {
        const cpt = cptFromUtf8(utf8, &offset) catch blk: {
            offset += 1;
            break :blk 0xFFFD;
        };
        result.append(allocator, cpt) catch oom();
    }
    return result.toOwnedSlice(allocator) catch oom();
}

// -----------------------------------------------------------------------------
// Code point properties

/// `wchar_t`'s width: 16 bits on Windows, 32 everywhere else.
const wchar_bits = if (@import("builtin").os.tag == .windows) 16 else 32;

/// `unicode_cpt_flags_array`'s result, built on first use as the C++'s
/// function-local `static` is.
var flags_table: [max_codepoints]CptFlags = undefined;
var flags_ready: std.atomic.Value(bool) = .init(false);
var flags_mutex: impl.CxxMutex = .{};

/// Ports `unicode_cpt_flags_array` (src/unicode.cpp:116 @b29c606e2): each
/// range's flags, then the whitespace bit, then the case and NFD bits on
/// every code point some mapping *produces*.
fn buildFlagsTable() void {
    @memset(&flags_table, CptFlags.fromUint(CptFlags.undefined_));
    var i: usize = 1;
    while (i < data.ranges_flags.len) : (i += 1) {
        const range_ini = data.ranges_flags[i - 1];
        const range_end = data.ranges_flags[i];
        var cpt = range_ini[0];
        while (cpt < range_end[0]) : (cpt += 1) {
            flags_table[cpt] = CptFlags.fromUint(range_ini[1]);
        }
    }
    for (data.set_whitespace) |cpt| flags_table[cpt].is_whitespace = true;
    for (data.map_lowercase) |p| flags_table[p[1]].is_lowercase = true;
    for (data.map_uppercase) |p| flags_table[p[1]].is_uppercase = true;
    for (data.ranges_nfd) |range| flags_table[range.nfd].is_nfd = true;
}

/// Ports `unicode_cpt_flags_from_cpt` (src/unicode.cpp:1147 @b29c606e2).
///
/// Parameters:
/// - `cpt`: any value.
///
/// Return: the flags; `UNDEFINED` at and above `max_codepoints`.
pub fn cptFlagsFromCpt(cpt: u32) CptFlags {
    if (!flags_ready.load(.acquire)) {
        flags_mutex.lock();
        defer flags_mutex.unlock();
        if (!flags_ready.load(.monotonic)) {
            buildFlagsTable();
            flags_ready.store(true, .release);
        }
    }
    return if (cpt < max_codepoints) flags_table[cpt] else CptFlags.fromUint(CptFlags.undefined_);
}

/// Ports `unicode_cpt_flags_from_utf8` (src/unicode.cpp:1153 @b29c606e2): the
/// flags of the first code point.
///
/// Return: the flags, `UNDEFINED` for an empty string, or
/// `error.InvalidCharacter` where the C++ throws.
pub fn cptFlagsFromUtf8(utf8: []const u8) error{InvalidCharacter}!CptFlags {
    if (utf8.len == 0) return CptFlags.fromUint(CptFlags.undefined_);
    var offset: usize = 0;
    return cptFlagsFromCpt(try cptFromUtf8(utf8, &offset));
}

/// Ports `unicode_tolower` (src/unicode.cpp:1172 @b29c606e2): a binary search
/// of the lowercase map.
///
/// Return: the mapped code point, or `cpt` when it has none.
pub fn tolower(cpt: u32) u32 {
    var lo: usize = 0;
    var hi: usize = data.map_lowercase.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (data.map_lowercase[mid][0] < cpt) lo = mid + 1 else hi = mid;
    }
    if (lo != data.map_lowercase.len and data.map_lowercase[lo][0] == cpt) return data.map_lowercase[lo][1];
    return cpt;
}

/// Ports `unicode_cpt_is_han` (src/unicode.cpp:1184 @b29c606e2).
pub fn cptIsHan(cpt: u32) bool {
    if (cpt >= 0x4E00 and cpt <= 0x9FFF) return true;
    if (cpt >= 0x3400 and cpt <= 0x4DBF) return true;
    if (cpt >= 0x20000 and cpt <= 0x2A6DF) return true;
    if (cpt >= 0x2A700 and cpt <= 0x2B73F) return true;
    if (cpt >= 0x2B740 and cpt <= 0x2B81F) return true;
    if (cpt >= 0x2B820 and cpt <= 0x2CEAF) return true;
    if (cpt >= 0x2CEB0 and cpt <= 0x2EBEF) return true;
    if (cpt >= 0xF900 and cpt <= 0xFAFF) return true;
    if (cpt >= 0x2F800 and cpt <= 0x2FA1F) return true;
    return false;
}

/// Ports `unicode_cpts_normalize_nfd` (src/unicode.cpp:1117 @b29c606e2): each
/// code point through the last NFD range starting at or below it.
///
/// Parameters:
/// - `cpts`: the code points.
///
/// Return: the mapped code points, the same length, owned by the caller.
pub fn cptsNormalizeNfd(cpts: []const u32) []u32 {
    const result = allocator.alloc(u32, cpts.len) catch oom();
    for (cpts, result) |cpt, *r| {
        // `std::upper_bound` on `first`, minus one. The first range starts
        // at 0, so the step back is always in bounds.
        var lo: usize = 0;
        var hi: usize = data.ranges_nfd.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (cpt < data.ranges_nfd[mid].first) hi = mid else lo = mid + 1;
        }
        const it = data.ranges_nfd[lo - 1];
        r.* = if (it.first <= cpt and cpt <= it.last) it.nfd else cpt;
    }
    return result;
}

// -----------------------------------------------------------------------------
// The GPT-2 byte map

/// The code point each byte stands for: printable Latin-1 as itself, the
/// rest from U+0100 up in byte order.
const byte_cpt: [256]u32 = blk: {
    var t: [256]u32 = undefined;
    var n: u32 = 0;
    for (0..256) |ch| {
        if ((ch >= 0x21 and ch <= 0x7E) or (ch >= 0xA1 and ch <= 0xAC) or (ch >= 0xAE and ch <= 0xFF)) {
            t[ch] = ch;
        } else {
            t[ch] = 256 + n;
            n += 1;
        }
    }
    break :blk t;
};

/// `byte_cpt` encoded: every value is at most U+0143, two bytes at most.
const byte_utf8: [256][]const u8 = blk: {
    var t: [256][]const u8 = undefined;
    for (0..256) |ch| {
        const cpt = byte_cpt[ch];
        t[ch] = if (cpt < 0x80) &[_]u8{@intCast(cpt)} else &[_]u8{ @intCast(0xc0 | (cpt >> 6)), @intCast(0x80 | (cpt & 0x3f)) };
    }
    break :blk t;
};

/// The inverse of `byte_cpt`.
const cpt_byte: [256 + 68]?u8 = blk: {
    var t: [256 + 68]?u8 = @splat(null);
    for (0..256) |ch| t[byte_cpt[ch]] = ch;
    break :blk t;
};

/// Ports `unicode_byte_to_utf8` (src/unicode.cpp:1162 @b29c606e2), with
/// `unicode_byte_to_utf8_map` (src/unicode.cpp:148 @b29c606e2) built at
/// compile time.
///
/// Return: the byte's UTF-8 stand-in, static.
pub fn byteToUtf8(byte: u8) []const u8 {
    return byte_utf8[byte];
}

/// Ports `unicode_utf8_to_byte` (src/unicode.cpp:1167 @b29c606e2), with
/// `unicode_utf8_to_byte_map` (src/unicode.cpp:172 @b29c606e2) as the inverse
/// of the byte map. The C++'s keys are exactly the 256 canonical encodings
/// `byteToUtf8` returns, so a string is a key iff it is one of them.
///
/// Return: the byte, or `error.OutOfRange` where `map.at` throws.
pub fn utf8ToByte(utf8: []const u8) error{OutOfRange}!u8 {
    var cpt: u32 = undefined;
    if (utf8.len == 1 and utf8[0] < 0x80) {
        cpt = utf8[0];
    } else if (utf8.len == 2 and (utf8[0] & 0xe0) == 0xc0 and (utf8[1] & 0xc0) == 0x80) {
        cpt = (@as(u32, utf8[0] & 0x1f) << 6) | (utf8[1] & 0x3f);
    } else return error.OutOfRange;
    if (cpt >= cpt_byte.len) return error.OutOfRange;
    const byte = cpt_byte[cpt] orelse return error.OutOfRange;
    if (!std.mem.eql(u8, byte_utf8[byte], utf8)) return error.OutOfRange;
    return byte;
}

// -----------------------------------------------------------------------------
// The custom splitters

const out_of_range: u32 = 0xFFFFFFFF;

/// The lambdas every custom splitter shares: one `offsets` span of
/// `cpts`, read with bounds that answer `OUT_OF_RANGE` and empty flags, and
/// the running end of the last token.
const Span = struct {
    cpts: []const u32,
    ini: usize,
    end: usize,
    prev_end: usize,
    out: *std.ArrayList(usize),

    /// `_get_cpt`.
    fn cpt(self: *const Span, pos: usize) u32 {
        return if (self.ini <= pos and pos < self.end) self.cpts[pos] else out_of_range;
    }

    /// `_get_flags`.
    fn flags(self: *const Span, pos: usize) CptFlags {
        return if (self.ini <= pos and pos < self.end) cptFlagsFromCpt(self.cpts[pos]) else .{};
    }

    /// `_add_token`: the token ending at `end`, if it is not empty.
    fn add(self: *Span, end: usize) usize {
        const len = end -% self.prev_end;
        if (len > 0) self.out.append(allocator, len) catch oom();
        self.prev_end = end;
        return len;
    }
};

fn isContractionPair(a: u32, b: u32) bool {
    return (a == 'r' and b == 'e') or (a == 'v' and b == 'e') or (a == 'l' and b == 'l');
}

fn isContractionOne(a: u32) bool {
    return a == 's' or a == 't' or a == 'm' or a == 'd';
}

/// Ports `unicode_regex_split_custom_gpt2` (src/unicode.cpp:215 @b29c606e2):
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
fn splitCustomGpt2(cpts: []const u32, offsets: []const usize, out: *std.ArrayList(usize)) void {
    var start: usize = 0;
    for (offsets) |offset| {
        var sp: Span = .{ .cpts = cpts, .ini = start, .end = start + offset, .prev_end = start, .out = out };
        start = sp.end;

        var pos = sp.ini;
        while (pos < sp.end) {
            const cpt = sp.cpt(pos);
            const flags = sp.flags(pos);

            // regex: 's|'t|'re|'ve|'m|'ll|'d
            if (cpt == '\'' and pos + 1 < sp.end) {
                const cpt_next = sp.cpt(pos + 1);
                if (isContractionOne(cpt_next)) {
                    pos += sp.add(pos + 2);
                    continue;
                }
                if (pos + 2 < sp.end) {
                    const cpt_next_next = sp.cpt(pos + 2);
                    if (isContractionPair(cpt_next, cpt_next_next)) {
                        pos += sp.add(pos + 3);
                        continue;
                    }
                }
            }

            var flags2 = if (cpt == ' ') sp.flags(pos + 1) else flags;
            // regex: <space>?\p{L}+
            if (flags2.is_letter) {
                pos += @intFromBool(cpt == ' ');
                while (flags2.is_letter) {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                _ = sp.add(pos);
                continue;
            }
            // regex: <space>?\p{N}+
            if (flags2.is_number) {
                pos += @intFromBool(cpt == ' ');
                while (flags2.is_number) {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                _ = sp.add(pos);
                continue;
            }
            // regex: <space>?[^\s\p{L}\p{N}]+
            if (!(flags2.is_whitespace or flags2.is_letter or flags2.is_number) and flags2.asUint() != 0) {
                pos += @intFromBool(cpt == ' ');
                while (!(flags2.is_whitespace or flags2.is_letter or flags2.is_number) and flags2.asUint() != 0) {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                _ = sp.add(pos);
                continue;
            }

            var num_whitespaces: usize = 0;
            while (sp.flags(pos + num_whitespaces).is_whitespace) num_whitespaces += 1;

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 and sp.cpt(pos + num_whitespaces) != out_of_range) {
                pos += num_whitespaces - 1;
                _ = sp.add(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _ = sp.add(pos);
                continue;
            }

            // no matches
            pos += 1;
            _ = sp.add(pos);
        }
    }
}

/// Which variant of the llama3 family `splitCustomLlama3Like` runs.
const Llama3Kind = enum {
    /// `unicode_regex_split_custom_llama3` (src/unicode.cpp:333 @b29c606e2):
    /// `\p{N}{1,3}`.
    llama3,
    /// `unicode_regex_split_custom_qwen2` (src/unicode.cpp:474 @b29c606e2):
    /// `\p{N}`.
    qwen2,
    /// `unicode_regex_split_custom_qwen35` (src/unicode.cpp:610 @b29c606e2):
    /// `\p{N}`, and letter runs take combining marks.
    qwen35,
};

/// The three splitters are one function in the C++ three times over,
/// differing in the digit run and in whether `\p{M}` counts as a letter.
/// Each line below is the same line in all three, apart from the switches
/// on `kind`.
///
/// One quirk is carried over: the punctuation branch tests
/// `flags.as_uint()`, the flags at `pos`, where the gpt2 splitter tests
/// `flags2`.
fn splitCustomLlama3Like(comptime kind: Llama3Kind, cpts: []const u32, offsets: []const usize, out: *std.ArrayList(usize)) void {
    const marks = kind == .qwen35;
    var start: usize = 0;
    for (offsets) |offset| {
        var sp: Span = .{ .cpts = cpts, .ini = start, .end = start + offset, .prev_end = start, .out = out };
        start = sp.end;

        var pos = sp.ini;
        while (pos < sp.end) {
            const cpt = sp.cpt(pos);
            const flags = sp.flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' and pos + 1 < sp.end) {
                const cpt_next = tolower(sp.cpt(pos + 1));
                if (isContractionOne(cpt_next)) {
                    pos += sp.add(pos + 2);
                    continue;
                }
                if (pos + 2 < sp.end) {
                    const cpt_next_next = tolower(sp.cpt(pos + 2));
                    if (isContractionPair(cpt_next, cpt_next_next)) {
                        pos += sp.add(pos + 3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+, or [\p{L}\p{M}]+ for qwen35
            if (!(cpt == '\r' or cpt == '\n' or flags.is_number)) {
                const starts = if (marks)
                    flags.is_letter or flags.is_accent_mark or sp.flags(pos + 1).is_accent_mark or sp.flags(pos + 1).is_letter
                else
                    flags.is_letter or sp.flags(pos + 1).is_letter;
                if (starts) {
                    pos += 1;
                    while (sp.flags(pos).is_letter or (marks and sp.flags(pos).is_accent_mark)) pos += 1;
                    _ = sp.add(pos);
                    continue;
                }
            }

            if (kind == .llama3) {
                // regex: \p{N}{1,3}
                if (flags.is_number) {
                    var ini = pos;
                    while (sp.flags(pos).is_number) {
                        pos += 1;
                        if (pos - ini >= 3) {
                            _ = sp.add(pos);
                            ini = pos;
                        }
                    }
                    _ = sp.add(pos);
                    continue;
                }
            } else {
                // regex: \p{N}
                if (flags.is_number) {
                    pos += 1;
                    _ = sp.add(pos);
                    continue;
                }
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*, with \p{M} for qwen35
            var flags2 = if (cpt == ' ') sp.flags(pos + 1) else flags;
            if (!(flags2.is_whitespace or flags2.is_letter or (marks and flags2.is_accent_mark) or flags2.is_number) and flags.asUint() != 0) {
                pos += @intFromBool(cpt == ' ');
                while (!(flags2.is_whitespace or flags2.is_letter or (marks and flags2.is_accent_mark) or flags2.is_number) and flags2.asUint() != 0) {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                var cpt2 = sp.cpt(pos);
                while (cpt2 == '\r' or cpt2 == '\n') {
                    pos += 1;
                    cpt2 = sp.cpt(pos);
                }
                _ = sp.add(pos);
                continue;
            }

            var num_whitespaces: usize = 0;
            var last_end_r_or_n: usize = 0;
            while (sp.flags(pos + num_whitespaces).is_whitespace) {
                const cpt2 = sp.cpt(pos + num_whitespaces);
                if (cpt2 == '\r' or cpt2 == '\n') last_end_r_or_n = pos + num_whitespaces + 1;
                num_whitespaces += 1;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _ = sp.add(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 and sp.cpt(pos + num_whitespaces) != out_of_range) {
                pos += num_whitespaces - 1;
                _ = sp.add(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _ = sp.add(pos);
                continue;
            }

            // no matches
            pos += 1;
            _ = sp.add(pos);
        }
    }
}

/// Ports `unicode_regex_split_custom_kimi_k2` (src/unicode.cpp:777 @b29c606e2):
/// all of Kimi K2's patterns at once, Han runs first.
fn splitCustomKimiK2(cpts: []const u32, offsets: []const usize, out: *std.ArrayList(usize)) void {
    var start: usize = 0;
    for (offsets) |offset| {
        var sp: Span = .{ .cpts = cpts, .ini = start, .end = start + offset, .prev_end = start, .out = out };
        start = sp.end;

        var pos = sp.ini;
        while (pos < sp.end) {
            const cpt = sp.cpt(pos);
            const flags = sp.flags(pos);

            // Pattern 1: [\p{Han}]+
            if (cptIsHan(cpt)) {
                while (cptIsHan(sp.cpt(pos))) pos += 1;
                _ = sp.add(pos);
                continue;
            }

            // Patterns 2 and 3: letter words excluding Han, with an
            // optional leading character and an optional contraction.
            const is_letter_pattern = (flags.is_letter and !cptIsHan(cpt)) or
                (!(cpt == '\r' or cpt == '\n' or flags.is_letter or flags.is_number) and
                    sp.flags(pos + 1).is_letter and !cptIsHan(sp.cpt(pos + 1)));

            if (is_letter_pattern) {
                var has_leading_char = false;
                if (!(cpt == '\r' or cpt == '\n' or flags.is_letter or flags.is_number)) {
                    has_leading_char = true;
                    pos += 1;
                }

                var has_letters = false;
                while (sp.flags(pos).is_letter and !cptIsHan(sp.cpt(pos))) {
                    has_letters = true;
                    pos += 1;
                }

                if (has_letters or (!has_leading_char and sp.flags(pos).is_letter and !cptIsHan(sp.cpt(pos)))) {
                    if (!has_letters) pos += 1;

                    while (sp.flags(pos).is_letter and !cptIsHan(sp.cpt(pos))) pos += 1;

                    // (?:'s|'t|'re|'ve|'m|'ll|'d)
                    if (sp.cpt(pos) == '\'' and pos + 1 < sp.end) {
                        const cpt_next = tolower(sp.cpt(pos + 1));
                        if (isContractionOne(cpt_next)) {
                            pos += 2;
                        } else if (pos + 2 < sp.end) {
                            const cpt_next_next = tolower(sp.cpt(pos + 2));
                            if (isContractionPair(cpt_next, cpt_next_next)) pos += 3;
                        }
                    }

                    _ = sp.add(pos);
                    continue;
                } else if (has_leading_char) {
                    pos -= 1;
                }
            }

            // Pattern 4: \p{N}{1,3}
            if (flags.is_number) {
                var ini = pos;
                while (sp.flags(pos).is_number) {
                    pos += 1;
                    if (pos - ini >= 3) {
                        _ = sp.add(pos);
                        ini = pos;
                    }
                }
                _ = sp.add(pos);
                continue;
            }

            // Pattern 5:  ?[^\s\p{L}\p{N}]+[\r\n]*
            var flags2 = if (cpt == ' ') sp.flags(pos + 1) else flags;
            if (!(flags2.is_whitespace or flags2.is_letter or flags2.is_number) and flags2.asUint() != 0) {
                pos += @intFromBool(cpt == ' ');
                while (!(flags2.is_whitespace or flags2.is_letter or flags2.is_number) and flags2.asUint() != 0) {
                    pos += 1;
                    flags2 = sp.flags(pos);
                }
                var cpt2 = sp.cpt(pos);
                while (cpt2 == '\r' or cpt2 == '\n') {
                    pos += 1;
                    cpt2 = sp.cpt(pos);
                }
                _ = sp.add(pos);
                continue;
            }

            var num_whitespaces: usize = 0;
            var last_end_r_or_n: usize = 0;
            while (sp.flags(pos + num_whitespaces).is_whitespace) {
                const cpt2 = sp.cpt(pos + num_whitespaces);
                if (cpt2 == '\r' or cpt2 == '\n') last_end_r_or_n = pos + num_whitespaces + 1;
                num_whitespaces += 1;
            }

            // Pattern 6: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _ = sp.add(pos);
                continue;
            }

            // Pattern 7: \s+(?!\S)
            if (num_whitespaces > 1 and sp.cpt(pos + num_whitespaces) != out_of_range) {
                pos += num_whitespaces - 1;
                _ = sp.add(pos);
                continue;
            }

            // Pattern 8: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _ = sp.add(pos);
                continue;
            }

            // No matches: consume one character.
            pos += 1;
            _ = sp.add(pos);
        }
    }
}

/// Ports `unicode_regex_split_custom_afmoe` (src/unicode.cpp:948 @b29c606e2):
/// digit runs split into groups of three from the right, everything else
/// left whole. A digit run's first group also takes any non-digits since
/// the last token, as there.
fn splitCustomAfmoe(cpts: []const u32, offsets: []const usize, out: *std.ArrayList(usize)) void {
    var start: usize = 0;
    for (offsets) |offset| {
        var sp: Span = .{ .cpts = cpts, .ini = start, .end = start + offset, .prev_end = start, .out = out };
        start = sp.end;

        var pos = sp.ini;
        while (pos < sp.end) {
            const flags = sp.flags(pos);

            if (flags.is_number) {
                const digit_start = pos;
                var digit_count: usize = 0;

                while (sp.flags(pos).is_number and pos < sp.end) {
                    digit_count += 1;
                    pos += 1;
                }

                const remainder = digit_count % 3;
                var current = digit_start;

                if (remainder > 0) {
                    _ = sp.add(current + remainder);
                    current += remainder;
                }

                while (current < digit_start + digit_count) {
                    _ = sp.add(current + 3);
                    current += 3;
                }
                continue;
            }

            pos += 1;
        }

        if (sp.prev_end < sp.end) _ = sp.add(sp.end);
    }
}

/// Ports `unicode_regex_split_custom_newlines` (src/unicode.cpp:1023 @b29c606e2):
/// `[^\n]+|[\n]+`.
fn splitCustomNewlines(cpts: []const u32, offsets: []const usize, out: *std.ArrayList(usize)) void {
    var start: usize = 0;
    for (offsets) |offset| {
        const offset_ini = start;
        const offset_end = start + offset;
        start = offset_end;

        var pos = offset_ini;
        while (pos < offset_end) {
            const is_newline = cpts[pos] == '\n';
            const run_start = pos;
            while (pos < offset_end and (cpts[pos] == '\n') == is_newline) pos += 1;
            out.append(allocator, pos - run_start) catch oom();
        }
    }
}

/// Ports `unicode_regex_split_custom` (src/unicode.cpp:1050 @b29c606e2): the
/// hand-written splitter for a pattern, chosen by exact string.
///
/// Return: the offsets, empty when no splitter matches the pattern -- or
/// when one does and the text is empty, which sends the C++ on to
/// `std::regex` too.
fn splitCustom(cpts: []const u32, regex_expr: []const u8, offsets: []const usize, out: *std.ArrayList(usize)) void {
    const eql = std.mem.eql;
    if (eql(u8, regex_expr, "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)")) {
        splitCustomGpt2(cpts, offsets, out);
    } else if (eql(u8, regex_expr, "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") or
        eql(u8, regex_expr, "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"))
    {
        splitCustomLlama3Like(.llama3, cpts, offsets, out);
    } else if (eql(u8, regex_expr, "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")) {
        splitCustomLlama3Like(.qwen2, cpts, offsets, out);
    } else if (eql(u8, regex_expr, "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")) {
        splitCustomLlama3Like(.qwen35, cpts, offsets, out);
    } else if (eql(u8, regex_expr, "\\p{Han}+")) {
        splitCustomKimiK2(cpts, offsets, out);
    } else if (eql(u8, regex_expr, "\\p{AFMoE_digits}")) {
        splitCustomAfmoe(cpts, offsets, out);
    } else if (eql(u8, regex_expr, "[^\\n]+|[\\n]+")) {
        splitCustomNewlines(cpts, offsets, out);
    } else if (eql(u8, regex_expr, "\\d{1,3}(?=(?:\\d{3})*\\b)")) {
        // tiny_aya's digit grouping, handled by the AFMoE splitter.
        splitCustomAfmoe(cpts, offsets, out);
    }
}

// -----------------------------------------------------------------------------
// The std::regex fallback

/// Ports `unicode_regex_split_stl` (src/unicode.cpp:739 @b29c606e2): each span
/// of `offsets` cut at every match of `pattern`, compiled with
/// `optimize | nosubs`, the text between matches kept as its own piece.
/// An empty match yields an empty piece, as there.
fn splitStl(comptime CharT: type, text: []const CharT, pattern: []const CharT, offsets: []const usize, out: *std.ArrayList(usize)) regex.Error!void {
    const Re = regex.BasicRegex(CharT);
    var expr = try Re.compileWith(pattern, .{ .nosubs = true });
    defer expr.deinit();
    out.ensureTotalCapacity(allocator, offsets.len) catch oom();
    var start: usize = 0;
    for (offsets) |offset| {
        var it = try Re.Iterator.init(&expr, text, start, start + offset);
        defer it.deinit();

        var start_idx: i64 = 0;
        while (it.valid()) {
            const position: i64 = @intCast(it.position());
            const length: i64 = @intCast(it.length());
            if (position > start_idx) out.append(allocator, @intCast(position - start_idx)) catch oom();
            out.append(allocator, @intCast(length)) catch oom();
            start_idx = position + length;
            try it.next();
        }

        if (start_idx < @as(i64, @intCast(offset))) out.append(allocator, offset - @as(usize, @intCast(start_idx))) catch oom();
        start += offset;
    }
}

/// The `\p{..}` names `unicode_regex_split` collapses, and the category
/// each stands for: `k_ucat_enum`.
const ucat_enum = [_]struct { []const u8, u16 }{
    .{ "\\p{N}", CptFlags.number },
    .{ "\\p{L}", CptFlags.letter },
    .{ "\\p{P}", CptFlags.punctuation },
    .{ "\\p{M}", CptFlags.accent_mark },
    .{ "\\p{S}", CptFlags.symbol },
    .{ "\\p{Lu}", CptFlags.letter },
    .{ "\\p{Ll}", CptFlags.letter },
    .{ "\\p{Lt}", CptFlags.letter },
    .{ "\\p{Lm}", CptFlags.letter },
    .{ "\\p{Lo}", CptFlags.letter },
};

/// `k_ucat_cpt`: the byte a category's code points collapse to.
fn ucatCpt(cat: u16) ?u8 {
    return switch (cat) {
        CptFlags.number => 0xD1,
        CptFlags.letter => 0xD2,
        CptFlags.punctuation => 0xD3,
        CptFlags.accent_mark => 0xD4,
        CptFlags.symbol => 0xD5,
        else => null,
    };
}

/// `k_ucat_map`: the ASCII members of each category, as bracket contents.
fn ucatMap(cat: u16) []const u8 {
    return switch (cat) {
        CptFlags.number => "\x30-\x39", // 0-9
        CptFlags.letter => "\x41-\x5A\x61-\x7A", // A-Za-z
        CptFlags.punctuation => "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D", // !-#%-*,-/:-;?-@\[-\]_\{\}
        CptFlags.accent_mark => "", // no sub-128 code points
        CptFlags.symbol => "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C\\\x7E", // $+<=>^`|~
        else => unreachable,
    };
}

fn ucatLookup(name: []const u8) ?u16 {
    for (ucat_enum) |e| {
        if (std.mem.eql(u8, e[0], name)) return e[1];
    }
    return null;
}

fn usesCategory(regex_expr: []const u8) bool {
    for (ucat_enum) |e| {
        if (std.mem.find(u8, regex_expr, e[0]) != null) return true;
    }
    return false;
}

/// The pieces `regexSplit` returns: one buffer, cut at `ends`.
pub const Words = struct {
    buf: std.ArrayList(u8) = .empty,
    ends: std.ArrayList(usize) = .empty,

    /// Return: the number of pieces.
    pub fn len(self: *const Words) usize {
        return self.ends.items.len;
    }

    /// Return: piece `i`, borrowed until `deinit`.
    pub fn get(self: *const Words, i: usize) []const u8 {
        const b = if (i == 0) 0 else self.ends.items[i - 1];
        return self.buf.items[b..self.ends.items[i]];
    }

    /// Return: nothing.
    pub fn deinit(self: *Words) void {
        self.buf.deinit(allocator);
        self.ends.deinit(allocator);
    }

    fn finish(self: *Words) void {
        self.ends.append(allocator, self.buf.items.len) catch oom();
    }
};

/// What `unicode_regex_split` throws.
pub const SplitError = error{
    /// `std::runtime_error("Failed to process regex")`, after a
    /// `std::regex_error` from compiling or running a pattern.
    RegexFailed,
    /// `std::runtime_error("Regex includes both unicode categories and
    /// non-ASCII characters - not supported")`.
    CategoryWithNonAscii,
    /// `std::invalid_argument("invalid codepoint")`, from
    /// `unicode_cpt_to_utf8` on a decoded value above U+10FFFF.
    InvalidCodepoint,
};

/// Ports `unicode_byte_encoding_process` (src/unicode.cpp:196 @b29c606e2):
/// every byte of every piece replaced by its GPT-2 stand-in. The C++ first
/// re-decodes and re-encodes each piece; the pieces are already the
/// encoding of their code points, so that round trip is the identity, but
/// it is kept for an exact port.
fn byteEncodingProcess(words: *const Words) Words {
    var out: Words = .{};
    var text_utf: std.ArrayList(u8) = .empty;
    defer text_utf.deinit(allocator);
    for (0..words.len()) |i| {
        const word = words.get(i);
        text_utf.clearRetainingCapacity();
        const utf_word = cptsFromUtf8(word);
        defer allocator.free(utf_word);
        for (utf_word) |cpt| appendCptUtf8(&text_utf, cpt) catch unreachable; // decoded from valid UTF-8
        for (text_utf.items) |ch| out.buf.appendSlice(allocator, byteToUtf8(ch)) catch oom();
        out.finish();
    }
    return out;
}

/// Ports `unicode_regex_split` (src/unicode.cpp:1216 @b29c606e2): the text cut
/// by each pattern in turn, every pattern refining the previous pieces.
///
/// A pattern with a hand-written splitter uses it. Otherwise one that names
/// a unicode category runs as a `std::regex` over a collapsed copy of the
/// text, every code point at or above 128 replaced by one byte for its
/// category and each `\p{..}` by a bracket of that byte plus the category's
/// ASCII members; one that names none runs as a `std::wregex` over the code
/// points, non-ASCII whitespace replaced by U+000B so `\s` sees it.
///
/// Parameters:
/// - `text`: UTF-8, not necessarily valid.
/// - `regex_exprs`: the patterns, in order.
/// - `byte_encode`: whether to apply the GPT-2 byte map to the pieces.
///
/// Return: the pieces, owned by the caller, or the error the C++ throws.
pub fn regexSplit(text: []const u8, regex_exprs: []const []const u8, byte_encode: bool) SplitError!Words {
    var need_collapse = false;
    for (regex_exprs) |e| {
        if (usesCategory(e)) need_collapse = true;
    }

    const cpts = cptsFromUtf8(text);
    defer allocator.free(cpts);

    // The "collapsed" text: one byte per code point.
    var text_collapsed: []u8 = &.{};
    defer if (need_collapse) allocator.free(text_collapsed);
    if (need_collapse) {
        text_collapsed = allocator.alloc(u8, cpts.len) catch oom();
        for (cpts, text_collapsed) |cpt, *tc| {
            if (cpt < 128) {
                tc.* = @intCast(cpt);
                continue;
            }
            const flags = cptFlagsFromCpt(cpt);
            if (flags.is_whitespace) {
                // `std::regex`'s `\s` does not match U+0085, so the
                // fallback is a vertical tab.
                tc.* = 0x0B;
            } else if (ucatCpt(flags.categoryFlag())) |b| {
                tc.* = b;
            } else {
                tc.* = 0xD0;
            }
        }
    }

    var bpe_offsets: std.ArrayList(usize) = .empty;
    defer bpe_offsets.deinit(allocator);
    bpe_offsets.append(allocator, cpts.len) catch oom();
    var tmp: std.ArrayList(usize) = .empty;
    defer tmp.deinit(allocator);

    for (regex_exprs) |regex_expr| {
        tmp.clearRetainingCapacity();
        splitCustom(cpts, regex_expr, bpe_offsets.items, &tmp);
        if (tmp.items.len != 0) {
            std.mem.swap(std.ArrayList(usize), &bpe_offsets, &tmp);
            continue;
        }

        const use_collapsed = usesCategory(regex_expr);
        const cpts_regex = cptsFromUtf8(regex_expr);
        defer allocator.free(cpts_regex);

        const r: regex.Error!void = blk: {
            if (use_collapsed) {
                for (cpts_regex) |cr| {
                    if (cr >= 128) return error.CategoryWithNonAscii;
                }

                var collapsed: std.ArrayList(u8) = .empty;
                defer collapsed.deinit(allocator);
                // Inside `[]`, where a category contributes its members
                // without a bracket of its own.
                var inside = false;
                var i: usize = 0;
                while (i < regex_expr.len) : (i += 1) {
                    if (regex_expr[i] == '[' and (i == 0 or regex_expr[i - 1] != '\\')) {
                        collapsed.append(allocator, '[') catch oom();
                        inside = true;
                        continue;
                    }
                    if (inside and regex_expr[i] == ']' and regex_expr[i - 1] != '\\') {
                        collapsed.append(allocator, ']') catch oom();
                        inside = false;
                        continue;
                    }
                    if (regex_expr[i] == '\\' and i + 3 < regex_expr.len and regex_expr[i + 1] == 'p' and regex_expr[i + 2] == '{') {
                        if (std.mem.findScalarPos(u8, regex_expr, i + 3, '}')) |closing_brace| {
                            if (closing_brace <= i + 10) {
                                if (ucatLookup(regex_expr[i .. closing_brace + 1])) |cat| {
                                    if (!inside) collapsed.append(allocator, '[') catch oom();
                                    collapsed.append(allocator, ucatCpt(cat).?) catch oom();
                                    collapsed.appendSlice(allocator, ucatMap(cat)) catch oom();
                                    if (!inside) collapsed.append(allocator, ']') catch oom();
                                    i = closing_brace;
                                    continue;
                                }
                            }
                        }
                    }
                    collapsed.append(allocator, regex_expr[i]) catch oom();
                }

                tmp.clearRetainingCapacity();
                break :blk splitStl(u8, text_collapsed, collapsed.items, bpe_offsets.items, &tmp);
            } else {
                // No category: a `std::wregex` over the code points.
                // `std::wstring(cpts.begin(), cpts.end())` converts each one
                // to `wchar_t`, which is 16 bits on Windows: there the pattern
                // and the text both lose everything above U+FFFF, and the
                // whitespace test reads the narrowed value.
                const wtext = allocator.dupe(u32, cpts) catch oom();
                defer allocator.free(wtext);
                const wexpr = allocator.dupe(u32, cpts_regex) catch oom();
                defer allocator.free(wexpr);
                if (comptime wchar_bits == 16) {
                    for (wtext) |*w| w.* = @as(u16, @truncate(w.*));
                    for (wexpr) |*w| w.* = @as(u16, @truncate(w.*));
                }
                for (wtext) |*w| {
                    if (w.* > 0x7F and cptFlagsFromCpt(w.*).is_whitespace) w.* = 0x0B;
                }
                tmp.clearRetainingCapacity();
                break :blk splitStl(u32, wtext, wexpr, bpe_offsets.items, &tmp);
            }
        };
        r catch |err| {
            impl.writeStderr(&.{ "Failed to process regex: '", regex_expr, "'\n" });
            impl.writeStderr(&.{ "Regex error: ", regex.errorText(err), "\n" });
            return error.RegexFailed;
        };
        std.mem.swap(std.ArrayList(usize), &bpe_offsets, &tmp);
    }

    var words: Words = .{};
    errdefer words.deinit();
    words.ends.ensureTotalCapacity(allocator, bpe_offsets.items.len) catch oom();
    var start: usize = 0;
    for (bpe_offsets.items) |offset| {
        for (cpts[start .. start + offset]) |cpt| try appendCptUtf8(&words.buf, cpt);
        words.finish();
        start += offset;
    }

    if (byte_encode) {
        defer words.deinit();
        return byteEncodingProcess(&words);
    }
    return words;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "the flag bits sit where the C++ bit-field puts them" {
    const f: CptFlags = .{ .is_number = true, .is_whitespace = true };
    try std.testing.expectEqual(@as(u16, CptFlags.number | 0x0100), f.asUint());
    try std.testing.expect(cptFlagsFromCpt('7').is_number);
    try std.testing.expect(cptFlagsFromCpt(' ').is_whitespace);
    try std.testing.expect(cptFlagsFromCpt(0x110000).is_undefined);
}

test "a malformed byte decodes as U+FFFD and decoding resumes after it" {
    const cpts = cptsFromUtf8("a\xffb\xc3\xa9");
    defer allocator.free(cpts);
    try std.testing.expectEqualSlices(u32, &.{ 'a', 0xFFFD, 'b', 0xE9 }, cpts);
}

test "the byte map round-trips every byte" {
    for (0..256) |b| {
        try std.testing.expectEqual(@as(u8, @intCast(b)), try utf8ToByte(byteToUtf8(@intCast(b))));
    }
    try std.testing.expectError(error.OutOfRange, utf8ToByte("ab"));
}

test "the gpt2 splitter keeps a contraction and a space-led word" {
    var words = try regexSplit("I'm here  now", &.{"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"}, false);
    defer words.deinit();
    const want = [_][]const u8{ "I", "'m", " here", " ", " now" };
    try std.testing.expectEqual(want.len, words.len());
    for (want, 0..) |w, i| try std.testing.expectEqualStrings(w, words.get(i));
}
