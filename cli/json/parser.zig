//! JSON text to a `Value`, as `ordered_json::parse` reads it.
//!
//! # Provenance
//!
//! Ports `nlohmann::detail::lexer`, `nlohmann::detail::parser` and the DOM
//! half of `json_sax_dom_parser` from `llama.cpp/vendor/nlohmann/json.hpp`
//! (nlohmann/json 3.12.0), and `common_json::parse` and `parse_no_throw`
//! from `llama.cpp/common/json.cpp` (llama.cpp v0.4.1, `b29c606e2`). Each
//! declaration names what it ports and the line it began at.
//!
//! # What has to match
//!
//! Acceptance, the number classification and the error text, byte for
//! byte, because `json-schema-to-grammar.cpp` and `chat.cpp` put `what()`
//! in front of the user. Four of the lexer's habits show in that text and
//! are kept on purpose:
//!
//! - **`last read` is everything since the last string or number.** The
//!   token string is cleared only when a string or number starts, so after
//!   `[1, tru]` it reads `1, tru]`, punctuation and whitespace included.
//! - **A NUL byte outside a string ends the input.** `scan()` treats `'\0'`
//!   as end of input, so `1\0garbage` parses as `1`.
//! - **A byte order mark is read only at the very start**, and a broken one
//!   is its own error.
//! - **The column counts every `get()`**, end of input included, and an
//!   `unget()` of a newline does not restore the column it reset.
//!
//! Comments are off (`ignore_comments` false), as `common_json::parse`
//! calls it, and that half of the lexer is not ported.

const std = @import("std");
const errors = @import("errors.zig");
const value_mod = @import("value.zig");

const Value = value_mod.Value;
const Error = errors.Error;
const Position = errors.Position;

/// `char_traits<char>::eof()`.
const eof: i32 = -1;

/// Ports `lexer_base::token_type`.
const TokenType = enum {
    uninitialized,
    literal_true,
    literal_false,
    literal_null,
    value_string,
    value_unsigned,
    value_integer,
    value_float,
    begin_array,
    begin_object,
    end_array,
    end_object,
    name_separator,
    value_separator,
    parse_error,
    end_of_input,
    literal_or_value,
};

/// Ports `token_type_name` (json.hpp:7085 @b29c606e2).
fn tokenTypeName(t: TokenType) []const u8 {
    return switch (t) {
        .uninitialized => "<uninitialized>",
        .literal_true => "true literal",
        .literal_false => "false literal",
        .literal_null => "null literal",
        .value_string => "string literal",
        .value_unsigned, .value_integer, .value_float => "number literal",
        .begin_array => "'['",
        .begin_object => "'{'",
        .end_array => "']'",
        .end_object => "'}'",
        .name_separator => "':'",
        .value_separator => "','",
        .parse_error => "<parse error>",
        .end_of_input => "end of input",
        .literal_or_value => "'[', '{', or a literal",
    };
}

/// The names `scan_string` gives the 32 control characters.
const control_names = [32][]const u8{
    "NUL", "SOH", "STX", "ETX", "EOT", "ENQ", "ACK", "BEL",
    "BS",  "HT",  "LF",  "VT",  "FF",  "CR",  "SO",  "SI",
    "DLE", "DC1", "DC2", "DC3", "DC4", "NAK", "SYN", "ETB",
    "CAN", "EM",  "SUB", "ESC", "FS",  "GS",  "RS",  "US",
};

/// `scan_string`'s message for an unescaped control character, one per
/// byte: `U+000A (LF) must be escaped to \u000A or \n`.
const control_messages = blk: {
    var out: [32][]const u8 = undefined;
    for (0..32) |c| {
        const short: []const u8 = switch (c) {
            0x08 => " or \\b",
            0x09 => " or \\t",
            0x0A => " or \\n",
            0x0C => " or \\f",
            0x0D => " or \\r",
            else => "",
        };
        out[c] = std.fmt.comptimePrint("invalid string: control character U+{X:0>4} ({s}) must be escaped to \\u{X:0>4}{s}", .{ c, control_names[c], c, short });
    }
    break :blk out;
};

/// Ports `lexer` (json.hpp:7134 @b29c606e2) over a byte string, the input
/// adapter `parse(const std::string &)` builds.
const Lexer = struct {
    input: []const u8,
    next_index: usize = 0,

    /// the current character
    current: i32 = eof,
    /// whether the next get() call should just return current
    next_unget: bool = false,
    /// the start position of the current token
    position: Position = .{},
    /// raw input token string (for error messages)
    token_string: std.ArrayList(u8),
    /// buffer for variable-length tokens (numbers, strings)
    token_buffer: std.ArrayList(u8),
    /// a description of occurred lexer errors
    error_message: []const u8 = "",

    value_integer: i64 = 0,
    value_unsigned: u64 = 0,
    value_float: f64 = 0,

    /// Both buffers hold at most what has been read since the last reset,
    /// so the input's length (plus the end-of-input slack) bounds them and
    /// they never grow.
    fn init(gpa: std.mem.Allocator, input: []const u8) Error!Lexer {
        var ts = try std.ArrayList(u8).initCapacity(gpa, input.len + 8);
        errdefer ts.deinit(gpa);
        const tb = try std.ArrayList(u8).initCapacity(gpa, input.len + 8);
        return .{ .input = input, .token_string = ts, .token_buffer = tb };
    }

    fn deinit(self: *Lexer, gpa: std.mem.Allocator) void {
        self.token_string.deinit(gpa);
        self.token_buffer.deinit(gpa);
    }

    /// The input adapter's `get_character`.
    fn getCharacter(self: *Lexer) i32 {
        if (self.next_index >= self.input.len) return eof;
        const c = self.input[self.next_index];
        self.next_index += 1;
        return c;
    }

    /// Ports `get` (json.hpp:8363 @b29c606e2).
    fn get(self: *Lexer) i32 {
        self.position.chars_read_total += 1;
        self.position.chars_read_current_line += 1;

        if (self.next_unget) {
            // just reset the next_unget variable and work with current
            self.next_unget = false;
        } else {
            self.current = self.getCharacter();
        }

        if (self.current != eof) {
            self.token_string.appendAssumeCapacity(@intCast(self.current));
        }

        if (self.current == '\n') {
            self.position.lines_read += 1;
            self.position.chars_read_current_line = 0;
        }

        return self.current;
    }

    /// Ports `unget` (json.hpp:8400 @b29c606e2).
    fn unget(self: *Lexer) void {
        self.next_unget = true;

        self.position.chars_read_total -= 1;

        // in case we "unget" a newline, we have to also decrement the lines_read
        if (self.position.chars_read_current_line == 0) {
            if (self.position.lines_read > 0) {
                self.position.lines_read -= 1;
            }
        } else {
            self.position.chars_read_current_line -= 1;
        }

        if (self.current != eof) {
            std.debug.assert(self.token_string.items.len > 0);
            _ = self.token_string.pop();
        }
    }

    /// Ports `add`.
    fn add(self: *Lexer, c: i32) void {
        self.token_buffer.appendAssumeCapacity(@truncate(@as(u32, @bitCast(c))));
    }

    /// Ports `reset` (json.hpp:8345 @b29c606e2).
    fn reset(self: *Lexer) void {
        self.token_buffer.clearRetainingCapacity();
        self.token_string.clearRetainingCapacity();
        self.token_string.appendAssumeCapacity(@truncate(@as(u32, @bitCast(self.current))));
    }

    /// Ports `get_codepoint` (json.hpp:7192 @b29c606e2): the four hex digits
    /// after `\u`, or -1.
    fn getCodepoint(self: *Lexer) i32 {
        std.debug.assert(self.current == 'u');
        var codepoint: i32 = 0;

        const factors = [_]u5{ 12, 8, 4, 0 };
        for (factors) |factor| {
            const c = self.get();
            if (c >= '0' and c <= '9') {
                codepoint += (c - 0x30) << factor;
            } else if (c >= 'A' and c <= 'F') {
                codepoint += (c - 0x37) << factor;
            } else if (c >= 'a' and c <= 'f') {
                codepoint += (c - 0x57) << factor;
            } else {
                return -1;
            }
        }

        std.debug.assert(0x0000 <= codepoint and codepoint <= 0xFFFF);
        return codepoint;
    }

    /// Ports `next_byte_in_range` (json.hpp:7240 @b29c606e2): adds the
    /// current byte, then reads one byte per range and checks it.
    fn nextByteInRange(self: *Lexer, ranges: []const [2]i32) bool {
        self.add(self.current);

        for (ranges) |range| {
            const c = self.get();
            if (range[0] <= c and c <= range[1]) {
                self.add(c);
            } else {
                self.error_message = "invalid string: ill-formed UTF-8 byte";
                return false;
            }
        }

        return true;
    }

    /// Ports `scan_string` (json.hpp:7277 @b29c606e2).
    fn scanString(self: *Lexer) TokenType {
        // reset token_buffer (ignore opening quote)
        self.reset();

        // we entered the function by reading an open quote
        std.debug.assert(self.current == '"');

        while (true) {
            // get next character
            const c = self.get();
            switch (c) {
                // end of file while parsing string
                eof => {
                    self.error_message = "invalid string: missing closing quote";
                    return .parse_error;
                },

                // closing quote
                '"' => return .value_string,

                // escapes
                '\\' => switch (self.get()) {
                    '"' => self.add('"'),
                    '\\' => self.add('\\'),
                    '/' => self.add('/'),
                    'b' => self.add(0x08),
                    'f' => self.add(0x0C),
                    'n' => self.add('\n'),
                    'r' => self.add('\r'),
                    't' => self.add('\t'),

                    // unicode escapes
                    'u' => {
                        const codepoint1 = self.getCodepoint();
                        var codepoint = codepoint1; // start with codepoint1

                        if (codepoint1 == -1) {
                            self.error_message = "invalid string: '\\u' must be followed by 4 hex digits";
                            return .parse_error;
                        }

                        // check if code point is a high surrogate
                        if (0xD800 <= codepoint1 and codepoint1 <= 0xDBFF) {
                            // expect next \uxxxx entry
                            if (self.get() == '\\' and self.get() == 'u') {
                                const codepoint2 = self.getCodepoint();

                                if (codepoint2 == -1) {
                                    self.error_message = "invalid string: '\\u' must be followed by 4 hex digits";
                                    return .parse_error;
                                }

                                // check if codepoint2 is a low surrogate
                                if (0xDC00 <= codepoint2 and codepoint2 <= 0xDFFF) {
                                    codepoint = @bitCast((@as(u32, @intCast(codepoint1)) << 10) +% @as(u32, @intCast(codepoint2)) -% 0x35FDC00);
                                } else {
                                    self.error_message = "invalid string: surrogate U+D800..U+DBFF must be followed by U+DC00..U+DFFF";
                                    return .parse_error;
                                }
                            } else {
                                self.error_message = "invalid string: surrogate U+D800..U+DBFF must be followed by U+DC00..U+DFFF";
                                return .parse_error;
                            }
                        } else {
                            if (0xDC00 <= codepoint1 and codepoint1 <= 0xDFFF) {
                                self.error_message = "invalid string: surrogate U+DC00..U+DFFF must follow U+D800..U+DBFF";
                                return .parse_error;
                            }
                        }

                        // result of the above calculation yields a proper codepoint
                        std.debug.assert(0x00 <= codepoint and codepoint <= 0x10FFFF);

                        // translate codepoint into bytes
                        const cp: u32 = @intCast(codepoint);
                        if (cp < 0x80) {
                            // 1-byte characters: 0xxxxxxx (ASCII)
                            self.add(@intCast(cp));
                        } else if (cp <= 0x7FF) {
                            // 2-byte characters: 110xxxxx 10xxxxxx
                            self.add(@intCast(0xC0 | (cp >> 6)));
                            self.add(@intCast(0x80 | (cp & 0x3F)));
                        } else if (cp <= 0xFFFF) {
                            // 3-byte characters: 1110xxxx 10xxxxxx 10xxxxxx
                            self.add(@intCast(0xE0 | (cp >> 12)));
                            self.add(@intCast(0x80 | ((cp >> 6) & 0x3F)));
                            self.add(@intCast(0x80 | (cp & 0x3F)));
                        } else {
                            // 4-byte characters: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                            self.add(@intCast(0xF0 | (cp >> 18)));
                            self.add(@intCast(0x80 | ((cp >> 12) & 0x3F)));
                            self.add(@intCast(0x80 | ((cp >> 6) & 0x3F)));
                            self.add(@intCast(0x80 | (cp & 0x3F)));
                        }
                    },

                    // other characters after escape
                    else => {
                        self.error_message = "invalid string: forbidden character after backslash";
                        return .parse_error;
                    },
                },

                // invalid control characters
                0x00...0x1F => {
                    self.error_message = control_messages[@intCast(c)];
                    return .parse_error;
                },

                // U+0020..U+007F (except U+0022 (quote) and U+005C (backslash))
                0x20...0x21, 0x23...0x5B, 0x5D...0x7F => self.add(c),

                // U+0080..U+07FF: bytes C2..DF 80..BF
                0xC2...0xDF => if (!self.nextByteInRange(&.{.{ 0x80, 0xBF }})) return .parse_error,

                // U+0800..U+0FFF: bytes E0 A0..BF 80..BF
                0xE0 => if (!self.nextByteInRange(&.{ .{ 0xA0, 0xBF }, .{ 0x80, 0xBF } })) return .parse_error,

                // U+1000..U+CFFF: bytes E1..EC 80..BF 80..BF
                // U+E000..U+FFFF: bytes EE..EF 80..BF 80..BF
                0xE1...0xEC, 0xEE...0xEF => if (!self.nextByteInRange(&.{ .{ 0x80, 0xBF }, .{ 0x80, 0xBF } })) return .parse_error,

                // U+D000..U+D7FF: bytes ED 80..9F 80..BF
                0xED => if (!self.nextByteInRange(&.{ .{ 0x80, 0x9F }, .{ 0x80, 0xBF } })) return .parse_error,

                // U+10000..U+3FFFF F0 90..BF 80..BF 80..BF
                0xF0 => if (!self.nextByteInRange(&.{ .{ 0x90, 0xBF }, .{ 0x80, 0xBF }, .{ 0x80, 0xBF } })) return .parse_error,

                // U+40000..U+FFFFF F1..F3 80..BF 80..BF 80..BF
                0xF1...0xF3 => if (!self.nextByteInRange(&.{ .{ 0x80, 0xBF }, .{ 0x80, 0xBF }, .{ 0x80, 0xBF } })) return .parse_error,

                // U+100000..U+10FFFF F4 80..8F 80..BF 80..BF
                0xF4 => if (!self.nextByteInRange(&.{ .{ 0x80, 0x8F }, .{ 0x80, 0xBF }, .{ 0x80, 0xBF } })) return .parse_error,

                // remaining bytes (80..C1 and F5..FF) are ill-formed
                else => {
                    self.error_message = "invalid string: ill-formed UTF-8 byte";
                    return .parse_error;
                },
            }
        }
    }

    /// Ports `scan_number` (json.hpp:7992 @b29c606e2): the state machine of
    /// RFC 8259's number grammar, then `strtoull`, `strtoll` or `strtod`.
    /// An integer that does not fit its type is a float, as `strtoull`'s
    /// `ERANGE` makes it.
    fn scanNumber(self: *Lexer) TokenType {
        // reset token_buffer to store the number's bytes
        self.reset();

        // the type of the parsed number; initially set to unsigned; will be
        // changed if minus sign, decimal point or exponent is read
        var number_type: TokenType = .value_unsigned;

        const State = enum { minus, zero, any1, decimal1, decimal2, exponent, sign, any2, done };

        // state (init): we just found out we need to scan a number
        var state: State = switch (self.current) {
            '-' => .minus,
            '0' => .zero,
            '1'...'9' => .any1,
            else => unreachable,
        };
        self.add(self.current);

        while (true) switch (state) {
            .minus => {
                // we just parsed a leading minus sign
                number_type = .value_integer;
                const c = self.get();
                switch (c) {
                    '0' => state = .zero,
                    '1'...'9' => state = .any1,
                    else => {
                        self.error_message = "invalid number; expected digit after '-'";
                        return .parse_error;
                    },
                }
                self.add(c);
            },
            .zero => {
                // we just parse a zero (maybe with a leading minus sign)
                const c = self.get();
                switch (c) {
                    '.' => {
                        self.add('.');
                        state = .decimal1;
                    },
                    'e', 'E' => {
                        self.add(c);
                        state = .exponent;
                    },
                    else => state = .done,
                }
            },
            .any1 => {
                // we just parsed a number 0-9 (maybe with a leading minus sign)
                const c = self.get();
                switch (c) {
                    '0'...'9' => self.add(c),
                    '.' => {
                        self.add('.');
                        state = .decimal1;
                    },
                    'e', 'E' => {
                        self.add(c);
                        state = .exponent;
                    },
                    else => state = .done,
                }
            },
            .decimal1 => {
                // we just parsed a decimal point
                number_type = .value_float;
                const c = self.get();
                switch (c) {
                    '0'...'9' => {
                        self.add(c);
                        state = .decimal2;
                    },
                    else => {
                        self.error_message = "invalid number; expected digit after '.'";
                        return .parse_error;
                    },
                }
            },
            .decimal2 => {
                // we just parsed at least one number after a decimal point
                const c = self.get();
                switch (c) {
                    '0'...'9' => self.add(c),
                    'e', 'E' => {
                        self.add(c);
                        state = .exponent;
                    },
                    else => state = .done,
                }
            },
            .exponent => {
                // we just parsed an exponent
                number_type = .value_float;
                const c = self.get();
                switch (c) {
                    '+', '-' => {
                        self.add(c);
                        state = .sign;
                    },
                    '0'...'9' => {
                        self.add(c);
                        state = .any2;
                    },
                    else => {
                        self.error_message = "invalid number; expected '+', '-', or digit after exponent";
                        return .parse_error;
                    },
                }
            },
            .sign => {
                // we just parsed an exponent sign
                const c = self.get();
                switch (c) {
                    '0'...'9' => {
                        self.add(c);
                        state = .any2;
                    },
                    else => {
                        self.error_message = "invalid number; expected digit after exponent sign";
                        return .parse_error;
                    },
                }
            },
            .any2 => {
                // we just parsed a number after the exponent or exponent sign
                const c = self.get();
                switch (c) {
                    '0'...'9' => self.add(c),
                    else => state = .done,
                }
            },
            .done => break,
        };

        // unget the character after the number (we only read it to know
        // that we are done scanning a number)
        self.unget();

        const text = self.token_buffer.items;

        // try to parse integers first and fall back to floats
        if (number_type == .value_unsigned) {
            if (std.fmt.parseInt(u64, text, 10)) |x| {
                self.value_unsigned = x;
                return .value_unsigned;
            } else |_| {}
        } else if (number_type == .value_integer) {
            if (std.fmt.parseInt(i64, text, 10)) |x| {
                self.value_integer = x;
                return .value_integer;
            } else |_| {}
        }

        // this code is reached if we parse a floating-point number or if an
        // integer conversion above failed
        self.value_float = std.fmt.parseFloat(f64, text) catch unreachable;
        return .value_float;
    }

    /// Ports `scan_literal` (json.hpp:8325 @b29c606e2).
    fn scanLiteral(self: *Lexer, literal_text: []const u8, return_type: TokenType) TokenType {
        std.debug.assert(self.current == literal_text[0]);
        for (literal_text[1..]) |ch| {
            if (self.get() != ch) {
                self.error_message = "invalid literal";
                return .parse_error;
            }
        }
        return return_type;
    }

    /// Ports `get_token_string` (json.hpp:8479 @b29c606e2): the bytes read
    /// for the current token, control characters as `<U+XXXX>`.
    fn tokenString(self: *const Lexer, out: *std.ArrayList(u8), gpa: std.mem.Allocator) Error!void {
        for (self.token_string.items) |c| {
            if (c <= 0x1F) {
                try out.print(gpa, "<U+{X:0>4}>", .{c});
            } else {
                try out.append(gpa, c);
            }
        }
    }

    /// Ports `skip_bom` (json.hpp:8517 @b29c606e2).
    fn skipBom(self: *Lexer) bool {
        if (self.get() == 0xEF) {
            // check if we completely parse the BOM
            return self.get() == 0xBB and self.get() == 0xBF;
        }

        // the first character is not the beginning of the BOM; unget it to
        // process is later
        self.unget();
        return true;
    }

    /// Ports `skip_whitespace` (json.hpp:8531 @b29c606e2).
    fn skipWhitespace(self: *Lexer) void {
        while (true) {
            const c = self.get();
            if (!(c == ' ' or c == '\t' or c == '\n' or c == '\r')) break;
        }
    }

    /// Ports `scan` (json.hpp:8540 @b29c606e2).
    fn scan(self: *Lexer) TokenType {
        // initially, skip the BOM
        if (self.position.chars_read_total == 0 and !self.skipBom()) {
            self.error_message = "invalid BOM; must be 0xEF 0xBB 0xBF if given";
            return .parse_error;
        }

        // read next character and ignore whitespace
        self.skipWhitespace();

        return switch (self.current) {
            // structural characters
            '[' => .begin_array,
            ']' => .end_array,
            '{' => .begin_object,
            '}' => .end_object,
            ':' => .name_separator,
            ',' => .value_separator,

            // literals
            't' => self.scanLiteral("true", .literal_true),
            'f' => self.scanLiteral("false", .literal_false),
            'n' => self.scanLiteral("null", .literal_null),

            // string
            '"' => self.scanString(),

            // number
            '-', '0'...'9' => self.scanNumber(),

            // end of input (the null byte is needed when parsing from
            // string literals)
            0, eof => .end_of_input,

            // error
            else => blk: {
                self.error_message = "invalid literal";
                break :blk .parse_error;
            },
        };
    }
};

/// Ports `parser` (json.hpp:12918 @b29c606e2) with `json_sax_dom_parser`
/// building the tree.
const Parser = struct {
    gpa: std.mem.Allocator,
    lexer: Lexer,
    last_token: TokenType = .uninitialized,
    allow_exceptions: bool,

    /// the parsed value
    root: Value = .null,
    /// stack to model hierarchy of values
    ref_stack: std.ArrayList(*Value) = .empty,
    /// helper to hold the reference for the next object element
    object_element: ?*Value = null,

    fn getToken(self: *Parser) TokenType {
        self.last_token = self.lexer.scan();
        return self.last_token;
    }

    /// Ports `exception_message` (json.hpp:13336 @b29c606e2).
    fn exceptionMessage(self: *Parser, out: *std.ArrayList(u8), expected: TokenType, context: []const u8) Error!void {
        const a = self.gpa;
        try out.appendSlice(a, "syntax error ");

        if (context.len != 0) {
            try out.print(a, "while parsing {s} ", .{context});
        }

        try out.appendSlice(a, "- ");

        if (self.last_token == .parse_error) {
            try out.print(a, "{s}; last read: '", .{self.lexer.error_message});
            try self.lexer.tokenString(out, a);
            try out.append(a, '\'');
        } else {
            try out.print(a, "unexpected {s}", .{tokenTypeName(self.last_token)});
        }

        if (expected != .uninitialized) {
            try out.print(a, "; expected {s}", .{tokenTypeName(expected)});
        }
    }

    /// `sax->parse_error(..., parse_error::create(101, ...))`: the error the
    /// parser throws, or (without exceptions) just the failure.
    fn syntaxError(self: *Parser, expected: TokenType, context: []const u8) Error {
        if (!self.allow_exceptions) return error.JsonError;
        var msg: std.ArrayList(u8) = .empty;
        defer msg.deinit(self.gpa);
        try self.exceptionMessage(&msg, expected, context);
        return errors.parseError(101, self.lexer.position, msg.items);
    }

    /// `parse_error::create(101, ..., "attempting to parse an empty input...")`.
    fn emptyInputError(self: *Parser) Error {
        if (!self.allow_exceptions) return error.JsonError;
        return errors.parseError(101, self.lexer.position, "attempting to parse an empty input; check that your input string or stream contains the expected JSON");
    }

    /// Ports `json_sax_dom_parser`'s `handle_value` (json.hpp:9071 @b29c606e2).
    fn handleValue(self: *Parser, v: Value) Error!*Value {
        if (self.ref_stack.items.len == 0) {
            self.root = v;
            return &self.root;
        }

        const top = self.ref_stack.items[self.ref_stack.items.len - 1];
        switch (top.*) {
            .array => |*arr| {
                var owned = v;
                arr.append(self.gpa, owned) catch |e| {
                    owned.deinit(self.gpa);
                    return e;
                };
                return &arr.items[arr.items.len - 1];
            },
            else => {
                const slot = self.object_element.?;
                slot.assign(self.gpa, v);
                return slot;
            },
        }
    }

    /// Ports `json_sax_dom_parser`'s `key` (json.hpp:8908 @b29c606e2): the
    /// member, found or added, that the next value lands in.
    fn key(self: *Parser) Error!void {
        const top = self.ref_stack.items[self.ref_stack.items.len - 1];
        self.object_element = try top.object.emplace(self.gpa, self.lexer.token_buffer.items);
    }

    fn string(self: *Parser) Error!Value {
        return Value.initString(self.gpa, self.lexer.token_buffer.items);
    }

    /// Ports `sax_parse_internal` (json.hpp:13038 @b29c606e2).
    fn saxParseInternal(self: *Parser) Error!void {
        // stack to remember the hierarchy of structured values we are
        // parsing; true = array; false = object
        var states: std.ArrayList(bool) = .empty;
        defer states.deinit(self.gpa);

        // value to avoid a goto (see comment where set to true)
        var skip_to_state_evaluation = false;

        while (true) {
            if (!skip_to_state_evaluation) {
                // invariant: get_token() was called before each iteration
                switch (self.last_token) {
                    .begin_object => {
                        try self.ref_stack.append(self.gpa, try self.handleValue(Value.emptyObject()));

                        // closing } -> we are done
                        if (self.getToken() == .end_object) {
                            _ = self.ref_stack.pop();
                        } else {
                            // parse key
                            if (self.last_token != .value_string) {
                                return self.syntaxError(.value_string, "object key");
                            }
                            try self.key();

                            // parse separator (:)
                            if (self.getToken() != .name_separator) {
                                return self.syntaxError(.name_separator, "object separator");
                            }

                            // remember we are now inside an object
                            try states.append(self.gpa, false);

                            // parse values
                            _ = self.getToken();
                            continue;
                        }
                    },

                    .begin_array => {
                        try self.ref_stack.append(self.gpa, try self.handleValue(Value.emptyArray()));

                        // closing ] -> we are done
                        if (self.getToken() == .end_array) {
                            _ = self.ref_stack.pop();
                        } else {
                            // remember we are now inside an array
                            try states.append(self.gpa, true);

                            // parse values (no need to call get_token)
                            continue;
                        }
                    },

                    .value_float => {
                        const res = self.lexer.value_float;

                        if (!std.math.isFinite(res)) {
                            if (!self.allow_exceptions) return error.JsonError;
                            var ts: std.ArrayList(u8) = .empty;
                            defer ts.deinit(self.gpa);
                            try self.lexer.tokenString(&ts, self.gpa);
                            return errors.fail(.out_of_range, 406, "number overflow parsing '{s}'", .{ts.items});
                        }

                        _ = try self.handleValue(.{ .number_float = res });
                    },

                    .literal_false => _ = try self.handleValue(.{ .boolean = false }),
                    .literal_null => _ = try self.handleValue(.null),
                    .literal_true => _ = try self.handleValue(.{ .boolean = true }),
                    .value_integer => _ = try self.handleValue(.{ .number_integer = self.lexer.value_integer }),
                    .value_string => _ = try self.handleValue(try self.string()),
                    .value_unsigned => _ = try self.handleValue(.{ .number_unsigned = self.lexer.value_unsigned }),

                    .parse_error => {
                        // using "uninitialized" to avoid "expected" message
                        return self.syntaxError(.uninitialized, "value");
                    },

                    .end_of_input => {
                        if (self.lexer.position.chars_read_total == 1) {
                            return self.emptyInputError();
                        }
                        return self.syntaxError(.literal_or_value, "value");
                    },

                    // the last token was unexpected
                    else => return self.syntaxError(.literal_or_value, "value"),
                }
            } else {
                skip_to_state_evaluation = false;
            }

            // we reached this line after we successfully parsed a value
            if (states.items.len == 0) {
                // empty stack: we reached the end of the hierarchy: done
                return;
            }

            if (states.items[states.items.len - 1]) { // array
                // comma -> next value
                if (self.getToken() == .value_separator) {
                    // parse a new value
                    _ = self.getToken();
                    continue;
                }

                // closing ]
                if (self.last_token == .end_array) {
                    _ = self.ref_stack.pop();

                    // We are done with this array. Before we can parse a
                    // new value, we need to evaluate the new state first.
                    // By setting skip_to_state_evaluation to false, we
                    // are effectively jumping to the beginning of this if.
                    _ = states.pop();
                    skip_to_state_evaluation = true;
                    continue;
                }

                return self.syntaxError(.end_array, "array");
            }

            // states.back() is false -> object

            // comma -> next value
            if (self.getToken() == .value_separator) {
                // parse key
                if (self.getToken() != .value_string) {
                    return self.syntaxError(.value_string, "object key");
                }

                try self.key();

                // parse separator (:)
                if (self.getToken() != .name_separator) {
                    return self.syntaxError(.name_separator, "object separator");
                }

                // parse values
                _ = self.getToken();
                continue;
            }

            // closing }
            if (self.last_token == .end_object) {
                _ = self.ref_stack.pop();

                // We are done with this object. Before we can parse a new
                // value, we need to evaluate the new state first. By setting
                // skip_to_state_evaluation to false, we are effectively
                // jumping to the beginning of this if.
                _ = states.pop();
                skip_to_state_evaluation = true;
                continue;
            }

            return self.syntaxError(.end_object, "object");
        }
    }

    /// Ports `parser`'s `parse` (json.hpp:12951 @b29c606e2) with `strict`:
    /// one value, then nothing but whitespace.
    fn parse(self: *Parser) Error!void {
        // read first token (the parser constructor does)
        _ = self.getToken();

        try self.saxParseInternal();

        // in strict mode, input must be completely read
        if (self.getToken() != .end_of_input) {
            return self.syntaxError(.end_of_input, "value");
        }
    }
};

fn run(gpa: std.mem.Allocator, text: []const u8, allow_exceptions: bool) Error!Value {
    var p: Parser = .{
        .gpa = gpa,
        .lexer = try Lexer.init(gpa, text),
        .allow_exceptions = allow_exceptions,
    };
    defer p.lexer.deinit(gpa);
    defer p.ref_stack.deinit(gpa);
    p.parse() catch |e| {
        p.root.deinit(gpa);
        return e;
    };
    return p.root;
}

/// Ports `common_json`'s `parse` (json.cpp:201 @b29c606e2) over
/// `basic_json`'s `parse` (json.hpp:24057 @b29c606e2).
///
/// Parameters:
/// - `gpa`: allocates the tree.
/// - `text`: the whole input; a NUL byte outside a string ends it.
///
/// Return: the value, owned by the caller; `error.JsonError` with
/// `errors.what()` set for text that is not JSON.
pub fn parse(gpa: std.mem.Allocator, text: []const u8) Error!Value {
    return run(gpa, text, true);
}

/// Ports `common_json`'s `parse_no_throw` (json.cpp:212 @b29c606e2): as
/// `parse`, but text that is not JSON gives a discarded value.
///
/// Return: the value, owned by the caller.
pub fn parseNoThrow(gpa: std.mem.Allocator, text: []const u8) error{OutOfMemory}!Value {
    return run(gpa, text, false) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.JsonError => .discarded,
    };
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

fn expectParseError(text: []const u8, what: []const u8) !void {
    try std.testing.expectError(error.JsonError, parse(std.testing.allocator, text));
    try std.testing.expectEqualStrings(what, errors.what());
    var d = try parseNoThrow(std.testing.allocator, text);
    try std.testing.expect(d.isDiscarded());
    d.deinit(std.testing.allocator);
}

test "parse errors read as nlohmann's" {
    try expectParseError("", "[json.exception.parse_error.101] parse error at line 1, column 1: attempting to parse an empty input; check that your input string or stream contains the expected JSON");
    try expectParseError("[1, tru]", "[json.exception.parse_error.101] parse error at line 1, column 8: syntax error while parsing value - invalid literal; last read: '1, tru]'");
    try expectParseError("{\"a\" 1}", "[json.exception.parse_error.101] parse error at line 1, column 6: syntax error while parsing object separator - unexpected number literal; expected ':'");
    try expectParseError("\"a\tb\"", "[json.exception.parse_error.101] parse error at line 1, column 3: syntax error while parsing value - invalid string: control character U+0009 (HT) must be escaped to \\u0009 or \\t; last read: '\"a<U+0009>'");
    try expectParseError("1e400", "[json.exception.out_of_range.406] number overflow parsing '1e400'");
    try expectParseError("1 2", "[json.exception.parse_error.101] parse error at line 1, column 3: syntax error while parsing value - unexpected number literal; expected end of input");
}

test "parse classifies numbers and keeps duplicate keys in first place" {
    const a = std.testing.allocator;
    var v = try parse(a, "{\"x\": 18446744073709551615, \"y\": -9223372036854775808, \"x\": 18446744073709551616, \"z\": -0}");
    defer v.deinit(a);
    try std.testing.expectEqual(@as(usize, 3), v.size());
    try std.testing.expectEqualStrings("x", v.object.members.items[0].key);
    try std.testing.expect(v.object.members.items[0].value == .number_float);
    try std.testing.expect(v.object.members.items[1].value == .number_integer);
    try std.testing.expectEqual(@as(i64, 0), v.object.members.items[2].value.number_integer);
    var n = try parse(a, "7\x00garbage");
    defer n.deinit(a);
    try std.testing.expectEqual(@as(u64, 7), n.number_unsigned);
}
