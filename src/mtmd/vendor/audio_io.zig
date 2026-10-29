//! The byte streams miniaudio's decoders read a memory buffer through.
//!
//! # Provenance
//!
//! Ports the memory read, seek and tell callbacks of
//! `llama.cpp/vendor/miniaudio/miniaudio.h` (miniaudio v0.11.25) at v0.4.1
//! (`b29c606e2`): `ma_dr_wav__on_read_memory` and friends, the dr_flac and
//! dr_mp3 equivalents, and `ma_decoder__on_read_memory` and friends.
//!
//! They are not interchangeable, which is why this is one type with a mode:
//! `ma_decoder_init_memory` first tries each backend over its own memory
//! stream, whose out-of-range seek **fails**, and when all three fail tries
//! them again through the decoder's callbacks, whose out-of-range seek
//! **clamps** and succeeds. A file a backend rejects one way it can accept
//! the other.

const std = @import("std");

/// `ma_dr_wav_seek_origin` and friends.
pub const Origin = enum { set, cur, end };

/// A memory stream with one backend's seek semantics.
pub const Io = struct {
    data: []const u8,
    pos: usize = 0,
    mode: Mode,

    pub const Mode = enum {
        /// `ma_dr_wav__on_seek_memory` (miniaudio.h:82551 @b29c606e2):
        /// fails outside `[0, size]`.
        dr_wav,
        /// `ma_dr_flac__on_seek_memory`: fails past the end.
        dr_flac,
        /// `ma_dr_mp3__on_seek_memory`: fails past the end.
        dr_mp3,
        /// `ma_decoder__on_seek_memory` (miniaudio.h:66412 @b29c606e2):
        /// clamps to `[0, size]` and always succeeds.
        decoder,
    };

    /// Ports `ma_dr_wav__on_read_memory` (miniaudio.h:82535 @b29c606e2)
    /// and `ma_decoder__on_read_memory` (miniaudio.h:66381 @b29c606e2),
    /// which agree on the bytes delivered.
    ///
    /// Return: the bytes copied into `buf`, fewer at the end.
    pub fn read(self: *Io, buf: []u8) usize {
        const n = @min(buf.len, self.data.len - self.pos);
        @memcpy(buf[0..n], self.data[self.pos..][0..n]);
        self.pos += n;
        return n;
    }

    /// One seek, as the mode's callback makes it.
    ///
    /// Return: whether the callback reported success.
    pub fn seek(self: *Io, offset: i64, origin: Origin) bool {
        switch (self.mode) {
            .decoder => {
                switch (origin) {
                    .cur => {
                        if (offset > 0) {
                            const room = self.data.len - self.pos;
                            self.pos += @intCast(@min(@as(u64, @intCast(offset)), room));
                        } else {
                            const back: u64 = @intCast(-offset);
                            self.pos -= @intCast(@min(back, self.pos));
                        }
                    },
                    .end => {
                        const back: u64 = @abs(offset);
                        self.pos = if (back > self.data.len) 0 else self.data.len - @as(usize, @intCast(back));
                    },
                    .set => {
                        // `(size_t)byteOffset`: a negative offset is a huge one
                        const off: u64 = @bitCast(offset);
                        self.pos = if (off <= self.data.len) @intCast(off) else self.data.len;
                    },
                }
                return true;
            },
            else => {
                const base: i64 = switch (origin) {
                    .set => 0,
                    .cur => @intCast(self.pos),
                    .end => @intCast(self.data.len),
                };
                const cursor = base + offset;
                if (cursor < 0) return false;
                if (cursor > self.data.len) return false;
                self.pos = @intCast(cursor);
                return true;
            },
        }
    }

    /// Ports `ma_dr_wav__seek_forward` (miniaudio.h:80372 @b29c606e2):
    /// forward in steps of at most 0x7FFFFFFF.
    ///
    /// A clamping stream reaches its end on the first step and stays, so
    /// its remaining steps are taken at once; the C takes them one by one,
    /// which for a wrapped size is 2^33 calls.
    pub fn seekForward(self: *Io, offset: u64) bool {
        var remaining = offset;
        while (remaining > 0) {
            const step: u64 = @min(remaining, 0x7FFFFFFF);
            if (!self.seek(@intCast(step), .cur)) return false;
            remaining -= step;
            if (self.mode == .decoder and self.pos == self.data.len) return true;
        }
        return true;
    }

    /// Ports `ma_dr_wav__seek_from_start` (miniaudio.h:80390 @b29c606e2).
    pub fn seekFromStart(self: *Io, offset: u64) bool {
        if (offset <= 0x7FFFFFFF) return self.seek(@intCast(offset), .set);
        if (!self.seek(0x7FFFFFFF, .set)) return false;
        return self.seekForward(offset - 0x7FFFFFFF);
    }

    /// Ports `ma_dr_wav__on_tell_memory` (miniaudio.h:82629 @b29c606e2)
    /// and `ma_decoder__on_tell_memory` (miniaudio.h:66455 @b29c606e2).
    pub fn tell(self: *const Io) i64 {
        return @intCast(self.pos);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "a strict stream refuses to seek past its end, a clamping one stops there" {
    var strict: Io = .{ .data = "abcd", .mode = .dr_wav };
    try std.testing.expect(!strict.seek(5, .set));
    try std.testing.expectEqual(@as(usize, 0), strict.pos);
    var clamp: Io = .{ .data = "abcd", .mode = .decoder };
    try std.testing.expect(clamp.seek(5, .set));
    try std.testing.expectEqual(@as(usize, 4), clamp.pos);
}
