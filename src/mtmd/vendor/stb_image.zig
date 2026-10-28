//! The image decoder `mtmd-helper` reaches: `stbi_load_from_memory`, every
//! format stb_image compiles by default.
//!
//! # Provenance
//!
//! Ports `llama.cpp/vendor/stb/stb_image.h` (stb_image v2.30) at v0.4.0
//! (`5266f24da`), the memory-loading path only. Each declaration names the
//! function it ports and the line it began at.
//!
//! # As `mtmd-helper.cpp` compiles it
//!
//! `mtmd-helper.cpp` defines `STB_IMAGE_IMPLEMENTATION` and nothing else, in
//! a C++ translation unit, so:
//!
//! - **no SIMD on aarch64** (`STBI_NEON` is opt-in), and on x86 the SSE2
//!   kernels are documented bit-identical to the scalar ones this file
//!   ports (`stbi__idct_simd`, `stbi__YCbCr_to_RGB_simd`: stb_image.h:2528, 3657 @5266f24da);
//! - **`pow(float, float)` is C++'s float overload**, `powf`, in the HDR to
//!   LDR conversion -- a C build would call the double `pow`;
//! - `STBI_ASSERT` is `assert`, compiled out under `NDEBUG`.
//!
//! Only `stbi__load_main`'s 8-bit path is reachable: no flipping (the flag
//! is never set), no 16-bit or float API, no callbacks. A buffer stb leaves
//! uninitialized -- a JPEG whose scans do not cover every block, a truncated
//! TGA -- is zeroed here; `make mtmd-decode-diff` compiles the reference with
//! `calloc` for the same reason.

const std = @import("std");

/// What `stbi_load_from_memory` returns.
pub const Image = struct {
    /// The pixels, `x * y * comp_out` bytes, owned by the caller (free with
    /// the allocator passed in). `comp_out` is `req_comp`, or `comp` when
    /// `req_comp` is 0.
    pixels: []u8,
    x: c_int,
    y: c_int,
    /// The components the file holds, as stb reports them.
    comp: c_int,
};

/// The result of a load: the image, or the reason stb gives for failing
/// (`stbi_failure_reason`), which is null where the failing path sets none.
pub const Result = union(enum) {
    ok: Image,
    failed: ?[]const u8,
};

const max_dimensions: u32 = 1 << 24;

/// Set by a load whose C counterpart read memory it never wrote, so its
/// pixels are unspecified: a BMP index past the palette `stbi__bmp_load`
/// filled (its `pal` is an uninitialized local), or a short read in the flat
/// HDR loop (its `rgbe` is). The port reads zeros there; the gate compares
/// such a load's shape and outcome but not its pixels.
pub threadlocal var read_indeterminate: bool = false;

/// Ports `stbi_load_from_memory` (stb_image.h:1429 @5266f24da).
///
/// Parameters:
/// - `gpa`: allocates the pixels and every intermediate buffer.
/// - `buffer`: the file's bytes; only `len` truncated to `int` is read, as
///   the C does.
/// - `req_comp`: the components wanted, 0 for the file's own.
///
/// Return: the image or the failure. On success the caller owns `pixels`.
pub fn loadFromMemory(gpa: std.mem.Allocator, buffer: []const u8, req_comp: c_int) Result {
    read_indeterminate = false;
    const len: usize = @intCast(@as(u32, @truncate(buffer.len)) & 0x7fffffff);
    var s: Context = .{ .gpa = gpa, .buf = buffer[0..len] };
    var x: c_int = 0;
    var y: c_int = 0;
    var comp: c_int = 0;
    const data = s.loadAndPostprocess8bit(&x, &y, &comp, req_comp) orelse return .{ .failed = s.failure };
    return .{ .ok = .{ .pixels = data, .x = x, .y = y, .comp = comp } };
}

// -----------------------------------------------------------------------------
// the context

/// Ports `stbi__context` (stb_image.h:820 @5266f24da), for memory only.
const Context = struct {
    gpa: std.mem.Allocator,
    img_x: u32 = 0,
    img_y: u32 = 0,
    img_n: c_int = 0,
    img_out_n: c_int = 0,
    buf: []const u8,
    /// `img_buffer - img_buffer_original`; may run past `buf.len` after a
    /// skip, as the C pointer does.
    pos: usize = 0,
    /// `stbi__g_failure_reason`.
    failure: ?[]const u8 = null,

    /// Ports `stbi__err` (stb_image.h:978 @5266f24da).
    fn err(self: *Context, str: []const u8) c_int {
        self.failure = str;
        return 0;
    }

    /// `stbi__errpuc`: the error, as a null pointer.
    fn errpuc(self: *Context, str: []const u8) ?[]u8 {
        self.failure = str;
        return null;
    }

    /// Ports `stbi__rewind` (stb_image.h:886 @5266f24da).
    fn rewind(self: *Context) void {
        self.pos = 0;
    }

    /// Ports `stbi__get8` (stb_image.h:1614 @5266f24da).
    fn get8(self: *Context) u8 {
        if (self.pos < self.buf.len) {
            const v = self.buf[self.pos];
            self.pos += 1;
            return v;
        }
        return 0;
    }

    /// Ports `stbi__at_eof` (stb_image.h:1628 @5266f24da).
    fn atEof(self: *const Context) bool {
        return self.pos >= self.buf.len;
    }

    /// Ports `stbi__skip` (stb_image.h:1644 @5266f24da).
    fn skip(self: *Context, n: c_int) void {
        if (n == 0) return; // already there!
        if (n < 0) {
            self.pos = self.buf.len;
            return;
        }
        self.pos += @intCast(n);
    }

    /// Ports `stbi__getn` (stb_image.h:1666 @5266f24da).
    fn getn(self: *Context, out: []u8) bool {
        if (self.pos <= self.buf.len and out.len <= self.buf.len - self.pos) {
            @memcpy(out, self.buf[self.pos..][0..out.len]);
            self.pos += out.len;
            return true;
        }
        return false;
    }

    /// Ports `stbi__get16be` (stb_image.h:1694 @5266f24da).
    fn get16be(self: *Context) c_int {
        const z: c_int = self.get8();
        return (z << 8) + self.get8();
    }

    /// Ports `stbi__get32be` (stb_image.h:1704 @5266f24da).
    fn get32be(self: *Context) u32 {
        const z: u32 = @intCast(self.get16be());
        return (z << 16) +% @as(u32, @intCast(self.get16be()));
    }

    /// Ports `stbi__get16le` (stb_image.h:1714 @5266f24da).
    fn get16le(self: *Context) c_int {
        const z: c_int = self.get8();
        return z + (@as(c_int, self.get8()) << 8);
    }

    /// Ports `stbi__get32le` (stb_image.h:1722 @5266f24da).
    fn get32le(self: *Context) u32 {
        var z: u32 = @intCast(self.get16le());
        z +%= @as(u32, @intCast(self.get16le())) << 16;
        return z;
    }

    /// `stbi__malloc` of `n` bytes, zeroed (see the file header).
    fn alloc(self: *Context, n: usize) ?[]u8 {
        const p = self.gpa.alloc(u8, n) catch return null;
        @memset(p, 0);
        return p;
    }

    fn free(self: *Context, p: ?[]u8) void {
        if (p) |q| self.gpa.free(q);
    }

    /// Ports `stbi__malloc_mad2` (stb_image.h:1048 @5266f24da).
    fn mallocMad2(self: *Context, a: c_int, b: c_int, add: c_int) ?[]u8 {
        if (!mad2sizesValid(a, b, add)) return null;
        return self.alloc(@intCast(a * b + add));
    }

    /// Ports `stbi__malloc_mad3` (stb_image.h:1055 @5266f24da).
    fn mallocMad3(self: *Context, a: c_int, b: c_int, c: c_int, add: c_int) ?[]u8 {
        if (!mad3sizesValid(a, b, c, add)) return null;
        return self.alloc(@intCast(a * b * c + add));
    }

    /// Ports `stbi__malloc_mad4` (stb_image.h:1062 @5266f24da).
    fn mallocMad4(self: *Context, a: c_int, b: c_int, c: c_int, d: c_int, add: c_int) ?[]u8 {
        if (!mad4sizesValid(a, b, c, d, add)) return null;
        return self.alloc(@intCast(a * b * c * d + add));
    }

    fn pngTest(self: *Context) bool {
        return pngTestImpl(self);
    }
    fn pngLoad(self: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int, bits: *c_int) ?[]u8 {
        return pngLoadImpl(self, x, y, comp, req_comp, bits);
    }
    fn bmpTest(self: *Context) bool {
        return bmpTestImpl(self);
    }
    fn bmpLoad(self: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
        return bmpLoadImpl(self, x, y, comp, req_comp);
    }
    fn gifTest(self: *Context) bool {
        return gifTestImpl(self);
    }
    fn gifLoad(self: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
        return gifLoadImpl(self, x, y, comp, req_comp);
    }
    fn psdTest(self: *Context) bool {
        return psdTestImpl(self);
    }
    fn psdLoad(self: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
        return psdLoadImpl(self, x, y, comp, req_comp);
    }
    fn picTest(self: *Context) bool {
        return picTestImpl(self);
    }
    fn picLoad(self: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
        return picLoadImpl(self, x, y, comp, req_comp);
    }
    fn jpegTest(self: *Context) bool {
        return jpegTestImpl(self);
    }
    fn jpegLoad(self: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
        return jpegLoadImpl(self, x, y, comp, req_comp);
    }
    fn pnmTest(self: *Context) bool {
        return pnmTestImpl(self);
    }
    fn pnmLoad(self: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int, bits: *c_int) ?[]u8 {
        return pnmLoadImpl(self, x, y, comp, req_comp, bits);
    }
    fn hdrTest(self: *Context) bool {
        return hdrTestImpl(self);
    }
    fn hdrLoad(self: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]f32 {
        return hdrLoadImpl(self, x, y, comp, req_comp);
    }
    fn tgaTest(self: *Context) bool {
        return tgaTestImpl(self);
    }
    fn tgaLoad(self: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
        return tgaLoadImpl(self, x, y, comp, req_comp);
    }

    /// Ports `stbi__load_main` (stb_image.h:1137 @5266f24da), with `bpc` 8.
    /// Return: the pixels and their bits per channel, 8 or 16.
    fn loadMain(self: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int, bits: *c_int) ?[]u8 {
        bits.* = 8; // default is 8 so most paths don't have to be changed

        // test the formats with a very explicit header first (at least a FOURCC
        // or distinctive magic number first)
        if (self.pngTest()) return self.pngLoad(x, y, comp, req_comp, bits);
        if (self.bmpTest()) return self.bmpLoad(x, y, comp, req_comp);
        if (self.gifTest()) return self.gifLoad(x, y, comp, req_comp);
        if (self.psdTest()) return self.psdLoad(x, y, comp, req_comp);
        if (self.picTest()) return self.picLoad(x, y, comp, req_comp);

        // then the formats that can end up attempting to load with just 1 or 2
        // bytes matching expectations; these are prone to false positives, so
        // try them later
        if (self.jpegTest()) return self.jpegLoad(x, y, comp, req_comp);
        if (self.pnmTest()) return self.pnmLoad(x, y, comp, req_comp, bits);

        if (self.hdrTest()) {
            const hdr = self.hdrLoad(x, y, comp, req_comp);
            return self.hdrToLdr(hdr, x.*, y.*, if (req_comp != 0) req_comp else comp.*);
        }

        // test tga last because it's a crappy test!
        if (self.tgaTest()) return self.tgaLoad(x, y, comp, req_comp);

        return self.errpuc("unknown image type");
    }

    /// Ports `stbi__load_and_postprocess_8bit` (stb_image.h:1260 @5266f24da).
    fn loadAndPostprocess8bit(self: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
        var bits: c_int = 8;
        const result = self.loadMain(x, y, comp, req_comp, &bits) orelse return null;
        if (bits != 8) return self.convert16to8(result, x.*, y.*, if (req_comp == 0) comp.* else req_comp);
        return result;
    }

    /// Ports `stbi__convert_16_to_8` (stb_image.h:1190 @5266f24da): the top
    /// byte of each native-order sample.
    fn convert16to8(self: *Context, orig: []u8, w: c_int, h: c_int, channels: c_int) ?[]u8 {
        const img_len: usize = @intCast(w * h * channels);
        const reduced = self.alloc(img_len) orelse {
            self.free(orig);
            return self.errpuc("outofmem");
        };
        for (0..img_len) |i| reduced[i] = @truncate(rd16(orig, i) >> 8);
        self.free(orig);
        return reduced;
    }

    /// Ports `stbi__convert_format` (stb_image.h:1755 @5266f24da).
    fn convertFormat(self: *Context, data: []u8, img_n: c_int, req_comp: c_int, x: u32, y: u32) ?[]u8 {
        if (req_comp == img_n) return data;
        const good = self.mallocMad3(req_comp, @bitCast(x), @bitCast(y), 0) orelse {
            self.free(data);
            return self.errpuc("outofmem");
        };
        const in_n: usize = @intCast(img_n);
        const out_n: usize = @intCast(req_comp);
        const combo = img_n * 8 + req_comp;
        // the C switches once per row, so a combination it has no case for
        // fails only when there is a row
        if (y > 0 and !validCombo(combo)) {
            self.free(data);
            self.free(good);
            return self.errpuc("unsupported");
        }
        for (0..y) |j| {
            var src = j * x * in_n;
            var dest = j * x * out_n;
            for (0..x) |_| {
                const s = data[src..];
                const d = good[dest..];
                switch (combo) {
                    1 * 8 + 2 => {
                        d[0] = s[0];
                        d[1] = 255;
                    },
                    1 * 8 + 3 => {
                        d[0] = s[0];
                        d[1] = s[0];
                        d[2] = s[0];
                    },
                    1 * 8 + 4 => {
                        d[0] = s[0];
                        d[1] = s[0];
                        d[2] = s[0];
                        d[3] = 255;
                    },
                    2 * 8 + 1 => d[0] = s[0],
                    2 * 8 + 3 => {
                        d[0] = s[0];
                        d[1] = s[0];
                        d[2] = s[0];
                    },
                    2 * 8 + 4 => {
                        d[0] = s[0];
                        d[1] = s[0];
                        d[2] = s[0];
                        d[3] = s[1];
                    },
                    3 * 8 + 4 => {
                        d[0] = s[0];
                        d[1] = s[1];
                        d[2] = s[2];
                        d[3] = 255;
                    },
                    3 * 8 + 1 => d[0] = computeY(s[0], s[1], s[2]),
                    3 * 8 + 2 => {
                        d[0] = computeY(s[0], s[1], s[2]);
                        d[1] = 255;
                    },
                    4 * 8 + 1 => d[0] = computeY(s[0], s[1], s[2]),
                    4 * 8 + 2 => {
                        d[0] = computeY(s[0], s[1], s[2]);
                        d[1] = s[3];
                    },
                    4 * 8 + 3 => {
                        d[0] = s[0];
                        d[1] = s[1];
                        d[2] = s[2];
                    },
                    else => {
                        self.free(data);
                        self.free(good);
                        return self.errpuc("unsupported");
                    },
                }
                src += in_n;
                dest += out_n;
            }
        }
        self.free(data);
        return good;
    }

    /// Ports `stbi__convert_format16` (stb_image.h:1812 @5266f24da), over
    /// native-order 16-bit samples.
    fn convertFormat16(self: *Context, data: []u8, img_n: c_int, req_comp: c_int, x: u32, y: u32) ?[]u8 {
        if (req_comp == img_n) return data;
        const good = self.alloc(@as(usize, @intCast(req_comp)) * x * y * 2) orelse {
            self.free(data);
            return self.errpuc("outofmem");
        };
        const in_n: usize = @intCast(img_n);
        const out_n: usize = @intCast(req_comp);
        const combo = img_n * 8 + req_comp;
        // the C switches once per row, so a combination it has no case for
        // fails only when there is a row
        if (y > 0 and !validCombo(combo)) {
            self.free(data);
            self.free(good);
            return self.errpuc("unsupported");
        }
        for (0..y) |j| {
            var src = j * x * in_n;
            var dest = j * x * out_n;
            for (0..x) |_| {
                const s0 = rd16(data, src);
                switch (combo) {
                    1 * 8 + 2 => {
                        wr16(good, dest, s0);
                        wr16(good, dest + 1, 0xffff);
                    },
                    1 * 8 + 3, 2 * 8 + 3 => {
                        wr16(good, dest, s0);
                        wr16(good, dest + 1, s0);
                        wr16(good, dest + 2, s0);
                    },
                    1 * 8 + 4 => {
                        wr16(good, dest, s0);
                        wr16(good, dest + 1, s0);
                        wr16(good, dest + 2, s0);
                        wr16(good, dest + 3, 0xffff);
                    },
                    2 * 8 + 1 => wr16(good, dest, s0),
                    2 * 8 + 4 => {
                        wr16(good, dest, s0);
                        wr16(good, dest + 1, s0);
                        wr16(good, dest + 2, s0);
                        wr16(good, dest + 3, rd16(data, src + 1));
                    },
                    3 * 8 + 4 => {
                        wr16(good, dest, s0);
                        wr16(good, dest + 1, rd16(data, src + 1));
                        wr16(good, dest + 2, rd16(data, src + 2));
                        wr16(good, dest + 3, 0xffff);
                    },
                    3 * 8 + 1, 4 * 8 + 1 => wr16(good, dest, computeY16(s0, rd16(data, src + 1), rd16(data, src + 2))),
                    3 * 8 + 2 => {
                        wr16(good, dest, computeY16(s0, rd16(data, src + 1), rd16(data, src + 2)));
                        wr16(good, dest + 1, 0xffff);
                    },
                    4 * 8 + 2 => {
                        wr16(good, dest, computeY16(s0, rd16(data, src + 1), rd16(data, src + 2)));
                        wr16(good, dest + 1, rd16(data, src + 3));
                    },
                    4 * 8 + 3 => {
                        wr16(good, dest, s0);
                        wr16(good, dest + 1, rd16(data, src + 1));
                        wr16(good, dest + 2, rd16(data, src + 2));
                    },
                    else => {
                        self.free(data);
                        self.free(good);
                        return self.errpuc("unsupported");
                    },
                }
                src += in_n;
                dest += out_n;
            }
        }
        self.free(data);
        return good;
    }

    /// Ports `stbi__hdr_to_ldr` (stb_image.h:1884 @5266f24da). `pow` is the
    /// C++ float overload, and `pow(...) * 255 + 0.5f` contracts into one
    /// fused multiply-add.
    fn hdrToLdr(self: *Context, data_in: ?[]f32, x: c_int, y: c_int, comp: c_int) ?[]u8 {
        const data = data_in orelse return null;
        defer self.gpa.free(data);
        const output = self.mallocMad3(x, y, comp, 0) orelse return self.errpuc("outofmem");
        // compute number of non-alpha components
        const n: c_int = if (comp & 1 != 0) comp else comp - 1;
        const cu: usize = @intCast(comp);
        const nu: usize = @intCast(n);
        const npix: usize = @intCast(x * y);
        for (0..npix) |i| {
            var k: usize = 0;
            while (k < nu) : (k += 1) {
                var z = @mulAdd(f32, powf(data[i * cu + k] * h2l_scale_i, h2l_gamma_i), 255, 0.5);
                if (z < 0) z = 0;
                if (z > 255) z = 255;
                output[i * cu + k] = @intCast(@as(c_int, @intFromFloat(z)));
            }
            if (k < cu) {
                var z = @mulAdd(f32, data[i * cu + k], 255, 0.5);
                if (z < 0) z = 0;
                if (z > 255) z = 255;
                output[i * cu + k] = @intCast(@as(c_int, @intFromFloat(z)));
            }
        }
        return output;
    }
};

extern "c" fn powf(x: f32, y: f32) f32;
extern "c" fn ldexpf(x: f32, e: c_int) f32;

/// The `(img_n, req_comp)` pairs `stbi__convert_format` has a case for.
fn validCombo(combo: c_int) bool {
    return switch (combo) {
        1 * 8 + 2, 1 * 8 + 3, 1 * 8 + 4, 2 * 8 + 1, 2 * 8 + 3, 2 * 8 + 4, 3 * 8 + 4, 3 * 8 + 1, 3 * 8 + 2, 4 * 8 + 1, 4 * 8 + 2, 4 * 8 + 3 => true,
        else => false,
    };
}

/// `stbi__h2l_gamma_i` and `stbi__h2l_scale_i` (stb_image.h:1579 @5266f24da).
const h2l_gamma_i: f32 = 1.0 / 2.2;
const h2l_scale_i: f32 = 1.0;

fn rd16(buf: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, buf[i * 2 ..][0..2], .native);
}

fn wr16(buf: []u8, i: usize, v: u16) void {
    std.mem.writeInt(u16, buf[i * 2 ..][0..2], v, .native);
}

/// Ports `stbi__addsizes_valid` (stb_image.h:1002 @5266f24da).
fn addsizesValid(a: c_int, b: c_int) bool {
    if (b < 0) return false;
    return a <= std.math.maxInt(c_int) - b;
}

/// Ports `stbi__mul2sizes_valid` (stb_image.h:1014 @5266f24da).
fn mul2sizesValid(a: c_int, b: c_int) bool {
    if (a < 0 or b < 0) return false;
    if (b == 0) return true; // mul-by-0 is always safe
    return a <= @divTrunc(std.math.maxInt(c_int), b);
}

/// Ports `stbi__mad2sizes_valid` (stb_image.h:1024 @5266f24da).
fn mad2sizesValid(a: c_int, b: c_int, add: c_int) bool {
    return mul2sizesValid(a, b) and addsizesValid(a * b, add);
}

/// Ports `stbi__mad3sizes_valid` (stb_image.h:1031 @5266f24da).
fn mad3sizesValid(a: c_int, b: c_int, c: c_int, add: c_int) bool {
    return mul2sizesValid(a, b) and mul2sizesValid(a * b, c) and addsizesValid(a * b * c, add);
}

/// Ports `stbi__mad4sizes_valid` (stb_image.h:1039 @5266f24da).
fn mad4sizesValid(a: c_int, b: c_int, c: c_int, d: c_int, add: c_int) bool {
    return mul2sizesValid(a, b) and mul2sizesValid(a * b, c) and
        mul2sizesValid(a * b * c, d) and addsizesValid(a * b * c * d, add);
}

/// Ports `stbi__addints_valid` (stb_image.h:1070 @5266f24da).
fn addintsValid(a: c_int, b: c_int) bool {
    if ((a >= 0) != (b >= 0)) return true; // a and b have different signs, so no overflow
    if (a < 0 and b < 0) return a >= std.math.minInt(c_int) - b;
    return a <= std.math.maxInt(c_int) - b;
}

/// Ports `stbi__mul2shorts_valid` (stb_image.h:1078 @5266f24da).
fn mul2shortsValid(a: c_int, b: c_int) bool {
    if (b == 0 or b == -1) return true; // multiplication by 0 is always 0; check for -1 so SHRT_MIN/b doesn't overflow
    if ((a >= 0) == (b >= 0)) return a <= @divTrunc(@as(c_int, std.math.maxInt(i16)), b);
    if (b < 0) return a <= @divTrunc(@as(c_int, std.math.minInt(i16)), b);
    return a >= @divTrunc(@as(c_int, std.math.minInt(i16)), b);
}

/// Ports `stbi__compute_y` (stb_image.h:1746 @5266f24da).
fn computeY(r: c_int, g: c_int, b: c_int) u8 {
    return @truncate(@as(c_uint, @bitCast(((r * 77) + (g * 150) + (29 * b)) >> 8)));
}

/// Ports `stbi__compute_y_16` (stb_image.h:1803 @5266f24da).
fn computeY16(r: c_int, g: c_int, b: c_int) u16 {
    return @truncate(@as(c_uint, @bitCast(((r * 77) + (g * 150) + (29 * b)) >> 8)));
}

// -----------------------------------------------------------------------------
// JPEG

const fast_bits = 9;

/// Ports `stbi__huffman` (stb_image.h:1947 @5266f24da).
const Huffman = struct {
    fast: [1 << fast_bits]u8 = @splat(0),
    code: [256]u16 = @splat(0),
    values: [256]u8 = @splat(0),
    size: [257]u8 = @splat(0),
    maxcode: [18]u32 = @splat(0),
    delta: [17]c_int = @splat(0), // old 'firstsymbol' - old 'firstcode'
};

/// One of the components of `stbi__jpeg` (stb_image.h:2001 @5266f24da).
const JpegComp = struct {
    id: c_int = 0,
    h: c_int = 0,
    v: c_int = 0,
    tq: c_int = 0,
    hd: c_int = 0,
    ha: c_int = 0,
    dc_pred: c_int = 0,
    x: c_int = 0,
    y: c_int = 0,
    w2: c_int = 0,
    h2: c_int = 0,
    data: ?[]u8 = null,
    linebuf: ?[]u8 = null,
    coeff: ?[]i16 = null, // progressive only
    coeff_w: c_int = 0, // number of 8x8 coefficient blocks
    coeff_h: c_int = 0,
};

/// Ports `stbi__jpeg` (stb_image.h:2001 @5266f24da).
const Jpeg = struct {
    s: *Context,
    huff_dc: [4]Huffman = @splat(.{}),
    huff_ac: [4]Huffman = @splat(.{}),
    dequant: [4][64]u16 = @splat(@splat(0)),
    fast_ac: [4][1 << fast_bits]i16 = @splat(@splat(0)),

    // sizes for components, interleaved MCUs
    img_h_max: c_int = 0,
    img_v_max: c_int = 0,
    img_mcu_x: c_int = 0,
    img_mcu_y: c_int = 0,
    img_mcu_w: c_int = 0,
    img_mcu_h: c_int = 0,

    img_comp: [4]JpegComp = @splat(.{}),

    code_buffer: u32 = 0, // jpeg entropy-coded buffer
    code_bits: c_int = 0, // number of valid bits
    marker: u8 = 0, // marker seen while filling entropy buffer
    nomore: bool = false, // flag if we saw a marker so must stop

    progressive: bool = false,
    spec_start: c_int = 0,
    spec_end: c_int = 0,
    succ_high: c_int = 0,
    succ_low: c_int = 0,
    eob_run: c_int = 0,
    jfif: bool = false,
    app14_color_transform: c_int = 0, // Adobe APP14 tag
    rgb: c_int = 0,

    scan_n: c_int = 0,
    order: [4]c_int = @splat(0),
    restart_interval: c_int = 0,
    todo: c_int = 0,

    /// Ports `stbi__grow_buffer_unsafe` (stb_image.h:2075 @5266f24da).
    fn growBufferUnsafe(j: *Jpeg) void {
        while (true) {
            const b: u32 = if (j.nomore) 0 else j.s.get8();
            if (b == 0xff) {
                var c = j.s.get8();
                while (c == 0xff) c = j.s.get8(); // consume fill bytes
                if (c != 0) {
                    j.marker = c;
                    j.nomore = true;
                    return;
                }
            }
            j.code_buffer |= b << @intCast(24 - j.code_bits);
            j.code_bits += 8;
            if (j.code_bits > 24) break;
        }
    }

    /// Ports `stbi__jpeg_huff_decode` (stb_image.h:2097 @5266f24da).
    fn huffDecode(j: *Jpeg, h: *const Huffman) c_int {
        if (j.code_bits < 16) j.growBufferUnsafe();

        // look at the top FAST_BITS and determine what symbol ID it is,
        // if the code is <= FAST_BITS
        var c: c_int = @intCast((j.code_buffer >> (32 - fast_bits)) & ((1 << fast_bits) - 1));
        var k: c_int = h.fast[@intCast(c)];
        if (k < 255) {
            const s: c_int = h.size[@intCast(k)];
            if (s > j.code_bits) return -1;
            j.code_buffer = shl32(j.code_buffer, s);
            j.code_bits -= s;
            return h.values[@intCast(k)];
        }

        // naive test is to shift the code_buffer down so k bits are valid,
        // then test against maxcode, preshifted
        const temp = j.code_buffer >> 16;
        k = fast_bits + 1;
        while (true) : (k += 1) {
            if (temp < h.maxcode[@intCast(k)]) break;
        }
        if (k == 17) {
            // error! code not found
            j.code_bits -= 16;
            return -1;
        }

        if (k > j.code_bits) return -1;

        // convert the huffman code to the symbol id
        c = @as(c_int, @intCast((j.code_buffer >> @intCast(32 - k)) & bmask[@intCast(k)])) + h.delta[@intCast(k)];
        if (c < 0 or c >= 256) return -1; // symbol id out of bounds!

        // convert the id to a symbol
        j.code_bits -= k;
        j.code_buffer = shl32(j.code_buffer, k);
        return h.values[@intCast(c)];
    }

    /// Ports `stbi__extend_receive` (stb_image.h:2153 @5266f24da).
    fn extendReceive(j: *Jpeg, n: c_int) c_int {
        if (j.code_bits < n) j.growBufferUnsafe();
        if (j.code_bits < n) return 0; // ran out of bits from stream, return 0s intead of continuing

        const sgn: c_int = @intCast(j.code_buffer >> 31); // sign bit always in MSB
        var k = std.math.rotl(u32, j.code_buffer, @as(u32, @intCast(n)));
        j.code_buffer = k & ~bmask[@intCast(n)];
        k &= bmask[@intCast(n)];
        j.code_bits -= n;
        return @as(c_int, @bitCast(k)) +% (jbias[@intCast(n)] & (sgn - 1));
    }

    /// Ports `stbi__jpeg_get_bits` (stb_image.h:2169 @5266f24da).
    fn getBits(j: *Jpeg, n: c_int) c_int {
        if (j.code_bits < n) j.growBufferUnsafe();
        if (j.code_bits < n) return 0;
        var k = std.math.rotl(u32, j.code_buffer, @as(u32, @intCast(n)));
        j.code_buffer = k & ~bmask[@intCast(n)];
        k &= bmask[@intCast(n)];
        j.code_bits -= n;
        return @bitCast(k);
    }

    /// Ports `stbi__jpeg_get_bit` (stb_image.h:2181 @5266f24da).
    fn getBit(j: *Jpeg) bool {
        if (j.code_bits < 1) j.growBufferUnsafe();
        if (j.code_bits < 1) return false;
        const k = j.code_buffer;
        j.code_buffer <<= 1;
        j.code_bits -= 1;
        return k & 0x80000000 != 0;
    }

    /// Ports `stbi__jpeg_decode_block` (stb_image.h:2210 @5266f24da).
    fn decodeBlock(j: *Jpeg, data: *[64]i16, hdc: *const Huffman, hac: *const Huffman, fac: *const [1 << fast_bits]i16, b: usize, dequant: *const [64]u16) bool {
        if (j.code_bits < 16) j.growBufferUnsafe();
        const t = j.huffDecode(hdc);
        if (t < 0 or t > 15) return j.s.err("bad huffman code") != 0;

        // 0 all the ac values now so we can do it 32-bits at a time
        data.* = @splat(0);

        const diff: c_int = if (t != 0) j.extendReceive(t) else 0;
        if (!addintsValid(j.img_comp[b].dc_pred, diff)) return j.s.err("bad delta") != 0;
        const dc = j.img_comp[b].dc_pred + diff;
        j.img_comp[b].dc_pred = dc;
        if (!mul2shortsValid(dc, dequant[0])) return j.s.err("can't merge dc and ac") != 0;
        data[0] = @truncate(dc * dequant[0]);

        // decode AC components, see JPEG spec
        var k: c_int = 1;
        while (true) {
            if (j.code_bits < 16) j.growBufferUnsafe();
            const c: usize = @intCast((j.code_buffer >> (32 - fast_bits)) & ((1 << fast_bits) - 1));
            var r: c_int = fac[c];
            if (r != 0) { // fast-AC path
                k += (r >> 4) & 15; // run
                const s = r & 15; // combined length
                if (s > j.code_bits) return j.s.err("bad huffman code") != 0;
                j.code_buffer = shl32(j.code_buffer, s);
                j.code_bits -= s;
                // decode into unzigzag'd location
                const zig = jpeg_dezigzag[@intCast(k)];
                k += 1;
                data[zig] = @truncate((r >> 8) *% dequant[zig]);
            } else {
                const rs = j.huffDecode(hac);
                if (rs < 0) return j.s.err("bad huffman code") != 0;
                const s = rs & 15;
                r = rs >> 4;
                if (s == 0) {
                    if (rs != 0xf0) break; // end block
                    k += 16;
                } else {
                    k += r;
                    // decode into unzigzag'd location
                    const zig = jpeg_dezigzag[@intCast(k)];
                    k += 1;
                    data[zig] = @truncate(j.extendReceive(s) *% dequant[zig]);
                }
            }
            if (k >= 64) break;
        }
        return true;
    }

    /// Ports `stbi__jpeg_decode_block_prog_dc` (stb_image.h:2265 @5266f24da).
    fn decodeBlockProgDc(j: *Jpeg, data: []i16, hdc: *const Huffman, b: usize) bool {
        if (j.spec_end != 0) return j.s.err("can't merge dc and ac") != 0;

        if (j.code_bits < 16) j.growBufferUnsafe();

        if (j.succ_high == 0) {
            // first scan for DC coefficient, must be first
            @memset(data[0..64], 0); // 0 all the ac values now
            const t = j.huffDecode(hdc);
            if (t < 0 or t > 15) return j.s.err("can't merge dc and ac") != 0;
            const diff: c_int = if (t != 0) j.extendReceive(t) else 0;

            if (!addintsValid(j.img_comp[b].dc_pred, diff)) return j.s.err("bad delta") != 0;
            const dc = j.img_comp[b].dc_pred + diff;
            j.img_comp[b].dc_pred = dc;
            if (!mul2shortsValid(dc, @as(c_int, 1) << @intCast(j.succ_low))) return j.s.err("can't merge dc and ac") != 0;
            data[0] = @truncate(dc * (@as(c_int, 1) << @intCast(j.succ_low)));
        } else {
            // refinement scan for DC coefficient
            if (j.getBit()) data[0] +%= @truncate(@as(c_int, 1) << @intCast(j.succ_low));
        }
        return true;
    }

    /// Ports `stbi__jpeg_decode_block_prog_ac` (stb_image.h:2295 @5266f24da).
    fn decodeBlockProgAc(j: *Jpeg, data: []i16, hac: *const Huffman, fac: *const [1 << fast_bits]i16) bool {
        if (j.spec_start == 0) return j.s.err("can't merge dc and ac") != 0;

        if (j.succ_high == 0) {
            const shift = j.succ_low;

            if (j.eob_run != 0) {
                j.eob_run -= 1;
                return true;
            }

            var k = j.spec_start;
            while (true) {
                if (j.code_bits < 16) j.growBufferUnsafe();
                const c: usize = @intCast((j.code_buffer >> (32 - fast_bits)) & ((1 << fast_bits) - 1));
                var r: c_int = fac[c];
                if (r != 0) { // fast-AC path
                    k += (r >> 4) & 15; // run
                    const s = r & 15; // combined length
                    if (s > j.code_bits) return j.s.err("bad huffman code") != 0;
                    j.code_buffer = shl32(j.code_buffer, s);
                    j.code_bits -= s;
                    const zig = jpeg_dezigzag[@intCast(k)];
                    k += 1;
                    data[zig] = @truncate((r >> 8) *% (@as(c_int, 1) << @intCast(shift)));
                } else {
                    const rs = j.huffDecode(hac);
                    if (rs < 0) return j.s.err("bad huffman code") != 0;
                    const s = rs & 15;
                    r = rs >> 4;
                    if (s == 0) {
                        if (r < 15) {
                            j.eob_run = @as(c_int, 1) << @intCast(r);
                            if (r != 0) j.eob_run += j.getBits(r);
                            j.eob_run -= 1;
                            break;
                        }
                        k += 16;
                    } else {
                        k += r;
                        const zig = jpeg_dezigzag[@intCast(k)];
                        k += 1;
                        data[zig] = @truncate(j.extendReceive(s) *% (@as(c_int, 1) << @intCast(shift)));
                    }
                }
                if (k > j.spec_end) break;
            }
        } else {
            // refinement scan for these AC coefficients
            const bit: i16 = @truncate(@as(c_int, 1) << @intCast(j.succ_low));

            if (j.eob_run != 0) {
                j.eob_run -= 1;
                var k = j.spec_start;
                while (k <= j.spec_end) : (k += 1) {
                    const p = &data[jpeg_dezigzag[@intCast(k)]];
                    if (p.* != 0) {
                        if (j.getBit()) {
                            if ((p.* & bit) == 0) {
                                if (p.* > 0) p.* +%= bit else p.* -%= bit;
                            }
                        }
                    }
                }
            } else {
                var k = j.spec_start;
                while (true) {
                    const rs = j.huffDecode(hac);
                    if (rs < 0) return j.s.err("bad huffman code") != 0;
                    var s: c_int = rs & 15;
                    var r: c_int = rs >> 4;
                    if (s == 0) {
                        if (r < 15) {
                            j.eob_run = (@as(c_int, 1) << @intCast(r)) - 1;
                            if (r != 0) j.eob_run += j.getBits(r);
                            r = 64; // force end of block
                        } else {
                            // r=15 s=0 should write 16 0s, so we just do a run of
                            // 15 0s and then write s (which is 0)
                        }
                    } else {
                        if (s != 1) return j.s.err("bad huffman code") != 0;
                        // sign bit
                        s = if (j.getBit()) bit else -bit;
                    }

                    // advance by r
                    while (k <= j.spec_end) {
                        const p = &data[jpeg_dezigzag[@intCast(k)]];
                        k += 1;
                        if (p.* != 0) {
                            if (j.getBit()) {
                                if ((p.* & bit) == 0) {
                                    if (p.* > 0) p.* +%= bit else p.* -%= bit;
                                }
                            }
                        } else {
                            if (r == 0) {
                                p.* = @truncate(s);
                                break;
                            }
                            r -= 1;
                        }
                    }
                    if (k > j.spec_end) break;
                }
            }
        }
        return true;
    }

    /// Ports `stbi__get_marker` (stb_image.h:2919 @5266f24da).
    fn getMarker(j: *Jpeg) u8 {
        if (j.marker != marker_none) {
            const x = j.marker;
            j.marker = marker_none;
            return x;
        }
        var x = j.s.get8();
        if (x != 0xff) return marker_none;
        while (x == 0xff) x = j.s.get8(); // consume repeated 0xff fill bytes
        return x;
    }

    /// Ports `stbi__jpeg_reset` (stb_image.h:2936 @5266f24da).
    fn reset(j: *Jpeg) void {
        j.code_bits = 0;
        j.code_buffer = 0;
        j.nomore = false;
        for (&j.img_comp) |*c| c.dc_pred = 0;
        j.marker = marker_none;
        j.todo = if (j.restart_interval != 0) j.restart_interval else 0x7fffffff;
        j.eob_run = 0;
    }

    /// The restart countdown every MCU loop ends on. Return: false to stop
    /// decoding (successfully), true to go on.
    fn countdown(z: *Jpeg) bool {
        z.todo -= 1;
        if (z.todo <= 0) {
            if (z.code_bits < 24) z.growBufferUnsafe();
            // if it's NOT a restart, then just bail, so we get corrupt data
            // rather than no data
            if (!isRestart(z.marker)) return false;
            z.reset();
        }
        return true;
    }

    /// Ports `stbi__parse_entropy_coded_data` (stb_image.h:2949 @5266f24da).
    fn parseEntropyCodedData(z: *Jpeg) bool {
        z.reset();
        if (!z.progressive) {
            var data: [64]i16 = undefined;
            if (z.scan_n == 1) {
                const n: usize = @intCast(z.order[0]);
                // non-interleaved data, we just need to process one block at a time,
                // in trivial scanline order
                const w = (z.img_comp[n].x + 7) >> 3;
                const h = (z.img_comp[n].y + 7) >> 3;
                var j: c_int = 0;
                while (j < h) : (j += 1) {
                    var i: c_int = 0;
                    while (i < w) : (i += 1) {
                        const ha: usize = @intCast(z.img_comp[n].ha);
                        if (!z.decodeBlock(&data, &z.huff_dc[@intCast(z.img_comp[n].hd)], &z.huff_ac[ha], &z.fast_ac[ha], n, &z.dequant[@intCast(z.img_comp[n].tq)])) return false;
                        const off: usize = @intCast(z.img_comp[n].w2 * j * 8 + i * 8);
                        idctBlock(z.img_comp[n].data.?[off..].ptr, @intCast(z.img_comp[n].w2), &data);
                        // every data block is an MCU, so countdown the restart interval
                        if (!z.countdown()) return true;
                    }
                }
                return true;
            } else { // interleaved
                var j: c_int = 0;
                while (j < z.img_mcu_y) : (j += 1) {
                    var i: c_int = 0;
                    while (i < z.img_mcu_x) : (i += 1) {
                        // scan an interleaved mcu... process scan_n components in order
                        var k: usize = 0;
                        while (k < z.scan_n) : (k += 1) {
                            const n: usize = @intCast(z.order[k]);
                            var y: c_int = 0;
                            while (y < z.img_comp[n].v) : (y += 1) {
                                var x: c_int = 0;
                                while (x < z.img_comp[n].h) : (x += 1) {
                                    const x2 = (i * z.img_comp[n].h + x) * 8;
                                    const y2 = (j * z.img_comp[n].v + y) * 8;
                                    const ha: usize = @intCast(z.img_comp[n].ha);
                                    if (!z.decodeBlock(&data, &z.huff_dc[@intCast(z.img_comp[n].hd)], &z.huff_ac[ha], &z.fast_ac[ha], n, &z.dequant[@intCast(z.img_comp[n].tq)])) return false;
                                    const off: usize = @intCast(z.img_comp[n].w2 * y2 + x2);
                                    idctBlock(z.img_comp[n].data.?[off..].ptr, @intCast(z.img_comp[n].w2), &data);
                                }
                            }
                        }
                        // after all interleaved components, that's an interleaved MCU,
                        // so now count down the restart interval
                        if (!z.countdown()) return true;
                    }
                }
                return true;
            }
        } else {
            if (z.scan_n == 1) {
                const n: usize = @intCast(z.order[0]);
                const w = (z.img_comp[n].x + 7) >> 3;
                const h = (z.img_comp[n].y + 7) >> 3;
                var j: c_int = 0;
                while (j < h) : (j += 1) {
                    var i: c_int = 0;
                    while (i < w) : (i += 1) {
                        const off: usize = @intCast(64 * (i + j * z.img_comp[n].coeff_w));
                        const data = z.img_comp[n].coeff.?[off..];
                        if (z.spec_start == 0) {
                            if (!z.decodeBlockProgDc(data, &z.huff_dc[@intCast(z.img_comp[n].hd)], n)) return false;
                        } else {
                            const ha: usize = @intCast(z.img_comp[n].ha);
                            if (!z.decodeBlockProgAc(data, &z.huff_ac[ha], &z.fast_ac[ha])) return false;
                        }
                        // every data block is an MCU, so countdown the restart interval
                        if (!z.countdown()) return true;
                    }
                }
                return true;
            } else { // interleaved
                var j: c_int = 0;
                while (j < z.img_mcu_y) : (j += 1) {
                    var i: c_int = 0;
                    while (i < z.img_mcu_x) : (i += 1) {
                        var k: usize = 0;
                        while (k < z.scan_n) : (k += 1) {
                            const n: usize = @intCast(z.order[k]);
                            var y: c_int = 0;
                            while (y < z.img_comp[n].v) : (y += 1) {
                                var x: c_int = 0;
                                while (x < z.img_comp[n].h) : (x += 1) {
                                    const x2 = i * z.img_comp[n].h + x;
                                    const y2 = j * z.img_comp[n].v + y;
                                    const off: usize = @intCast(64 * (x2 + y2 * z.img_comp[n].coeff_w));
                                    if (!z.decodeBlockProgDc(z.img_comp[n].coeff.?[off..], &z.huff_dc[@intCast(z.img_comp[n].hd)], n)) return false;
                                }
                            }
                        }
                        if (!z.countdown()) return true;
                    }
                }
                return true;
            }
        }
    }

    /// Ports `stbi__jpeg_finish` (stb_image.h:3080 @5266f24da): dequantize
    /// and IDCT a progressive image's coefficients.
    fn finish(z: *Jpeg) void {
        if (!z.progressive) return;
        var n: usize = 0;
        while (n < z.s.img_n) : (n += 1) {
            const w = (z.img_comp[n].x + 7) >> 3;
            const h = (z.img_comp[n].y + 7) >> 3;
            var j: c_int = 0;
            while (j < h) : (j += 1) {
                var i: c_int = 0;
                while (i < w) : (i += 1) {
                    const off: usize = @intCast(64 * (i + j * z.img_comp[n].coeff_w));
                    const data: *[64]i16 = z.img_comp[n].coeff.?[off..][0..64];
                    // `stbi__jpeg_dequantize` (stb_image.h:3073 @5266f24da)
                    const dq = &z.dequant[@intCast(z.img_comp[n].tq)];
                    for (0..64) |q| data[q] = @truncate(@as(c_int, data[q]) *% dq[q]);
                    const doff: usize = @intCast(z.img_comp[n].w2 * j * 8 + i * 8);
                    idctBlock(z.img_comp[n].data.?[doff..].ptr, @intCast(z.img_comp[n].w2), data);
                }
            }
        }
    }

    /// Ports `stbi__process_marker` (stb_image.h:3099 @5266f24da).
    fn processMarker(z: *Jpeg, m: c_int) bool {
        const s = z.s;
        switch (m) {
            marker_none => return s.err("expected marker") != 0, // no marker found
            0xDD => { // DRI - specify restart interval
                if (s.get16be() != 4) return s.err("bad DRI len") != 0;
                z.restart_interval = s.get16be();
                return true;
            },
            0xDB => { // DQT - define quantization table
                var l = s.get16be() - 2;
                while (l > 0) {
                    const q = s.get8();
                    const p = q >> 4;
                    const sixteen = p != 0;
                    const t = q & 15;
                    if (p != 0 and p != 1) return s.err("bad DQT type") != 0;
                    if (t > 3) return s.err("bad DQT table") != 0;
                    for (0..64) |i| {
                        z.dequant[t][jpeg_dezigzag[i]] = @truncate(@as(c_uint, @bitCast(if (sixteen) s.get16be() else @as(c_int, s.get8()))));
                    }
                    l -= if (sixteen) 129 else 65;
                }
                return l == 0;
            },
            0xC4 => { // DHT - define huffman table
                var l = s.get16be() - 2;
                while (l > 0) {
                    var sizes: [16]c_int = undefined;
                    var n: c_int = 0;
                    const q = s.get8();
                    const tc = q >> 4;
                    const th = q & 15;
                    if (tc > 1 or th > 3) return s.err("bad DHT header") != 0;
                    for (0..16) |i| {
                        sizes[i] = s.get8();
                        n += sizes[i];
                    }
                    if (n > 256) return s.err("bad DHT header") != 0; // Loop over i < n would write past end of values!
                    l -= 17;
                    const h = if (tc == 0) &z.huff_dc[th] else &z.huff_ac[th];
                    if (!buildHuffman(s, h, &sizes)) return false;
                    for (0..@intCast(n)) |i| h.values[i] = s.get8();
                    if (tc != 0) buildFastAc(&z.fast_ac[th], &z.huff_ac[th]);
                    l -= n;
                }
                return l == 0;
            },
            else => {},
        }

        // check for comment block or APP blocks
        if ((m >= 0xE0 and m <= 0xEF) or m == 0xFE) {
            var l = s.get16be();
            if (l < 2) {
                if (m == 0xFE) return s.err("bad COM len") != 0 else return s.err("bad APP len") != 0;
            }
            l -= 2;

            if (m == 0xE0 and l >= 5) { // JFIF APP0 segment
                const tag = "JFIF\x00";
                var ok = true;
                for (tag) |t| {
                    if (s.get8() != t) ok = false;
                }
                l -= 5;
                if (ok) z.jfif = true;
            } else if (m == 0xEE and l >= 12) { // Adobe APP14 segment
                const tag = "Adobe\x00";
                var ok = true;
                for (tag) |t| {
                    if (s.get8() != t) ok = false;
                }
                l -= 6;
                if (ok) {
                    _ = s.get8(); // version
                    _ = s.get16be(); // flags0
                    _ = s.get16be(); // flags1
                    z.app14_color_transform = s.get8(); // color transform
                    l -= 6;
                }
            }

            s.skip(l);
            return true;
        }

        return s.err("unknown marker") != 0;
    }

    /// Ports `stbi__process_scan_header` (stb_image.h:3203 @5266f24da).
    fn processScanHeader(z: *Jpeg) bool {
        const s = z.s;
        const ls = s.get16be();
        z.scan_n = s.get8();
        if (z.scan_n < 1 or z.scan_n > 4 or z.scan_n > s.img_n) return s.err("bad SOS component count") != 0;
        if (ls != 6 + 2 * z.scan_n) return s.err("bad SOS len") != 0;
        var i: usize = 0;
        while (i < z.scan_n) : (i += 1) {
            const id: c_int = s.get8();
            const q: c_int = s.get8();
            var which: usize = 0;
            while (which < s.img_n) : (which += 1) {
                if (z.img_comp[which].id == id) break;
            }
            if (which == s.img_n) return false; // no match
            z.img_comp[which].hd = q >> 4;
            if (z.img_comp[which].hd > 3) return s.err("bad DC huff") != 0;
            z.img_comp[which].ha = q & 15;
            if (z.img_comp[which].ha > 3) return s.err("bad AC huff") != 0;
            z.order[i] = @intCast(which);
        }

        z.spec_start = s.get8();
        z.spec_end = s.get8(); // should be 63, but might be 0
        const aa: c_int = s.get8();
        z.succ_high = aa >> 4;
        z.succ_low = aa & 15;
        if (z.progressive) {
            if (z.spec_start > 63 or z.spec_end > 63 or z.spec_start > z.spec_end or z.succ_high > 13 or z.succ_low > 13)
                return s.err("bad SOS") != 0;
        } else {
            if (z.spec_start != 0) return s.err("bad SOS") != 0;
            if (z.succ_high != 0 or z.succ_low != 0) return s.err("bad SOS") != 0;
            z.spec_end = 63;
        }
        return true;
    }

    /// Ports `stbi__free_jpeg_components` (stb_image.h:3242 @5266f24da).
    fn freeComponents(z: *Jpeg, ncomp: usize) void {
        for (z.img_comp[0..ncomp]) |*c| {
            if (c.data) |d| z.s.gpa.free(d);
            c.data = null;
            if (c.coeff) |d| z.s.gpa.free(d);
            c.coeff = null;
            if (c.linebuf) |d| z.s.gpa.free(d);
            c.linebuf = null;
        }
    }

    /// Ports `stbi__process_frame_header` (stb_image.h:3264 @5266f24da).
    fn processFrameHeader(z: *Jpeg, scan: Scan) bool {
        const s = z.s;
        var h_max: c_int = 1;
        var v_max: c_int = 1;
        const lf = s.get16be();
        if (lf < 11) return s.err("bad SOF len") != 0; // JPEG
        const p = s.get8();
        if (p != 8) return s.err("only 8-bit") != 0; // JPEG baseline
        s.img_y = @intCast(s.get16be());
        if (s.img_y == 0) return s.err("no header height") != 0; // Legal, but we don't handle it--but neither does IJG
        s.img_x = @intCast(s.get16be());
        if (s.img_x == 0) return s.err("0 width") != 0; // JPEG requires
        if (s.img_y > max_dimensions) return s.err("too large") != 0;
        if (s.img_x > max_dimensions) return s.err("too large") != 0;
        const c = s.get8();
        if (c != 3 and c != 1 and c != 4) return s.err("bad component count") != 0;
        s.img_n = c;
        for (z.img_comp[0..c]) |*comp| {
            comp.data = null;
            comp.linebuf = null;
        }

        if (lf != 8 + 3 * s.img_n) return s.err("bad SOF len") != 0;

        z.rgb = 0;
        const rgb = "RGB";
        var i: usize = 0;
        while (i < s.img_n) : (i += 1) {
            z.img_comp[i].id = s.get8();
            if (s.img_n == 3 and z.img_comp[i].id == rgb[i]) z.rgb += 1;
            const q: c_int = s.get8();
            z.img_comp[i].h = q >> 4;
            if (z.img_comp[i].h == 0 or z.img_comp[i].h > 4) return s.err("bad H") != 0;
            z.img_comp[i].v = q & 15;
            if (z.img_comp[i].v == 0 or z.img_comp[i].v > 4) return s.err("bad V") != 0;
            z.img_comp[i].tq = s.get8();
            if (z.img_comp[i].tq > 3) return s.err("bad TQ") != 0;
        }

        if (scan != .load) return true;

        if (!mad3sizesValid(@intCast(s.img_x), @intCast(s.img_y), s.img_n, 0)) return s.err("too large") != 0;

        i = 0;
        while (i < s.img_n) : (i += 1) {
            if (z.img_comp[i].h > h_max) h_max = z.img_comp[i].h;
            if (z.img_comp[i].v > v_max) v_max = z.img_comp[i].v;
        }

        // check that plane subsampling factors are integer ratios; our resamplers can't deal with fractional ratios
        i = 0;
        while (i < s.img_n) : (i += 1) {
            if (@rem(h_max, z.img_comp[i].h) != 0) return s.err("bad H") != 0;
            if (@rem(v_max, z.img_comp[i].v) != 0) return s.err("bad V") != 0;
        }

        // compute interleaved mcu info
        z.img_h_max = h_max;
        z.img_v_max = v_max;
        z.img_mcu_w = h_max * 8;
        z.img_mcu_h = v_max * 8;
        // these sizes can't be more than 17 bits
        z.img_mcu_x = @intCast((s.img_x + @as(u32, @intCast(z.img_mcu_w)) - 1) / @as(u32, @intCast(z.img_mcu_w)));
        z.img_mcu_y = @intCast((s.img_y + @as(u32, @intCast(z.img_mcu_h)) - 1) / @as(u32, @intCast(z.img_mcu_h)));

        i = 0;
        while (i < s.img_n) : (i += 1) {
            const comp = &z.img_comp[i];
            // number of effective pixels (e.g. for non-interleaved MCU)
            comp.x = @intCast((s.img_x * @as(u32, @intCast(comp.h)) + @as(u32, @intCast(h_max)) - 1) / @as(u32, @intCast(h_max)));
            comp.y = @intCast((s.img_y * @as(u32, @intCast(comp.v)) + @as(u32, @intCast(v_max)) - 1) / @as(u32, @intCast(v_max)));
            // allocate enough memory to decode the bogus oversized data from
            // using interleaved MCUs and their big blocks
            comp.w2 = z.img_mcu_x * comp.h * 8;
            comp.h2 = z.img_mcu_y * comp.v * 8;
            comp.coeff = null;
            comp.linebuf = null;
            comp.data = s.mallocMad2(comp.w2, comp.h2, 15);
            if (comp.data == null) {
                z.freeComponents(i + 1);
                return s.err("outofmem") != 0;
            }
            if (z.progressive) {
                // w2, h2 are multiples of 8 (see above)
                comp.coeff_w = @divTrunc(comp.w2, 8);
                comp.coeff_h = @divTrunc(comp.h2, 8);
                if (!mad3sizesValid(comp.w2, comp.h2, 2, 15)) {
                    z.freeComponents(i + 1);
                    return s.err("outofmem") != 0;
                }
                const nc: usize = @intCast(comp.w2 * comp.h2);
                const coeff = s.gpa.alloc(i16, nc) catch {
                    z.freeComponents(i + 1);
                    return s.err("outofmem") != 0;
                };
                @memset(coeff, 0);
                comp.coeff = coeff;
            }
        }
        return true;
    }

    /// Ports `stbi__decode_jpeg_header` (stb_image.h:3365 @5266f24da).
    fn decodeHeader(z: *Jpeg, scan: Scan) bool {
        z.jfif = false;
        z.app14_color_transform = -1; // valid values are 0,1,2
        z.marker = marker_none; // initialize cached marker to empty
        var m: c_int = z.getMarker();
        if (m != 0xd8) return z.s.err("no SOI") != 0;
        if (scan == .type) return true;
        m = z.getMarker();
        while (!isSof(m)) {
            if (!z.processMarker(m)) return false;
            m = z.getMarker();
            while (m == marker_none) {
                // some files have extra padding after their blocks, so ok, we'll scan
                if (z.s.atEof()) return z.s.err("no SOF") != 0;
                m = z.getMarker();
            }
        }
        z.progressive = m == 0xc2;
        if (!z.processFrameHeader(scan)) return false;
        return true;
    }

    /// Ports `stbi__skip_jpeg_junk_at_end` (stb_image.h:3389 @5266f24da).
    fn skipJunkAtEnd(j: *Jpeg) u8 {
        // some JPEGs have junk at end, skip over it but if we find what looks
        // like a valid marker, resume there
        while (!j.s.atEof()) {
            var x = j.s.get8();
            while (x == 0xff) { // might be a marker
                if (j.s.atEof()) return marker_none;
                x = j.s.get8();
                if (x != 0x00 and x != 0xff) {
                    // not a stuffed zero or lead-in to another marker, looks
                    // like an actual marker, return it
                    return x;
                }
            }
        }
        return marker_none;
    }

    /// Ports `stbi__decode_jpeg_image` (stb_image.h:3412 @5266f24da): decode
    /// to YCbCr.
    fn decodeImage(j: *Jpeg) bool {
        for (&j.img_comp) |*c| {
            c.data = null;
            c.coeff = null;
        }
        j.restart_interval = 0;
        if (!j.decodeHeader(.load)) return false;
        var m: c_int = j.getMarker();
        while (m != 0xd9) {
            if (m == 0xda) {
                if (!j.processScanHeader()) return false;
                if (!j.parseEntropyCodedData()) return false;
                if (j.marker == marker_none) {
                    j.marker = j.skipJunkAtEnd();
                    // if we reach eof without hitting a marker, getMarker() below will fail and we'll eventually return 0
                }
                m = j.getMarker();
                if (isRestart(m)) m = j.getMarker();
            } else if (m == 0xdc) {
                const ld = j.s.get16be();
                const nl: u32 = @intCast(j.s.get16be());
                if (ld != 4) return j.s.err("bad DNL len") != 0;
                if (nl != j.s.img_y) return j.s.err("bad DNL height") != 0;
                m = j.getMarker();
            } else {
                if (!j.processMarker(m)) return true;
                m = j.getMarker();
            }
        }
        if (j.progressive) j.finish();
        return true;
    }
};

const Scan = enum { load, type, header };

const marker_none: u8 = 0xff;

fn isRestart(x: c_int) bool {
    return x >= 0xd0 and x <= 0xd7;
}

fn isSof(x: c_int) bool {
    return x == 0xc0 or x == 0xc1 or x == 0xc2;
}

/// `x << n`, where `n` may be 32: a shift the C never makes, kept defined.
fn shl32(x: u32, n: c_int) u32 {
    return if (n >= 32) 0 else x << @intCast(n);
}

/// `stbi__bmask` (stb_image.h:2094 @5266f24da): (1 << n) - 1.
const bmask = [17]u32{ 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535 };

/// `stbi__jbias` (stb_image.h:2149 @5266f24da): bias[n] = (-1<<n) + 1.
const jbias = [16]c_int{ 0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047, -4095, -8191, -16383, -32767 };

/// `stbi__jpeg_dezigzag` (stb_image.h:2194 @5266f24da), with the padding
/// that lets corrupt input sample past the end.
const jpeg_dezigzag = [64 + 15]u8{
    0,  1,  8,  16, 9,  2,  3,  10,
    17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
    // let corrupt input sample past end
    63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63,
};

/// Ports `stbi__build_huffman` (stb_image.h:2003 @5266f24da).
fn buildHuffman(s: *Context, h: *Huffman, count: *const [16]c_int) bool {
    var k: usize = 0;
    // build size list for each symbol (from JPEG spec)
    for (0..16) |i| {
        var j: c_int = 0;
        while (j < count[i]) : (j += 1) {
            h.size[k] = @intCast(i + 1);
            k += 1;
            if (k >= 257) return s.err("bad size list") != 0;
        }
    }
    h.size[k] = 0;

    // compute actual symbols (from jpeg spec)
    var code: c_uint = 0;
    k = 0;
    var j: usize = 1;
    while (j <= 16) : (j += 1) {
        // compute delta to add to code to compute symbol id
        h.delta[j] = @as(c_int, @intCast(k)) -% @as(c_int, @bitCast(code));
        if (h.size[k] == j) {
            while (h.size[k] == j) {
                h.code[k] = @truncate(code);
                k += 1;
                code += 1;
            }
            if (code - 1 >= (@as(c_uint, 1) << @intCast(j))) return s.err("bad code lengths") != 0;
        }
        // compute largest code + 1 for this size, preshifted as needed later
        h.maxcode[j] = code << @intCast(16 - j);
        code <<= 1;
    }
    h.maxcode[j] = 0xffffffff;

    // build non-spec acceleration table; 255 is flag for not-accelerated
    h.fast = @splat(255);
    for (0..k) |i| {
        const sz: usize = h.size[i];
        if (sz <= fast_bits) {
            const c = @as(usize, h.code[i]) << @intCast(fast_bits - sz);
            const m = @as(usize, 1) << @intCast(fast_bits - sz);
            for (0..m) |jj| h.fast[c + jj] = @intCast(i);
        }
    }
    return true;
}

/// Ports `stbi__build_fast_ac` (stb_image.h:2050 @5266f24da): a table that
/// decodes both magnitude and value of small ACs in one go.
fn buildFastAc(fast_ac: *[1 << fast_bits]i16, h: *const Huffman) void {
    for (0..(1 << fast_bits)) |i| {
        const fast = h.fast[i];
        fast_ac[i] = 0;
        if (fast < 255) {
            const rs: c_int = h.values[fast];
            const run = (rs >> 4) & 15;
            const magbits = rs & 15;
            const len: c_int = h.size[fast];

            if (magbits != 0 and len + magbits <= fast_bits) {
                // magnitude code followed by receive_extend code
                var k: c_int = ((@as(c_int, @intCast(i)) << @intCast(len)) & ((1 << fast_bits) - 1)) >> @intCast(fast_bits - magbits);
                const m = @as(c_int, 1) << @intCast(magbits - 1);
                if (k < m) k +%= @bitCast((~@as(c_uint, 0) << @intCast(magbits)) +% 1);
                // if the result is small enough, we can fit it in fast_ac table
                if (k >= -128 and k <= 127)
                    fast_ac[i] = @intCast((k * 256) + (run * 16) + (len + magbits));
            }
        }
    }
}

/// Ports `stbi__clamp` (stb_image.h:2416 @5266f24da).
fn clamp(x: c_int) u8 {
    if (x < 0) return 0;
    if (x > 255) return 255;
    return @intCast(x);
}

/// `stbi__f2f` (stb_image.h:2426 @5266f24da): `x * 4096` in float, then
/// `+ 0.5` in double.
fn f2f(comptime x: f32) c_int {
    return @intFromFloat(@as(f64, x * 4096) + 0.5);
}

/// The products of `STBI__IDCT_1D` (stb_image.h:2430 @5266f24da).
const Idct = struct { t0: c_int, t1: c_int, t2: c_int, t3: c_int, x0: c_int, x1: c_int, x2: c_int, x3: c_int };

/// Ports `STBI__IDCT_1D` (stb_image.h:2430 @5266f24da), derived from
/// jidctint's DCT_ISLOW. Wrapping, as corrupt coefficients overflow the C's
/// `int`.
fn idct1d(s0: c_int, s1: c_int, s2: c_int, s3: c_int, s4: c_int, s5: c_int, s6: c_int, s7: c_int) Idct {
    var p2 = s2;
    var p3 = s6;
    var p1 = (p2 +% p3) *% f2f(0.5411961);
    var t2 = p1 +% p3 *% f2f(-1.847759065);
    var t3 = p1 +% p2 *% f2f(0.765366865);
    p2 = s0;
    p3 = s4;
    var t0 = (p2 +% p3) *% 4096;
    var t1 = (p2 -% p3) *% 4096;
    const x0 = t0 +% t3;
    const x3 = t0 -% t3;
    const x1 = t1 +% t2;
    const x2 = t1 -% t2;
    t0 = s7;
    t1 = s5;
    t2 = s3;
    t3 = s1;
    p3 = t0 +% t2;
    var p4 = t1 +% t3;
    p1 = t0 +% t3;
    p2 = t1 +% t2;
    const p5 = (p3 +% p4) *% f2f(1.175875602);
    t0 = t0 *% f2f(0.298631336);
    t1 = t1 *% f2f(2.053119869);
    t2 = t2 *% f2f(3.072711026);
    t3 = t3 *% f2f(1.501321110);
    p1 = p5 +% p1 *% f2f(-0.899976223);
    p2 = p5 +% p2 *% f2f(-2.562915447);
    p3 = p3 *% f2f(-1.961570560);
    p4 = p4 *% f2f(-0.390180644);
    t3 +%= p1 +% p4;
    t2 +%= p2 +% p3;
    t1 +%= p2 +% p4;
    t0 +%= p1 +% p3;
    return .{ .t0 = t0, .t1 = t1, .t2 = t2, .t3 = t3, .x0 = x0, .x1 = x1, .x2 = x2, .x3 = x3 };
}

/// Ports `stbi__idct_block` (stb_image.h:2467 @5266f24da).
fn idctBlock(out: [*]u8, out_stride: usize, data: *const [64]i16) void {
    var val: [64]c_int = undefined;

    // columns
    for (0..8) |i| {
        const d = data[i..];
        const v = val[i..];
        // if all zeroes, shortcut -- this avoids dequantizing 0s and IDCTing
        if (d[8] == 0 and d[16] == 0 and d[24] == 0 and d[32] == 0 and d[40] == 0 and d[48] == 0 and d[56] == 0) {
            const dcterm = @as(c_int, d[0]) * 4;
            inline for (.{ 0, 8, 16, 24, 32, 40, 48, 56 }) |o| v[o] = dcterm;
        } else {
            var r = idct1d(d[0], d[8], d[16], d[24], d[32], d[40], d[48], d[56]);
            // constants scaled things up by 1<<12; let's bring them back
            // down, but keep 2 extra bits of precision
            r.x0 +%= 512;
            r.x1 +%= 512;
            r.x2 +%= 512;
            r.x3 +%= 512;
            v[0] = (r.x0 +% r.t3) >> 10;
            v[56] = (r.x0 -% r.t3) >> 10;
            v[8] = (r.x1 +% r.t2) >> 10;
            v[48] = (r.x1 -% r.t2) >> 10;
            v[16] = (r.x2 +% r.t1) >> 10;
            v[40] = (r.x2 -% r.t1) >> 10;
            v[24] = (r.x3 +% r.t0) >> 10;
            v[32] = (r.x3 -% r.t0) >> 10;
        }
    }

    for (0..8) |i| {
        const v = val[i * 8 ..];
        const o = out + i * out_stride;
        // no fast case since the first 1D IDCT spread components out
        var r = idct1d(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
        // round off the 1<<17 total scale, and add 128 before the shift
        r.x0 +%= 65536 + (128 << 17);
        r.x1 +%= 65536 + (128 << 17);
        r.x2 +%= 65536 + (128 << 17);
        r.x3 +%= 65536 + (128 << 17);
        o[0] = clamp((r.x0 +% r.t3) >> 17);
        o[7] = clamp((r.x0 -% r.t3) >> 17);
        o[1] = clamp((r.x1 +% r.t2) >> 17);
        o[6] = clamp((r.x1 -% r.t2) >> 17);
        o[2] = clamp((r.x2 +% r.t1) >> 17);
        o[5] = clamp((r.x2 -% r.t1) >> 17);
        o[3] = clamp((r.x3 +% r.t0) >> 17);
        o[4] = clamp((r.x3 -% r.t0) >> 17);
    }
}

const ResampleKind = enum { row1, v2, h2, hv2, generic };

/// Ports `stbi__resample` (stb_image.h:3856 @5266f24da); the lines are
/// offsets into the component's data.
const Resample = struct {
    kind: ResampleKind,
    line0: usize,
    line1: usize,
    hs: c_int, // expansion factor in each axis
    vs: c_int,
    w_lores: c_int, // horizontal pixels pre-expansion
    ystep: c_int, // how far through vertical expansion we are
    ypos: c_int, // which pre-expansion row we're on
};

fn div4(x: c_int) u8 {
    return @truncate(@as(c_uint, @bitCast(x >> 2)));
}

fn div16(x: c_int) u8 {
    return @truncate(@as(c_uint, @bitCast(x >> 4)));
}

/// Ports `resample_row_1`, `stbi__resample_row_v_2`, `stbi__resample_row_h_2`,
/// `stbi__resample_row_hv_2` and `stbi__resample_row_generic`
/// (stb_image.h:3456, 3465, 3475, 3505, 3646 @5266f24da).
///
/// Return: the row, `in_near` itself for `row1`, else `out`.
fn resampleRow(kind: ResampleKind, out: []u8, in_near: []const u8, in_far: []const u8, w: usize, hs: usize) []const u8 {
    switch (kind) {
        .row1 => return in_near,
        .v2 => {
            // need to generate two samples vertically for every one in input
            for (0..w) |i| out[i] = div4(3 * @as(c_int, in_near[i]) + in_far[i] + 2);
            return out;
        },
        .h2 => {
            // need to generate two samples horizontally for every one in input
            const input = in_near;
            if (w == 1) {
                // if only one sample, can't do any interpolation
                out[0] = input[0];
                out[1] = input[0];
                return out;
            }
            out[0] = input[0];
            out[1] = div4(@as(c_int, input[0]) * 3 + input[1] + 2);
            var i: usize = 1;
            while (i < w - 1) : (i += 1) {
                const n = 3 * @as(c_int, input[i]) + 2;
                out[i * 2 + 0] = div4(n + input[i - 1]);
                out[i * 2 + 1] = div4(n + input[i + 1]);
            }
            out[i * 2 + 0] = div4(@as(c_int, input[w - 2]) * 3 + input[w - 1] + 2);
            out[i * 2 + 1] = input[w - 1];
            return out;
        },
        .hv2 => {
            // need to generate 2x2 samples for every one in input
            if (w == 1) {
                const v = div4(3 * @as(c_int, in_near[0]) + in_far[0] + 2);
                out[0] = v;
                out[1] = v;
                return out;
            }
            var t1: c_int = 3 * @as(c_int, in_near[0]) + in_far[0];
            out[0] = div4(t1 + 2);
            for (1..w) |i| {
                const t0 = t1;
                t1 = 3 * @as(c_int, in_near[i]) + in_far[i];
                out[i * 2 - 1] = div16(3 * t0 + t1 + 8);
                out[i * 2] = div16(3 * t1 + t0 + 8);
            }
            out[w * 2 - 1] = div4(t1 + 2);
            return out;
        },
        .generic => {
            // resample with nearest-neighbor
            for (0..w) |i| {
                for (0..hs) |j| out[i * hs + j] = in_near[i];
            }
            return out;
        },
    }
}

/// `stbi__float2fixed` (stb_image.h:3659 @5266f24da) of the four YCbCr
/// constants: `x * 4096.0f + 0.5f`, truncated, shifted left by 8.
const fixed_1_40200: c_int = 5743 << 8;
const fixed_0_71414: c_int = 2925 << 8;
const fixed_0_34414: c_int = 1410 << 8;
const fixed_1_77200: c_int = 7258 << 8;

/// Ports `stbi__YCbCr_to_RGB_row` (stb_image.h:3660 @5266f24da): the
/// reduced-precision conversion the SIMD kernels match.
fn ycbcrToRgbRow(out: []u8, y: []const u8, pcb: []const u8, pcr: []const u8, count: usize, step: usize) void {
    var o: usize = 0;
    for (0..count) |i| {
        const y_fixed = (@as(c_int, y[i]) << 20) + (1 << 19); // rounding
        const cr = @as(c_int, pcr[i]) - 128;
        const cb = @as(c_int, pcb[i]) - 128;
        var r = y_fixed + cr * fixed_1_40200;
        var g = y_fixed + (cr * -fixed_0_71414) + ((cb * -fixed_0_34414) & @as(c_int, @bitCast(@as(c_uint, 0xffff0000))));
        var b = y_fixed + cb * fixed_1_77200;
        r >>= 20;
        g >>= 20;
        b >>= 20;
        out[o + 0] = clamp(r);
        out[o + 1] = clamp(g);
        out[o + 2] = clamp(b);
        out[o + 3] = 255;
        o += step;
    }
}

/// Ports `stbi__blinn_8x8` (stb_image.h:3859 @5266f24da): a rounded
/// 0..255 * 0..255 multiplication.
fn blinn8x8(x: u8, y: u8) u8 {
    const t: c_uint = @as(c_uint, x) * y + 128;
    return @truncate((t + (t >> 8)) >> 8);
}

/// Ports `load_jpeg_image` (stb_image.h:3865 @5266f24da).
fn loadJpegImage(z: *Jpeg, out_x: *c_int, out_y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
    const s = z.s;
    s.img_n = 0; // make cleanup safe

    // validate req_comp
    if (req_comp < 0 or req_comp > 4) return s.errpuc("bad req_comp");

    // load a jpeg image from whichever source, but leave in YCbCr format
    if (!z.decodeImage()) {
        z.freeComponents(@intCast(s.img_n));
        return null;
    }
    defer z.freeComponents(@intCast(s.img_n));

    // determine actual number of components to generate
    const n: c_int = if (req_comp != 0) req_comp else if (s.img_n >= 3) 3 else 1;

    const is_rgb = s.img_n == 3 and (z.rgb == 3 or (z.app14_color_transform == 0 and !z.jfif));

    const decode_n: c_int = if (s.img_n == 3 and n < 3 and !is_rgb) 1 else s.img_n;

    // nothing to do if no components requested
    if (decode_n <= 0) return null;

    var res_comp: [4]Resample = undefined;
    const dn: usize = @intCast(decode_n);
    for (0..dn) |k| {
        const r = &res_comp[k];
        // allocate line buffer big enough for upsampling off the edges
        // with upsample factor of 4
        z.img_comp[k].linebuf = s.alloc(s.img_x + 3) orelse return s.errpuc("outofmem");

        r.hs = @divTrunc(z.img_h_max, z.img_comp[k].h);
        r.vs = @divTrunc(z.img_v_max, z.img_comp[k].v);
        r.ystep = r.vs >> 1;
        r.w_lores = @intCast((s.img_x + @as(u32, @intCast(r.hs)) - 1) / @as(u32, @intCast(r.hs)));
        r.ypos = 0;
        r.line0 = 0;
        r.line1 = 0;

        r.kind = if (r.hs == 1 and r.vs == 1) .row1 else if (r.hs == 1 and r.vs == 2) .v2 else if (r.hs == 2 and r.vs == 1) .h2 else if (r.hs == 2 and r.vs == 2) .hv2 else .generic;
    }

    // can't error after this so, this is safe
    const output = s.mallocMad3(n, @intCast(s.img_x), @intCast(s.img_y), 1) orelse return s.errpuc("outofmem");

    const nu: usize = @intCast(n);
    var coutput: [4][]const u8 = undefined;
    // now go ahead and resample
    for (0..s.img_y) |j| {
        const out = output[nu * s.img_x * j ..];
        for (0..dn) |k| {
            const r = &res_comp[k];
            const data = z.img_comp[k].data.?;
            const y_bot = r.ystep >= (r.vs >> 1);
            coutput[k] = resampleRow(r.kind, z.img_comp[k].linebuf.?, data[if (y_bot) r.line1 else r.line0..], data[if (y_bot) r.line0 else r.line1..], @intCast(r.w_lores), @intCast(r.hs));
            r.ystep += 1;
            if (r.ystep >= r.vs) {
                r.ystep = 0;
                r.line0 = r.line1;
                r.ypos += 1;
                if (r.ypos < z.img_comp[k].y) r.line1 += @intCast(z.img_comp[k].w2);
            }
        }
        const xn: usize = s.img_x;
        if (n >= 3) {
            const y = coutput[0];
            if (s.img_n == 3) {
                if (is_rgb) {
                    var o: usize = 0;
                    for (0..xn) |i| {
                        out[o + 0] = y[i];
                        out[o + 1] = coutput[1][i];
                        out[o + 2] = coutput[2][i];
                        out[o + 3] = 255;
                        o += nu;
                    }
                } else {
                    ycbcrToRgbRow(out, y, coutput[1], coutput[2], xn, nu);
                }
            } else if (s.img_n == 4) {
                if (z.app14_color_transform == 0) { // CMYK
                    var o: usize = 0;
                    for (0..xn) |i| {
                        const m = coutput[3][i];
                        out[o + 0] = blinn8x8(coutput[0][i], m);
                        out[o + 1] = blinn8x8(coutput[1][i], m);
                        out[o + 2] = blinn8x8(coutput[2][i], m);
                        out[o + 3] = 255;
                        o += nu;
                    }
                } else if (z.app14_color_transform == 2) { // YCCK
                    ycbcrToRgbRow(out, y, coutput[1], coutput[2], xn, nu);
                    var o: usize = 0;
                    for (0..xn) |i| {
                        const m = coutput[3][i];
                        out[o + 0] = blinn8x8(255 - out[o + 0], m);
                        out[o + 1] = blinn8x8(255 - out[o + 1], m);
                        out[o + 2] = blinn8x8(255 - out[o + 2], m);
                        o += nu;
                    }
                } else { // YCbCr + alpha?  Ignore the fourth channel for now
                    ycbcrToRgbRow(out, y, coutput[1], coutput[2], xn, nu);
                }
            } else {
                var o: usize = 0;
                for (0..xn) |i| {
                    out[o + 0] = y[i];
                    out[o + 1] = y[i];
                    out[o + 2] = y[i];
                    out[o + 3] = 255; // not used if n==3
                    o += nu;
                }
            }
        } else {
            if (is_rgb) {
                if (n == 1) {
                    for (0..xn) |i| out[i] = computeY(coutput[0][i], coutput[1][i], coutput[2][i]);
                } else {
                    for (0..xn) |i| {
                        out[i * 2 + 0] = computeY(coutput[0][i], coutput[1][i], coutput[2][i]);
                        out[i * 2 + 1] = 255;
                    }
                }
            } else if (s.img_n == 4 and z.app14_color_transform == 0) {
                var o: usize = 0;
                for (0..xn) |i| {
                    const m = coutput[3][i];
                    const r = blinn8x8(coutput[0][i], m);
                    const g = blinn8x8(coutput[1][i], m);
                    const b = blinn8x8(coutput[2][i], m);
                    out[o + 0] = computeY(r, g, b);
                    out[o + 1] = 255;
                    o += nu;
                }
            } else if (s.img_n == 4 and z.app14_color_transform == 2) {
                var o: usize = 0;
                for (0..xn) |i| {
                    out[o + 0] = blinn8x8(255 - coutput[0][i], coutput[3][i]);
                    out[o + 1] = 255;
                    o += nu;
                }
            } else {
                const y = coutput[0];
                if (n == 1) {
                    for (0..xn) |i| out[i] = y[i];
                } else {
                    for (0..xn) |i| {
                        out[i * 2] = y[i];
                        out[i * 2 + 1] = 255;
                    }
                }
            }
        }
    }
    out_x.* = @intCast(s.img_x);
    out_y.* = @intCast(s.img_y);
    comp.* = if (s.img_n >= 3) 3 else 1; // report original components, not output
    // the extra byte took the alpha a 3-component row writes past its last
    // pixel; the image is the rest
    return s.gpa.realloc(output, output.len - 1) catch output[0 .. output.len - 1];
}

fn jpegNew(s: *Context) ?*Jpeg {
    const j = s.gpa.create(Jpeg) catch return null;
    j.* = .{ .s = s };
    return j;
}

/// Ports `stbi__jpeg_test` (stb_image.h:4042 @5266f24da).
fn jpegTestImpl(s: *Context) bool {
    const j = jpegNew(s) orelse return s.err("outofmem") != 0;
    defer s.gpa.destroy(j);
    const r = j.decodeHeader(.type);
    s.rewind();
    return r;
}

// -----------------------------------------------------------------------------
// zlib

const zfast_bits = 9;
const zfast_mask = (1 << zfast_bits) - 1;
const znsyms = 288; // number of symbols in literal/length alphabet

/// Ports `stbi__zhuffman` (stb_image.h:4105 @5266f24da).
const Zhuffman = struct {
    fast: [1 << zfast_bits]u16 = @splat(0),
    firstcode: [16]u16 = @splat(0),
    maxcode: [17]c_int = @splat(0),
    firstsymbol: [16]u16 = @splat(0),
    size: [znsyms]u8 = @splat(0),
    value: [znsyms]u16 = @splat(0),
};

/// Ports `stbi__bitreverse16` and `stbi__bit_reverse` (stb_image.h:4107,
/// 4116 @5266f24da).
fn bitReverse(v: c_int, bits: c_int) c_int {
    var n = v;
    n = ((n & 0xAAAA) >> 1) | ((n & 0x5555) << 1);
    n = ((n & 0xCCCC) >> 2) | ((n & 0x3333) << 2);
    n = ((n & 0xF0F0) >> 4) | ((n & 0x0F0F) << 4);
    n = ((n & 0xFF00) >> 8) | ((n & 0x00FF) << 8);
    return n >> @intCast(16 - bits);
}

/// Ports `stbi__zbuild_huffman` (stb_image.h:4124 @5266f24da).
fn zbuildHuffman(s: *Context, z: *Zhuffman, sizelist: []const u8) bool {
    var k: c_int = 0;
    var next_code: [16]c_int = undefined;
    var sizes: [17]c_int = @splat(0);

    // DEFLATE spec for generating codes
    z.fast = @splat(0);
    for (sizelist) |sz| sizes[sz] += 1;
    sizes[0] = 0;
    for (1..16) |i| {
        if (sizes[i] > (@as(c_int, 1) << @intCast(i))) return s.err("bad sizes") != 0;
    }
    var code: c_int = 0;
    for (1..16) |i| {
        next_code[i] = code;
        z.firstcode[i] = @truncate(@as(c_uint, @bitCast(code)));
        z.firstsymbol[i] = @truncate(@as(c_uint, @bitCast(k)));
        code = code + sizes[i];
        if (sizes[i] != 0) {
            if (code - 1 >= (@as(c_int, 1) << @intCast(i))) return s.err("bad codelengths") != 0;
        }
        z.maxcode[i] = code << @intCast(16 - i); // preshift for inner loop
        code <<= 1;
        k += sizes[i];
    }
    z.maxcode[16] = 0x10000; // sentinel
    for (sizelist, 0..) |sz, i| {
        if (sz != 0) {
            const c: usize = @intCast(next_code[sz] - z.firstcode[sz] + z.firstsymbol[sz]);
            const fastv: u16 = @truncate((@as(c_uint, sz) << 9) | @as(c_uint, @intCast(i)));
            z.size[c] = sz;
            z.value[c] = @intCast(i);
            if (sz <= zfast_bits) {
                var j = bitReverse(next_code[sz], sz);
                while (j < (1 << zfast_bits)) {
                    z.fast[@intCast(j)] = fastv;
                    j += @as(c_int, 1) << @intCast(sz);
                }
            }
            next_code[sz] += 1;
        }
    }
    return true;
}

/// Ports `stbi__zbuf` (stb_image.h:4190 @5266f24da).
const Zbuf = struct {
    s: *Context,
    zbuffer: []const u8,
    zpos: usize = 0,
    num_bits: c_int = 0,
    hit_zeof_once: bool = false,
    code_buffer: u32 = 0,

    zout: std.ArrayList(u8) = .empty,
    /// `zout_end - zout_start`: the capacity the C tracks and doubles.
    limit: usize = 0,
    z_expandable: bool = true,

    z_length: Zhuffman = .{},
    z_distance: Zhuffman = .{},

    /// Ports `stbi__zeof` (stb_image.h:4192 @5266f24da).
    fn zeof(a: *const Zbuf) bool {
        return a.zpos >= a.zbuffer.len;
    }

    /// Ports `stbi__zget8` (stb_image.h:4197 @5266f24da).
    fn zget8(a: *Zbuf) u8 {
        if (a.zeof()) return 0;
        const v = a.zbuffer[a.zpos];
        a.zpos += 1;
        return v;
    }

    /// Ports `stbi__fill_bits` (stb_image.h:4202 @5266f24da).
    fn fillBits(a: *Zbuf) void {
        while (true) {
            if (a.num_bits < 32 and a.code_buffer >= (@as(u32, 1) << @intCast(a.num_bits))) {
                a.zpos = a.zbuffer.len; // treat this as EOF so we fail.
                return;
            }
            a.code_buffer |= shl32(a.zget8(), a.num_bits);
            a.num_bits += 8;
            if (a.num_bits > 24) break;
        }
    }

    /// Ports `stbi__zreceive` (stb_image.h:4214 @5266f24da).
    fn zreceive(a: *Zbuf, n: c_int) c_uint {
        if (a.num_bits < n) a.fillBits();
        const k = a.code_buffer & ((@as(u32, 1) << @intCast(n)) -% 1);
        a.code_buffer = if (n >= 32) 0 else a.code_buffer >> @intCast(n);
        a.num_bits -= n;
        return k;
    }

    /// Ports `stbi__zhuffman_decode_slowpath` (stb_image.h:4224 @5266f24da).
    fn decodeSlowpath(a: *Zbuf, z: *const Zhuffman) c_int {
        // not resolved by fast table, so compute it the slow way
        const k = bitReverse(@bitCast(a.code_buffer & 0xffff), 16);
        var s: usize = zfast_bits + 1;
        while (true) : (s += 1) {
            if (k < z.maxcode[s]) break;
        }
        if (s >= 16) return -1; // invalid code!
        // code size is s, so:
        const b = (k >> @intCast(16 - s)) - z.firstcode[s] + z.firstsymbol[s];
        if (b >= znsyms) return -1; // some data was corrupt somewhere!
        if (b < 0) return -1;
        if (z.size[@intCast(b)] != s) return -1; // was originally an assert, but report failure instead.
        a.code_buffer >>= @intCast(s);
        a.num_bits -= @intCast(s);
        return z.value[@intCast(b)];
    }

    /// Ports `stbi__zhuffman_decode` (stb_image.h:4243 @5266f24da).
    fn decode(a: *Zbuf, z: *const Zhuffman) c_int {
        if (a.num_bits < 16) {
            if (a.zeof()) {
                if (!a.hit_zeof_once) {
                    // This is the first time we hit eof, insert 16 extra padding btis
                    // to allow us to keep going; if we actually consume any of them
                    // though, that is invalid data. This is caught later.
                    a.hit_zeof_once = true;
                    a.num_bits += 16; // add 16 implicit zero bits
                } else {
                    // We already inserted our extra 16 padding bits and are again
                    // out, this stream is actually prematurely terminated.
                    return -1;
                }
            } else {
                a.fillBits();
            }
        }
        const b: c_int = z.fast[a.code_buffer & zfast_mask];
        if (b != 0) {
            const s = b >> 9;
            a.code_buffer >>= @intCast(s);
            a.num_bits -= s;
            return b & 511;
        }
        return a.decodeSlowpath(z);
    }

    /// Ports `stbi__zexpand` (stb_image.h:4273 @5266f24da): make room for
    /// `n` more bytes, doubling the limit as the C does.
    fn zexpand(a: *Zbuf, n: usize) bool {
        if (!a.z_expandable) return a.s.err("output buffer limit") != 0;
        const cur = a.zout.items.len;
        var limit = a.limit;
        if (std.math.maxInt(c_uint) - cur < n) return a.s.err("outofmem") != 0;
        while (cur + n > limit) {
            if (limit > std.math.maxInt(c_uint) / 2) return a.s.err("outofmem") != 0;
            limit *= 2;
        }
        a.zout.ensureTotalCapacityPrecise(a.s.gpa, limit) catch return a.s.err("outofmem") != 0;
        a.limit = limit;
        return true;
    }

    /// Ports `stbi__parse_huffman_block` (stb_image.h:4309 @5266f24da).
    fn parseHuffmanBlock(a: *Zbuf) bool {
        while (true) {
            var z = a.decode(&a.z_length);
            if (z < 256) {
                if (z < 0) return a.s.err("bad huffman code") != 0; // error in huffman codes
                if (a.zout.items.len >= a.limit) {
                    if (!a.zexpand(1)) return false;
                }
                a.zout.appendAssumeCapacity(@intCast(z));
            } else {
                if (z == 256) {
                    if (a.hit_zeof_once and a.num_bits < 16) {
                        // the stream read past the end, so it is malformed
                        return a.s.err("unexpected end") != 0;
                    }
                    return true;
                }
                if (z >= 286) return a.s.err("bad huffman code") != 0; // per DEFLATE, length codes 286 and 287 must not appear in compressed data
                z -= 257;
                var len: usize = @intCast(zlength_base[@intCast(z)]);
                if (zlength_extra[@intCast(z)] != 0) len += a.zreceive(zlength_extra[@intCast(z)]);
                z = a.decode(&a.z_distance);
                if (z < 0 or z >= 30) return a.s.err("bad huffman code") != 0; // per DEFLATE, distance codes 30 and 31 must not appear in compressed data
                var dist: usize = @intCast(zdist_base[@intCast(z)]);
                if (zdist_extra[@intCast(z)] != 0) dist += a.zreceive(zdist_extra[@intCast(z)]);
                if (a.zout.items.len < dist) return a.s.err("bad dist") != 0;
                if (len > a.limit - a.zout.items.len) {
                    if (!a.zexpand(len)) return false;
                }
                var p = a.zout.items.len - dist;
                // run of one byte, or copy; byte by byte as the C, so overlaps repeat
                while (len > 0) : (len -= 1) {
                    a.zout.appendAssumeCapacity(a.zout.items[p]);
                    if (dist != 1) p += 1;
                }
            }
        }
    }

    /// Ports `stbi__compute_huffman_codes` (stb_image.h:4359 @5266f24da).
    fn computeHuffmanCodes(a: *Zbuf) bool {
        const length_dezigzag = [19]u8{ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
        var z_codelength: Zhuffman = .{};
        var lencodes: [286 + 32 + 137]u8 = undefined; //padding for maximum single op
        var codelength_sizes: [19]u8 = @splat(0);

        const hlit: usize = a.zreceive(5) + 257;
        const hdist: usize = a.zreceive(5) + 1;
        const hclen: usize = a.zreceive(4) + 4;
        const ntot = hlit + hdist;

        for (0..hclen) |i| {
            const s = a.zreceive(3);
            codelength_sizes[length_dezigzag[i]] = @intCast(s);
        }
        if (!zbuildHuffman(a.s, &z_codelength, &codelength_sizes)) return false;

        var n: usize = 0;
        while (n < ntot) {
            var c = a.decode(&z_codelength);
            if (c < 0 or c >= 19) return a.s.err("bad codelengths") != 0;
            if (c < 16) {
                lencodes[n] = @intCast(c);
                n += 1;
            } else {
                var fill: u8 = 0;
                if (c == 16) {
                    c = @as(c_int, @intCast(a.zreceive(2))) + 3;
                    if (n == 0) return a.s.err("bad codelengths") != 0;
                    fill = lencodes[n - 1];
                } else if (c == 17) {
                    c = @as(c_int, @intCast(a.zreceive(3))) + 3;
                } else if (c == 18) {
                    c = @as(c_int, @intCast(a.zreceive(7))) + 11;
                } else {
                    return a.s.err("bad codelengths") != 0;
                }
                const cu: usize = @intCast(c);
                if (ntot - n < cu) return a.s.err("bad codelengths") != 0;
                @memset(lencodes[n..][0..cu], fill);
                n += cu;
            }
        }
        if (n != ntot) return a.s.err("bad codelengths") != 0;
        if (!zbuildHuffman(a.s, &a.z_length, lencodes[0..hlit])) return false;
        if (!zbuildHuffman(a.s, &a.z_distance, lencodes[hlit..][0..hdist])) return false;
        return true;
    }

    /// Ports `stbi__parse_uncompressed_block` (stb_image.h:4409 @5266f24da).
    fn parseUncompressedBlock(a: *Zbuf) bool {
        var header: [4]u8 = undefined;
        if (a.num_bits & 7 != 0) _ = a.zreceive(a.num_bits & 7); // discard
        // drain the bit-packed data into header
        var k: usize = 0;
        while (a.num_bits > 0) {
            if (k < 4) header[k] = @truncate(a.code_buffer & 255);
            k += 1;
            a.code_buffer >>= 8;
            a.num_bits -= 8;
        }
        if (a.num_bits < 0) return a.s.err("zlib corrupt") != 0;
        // now fill header the normal way
        while (k < 4) {
            header[k] = a.zget8();
            k += 1;
        }
        const len = @as(usize, header[1]) * 256 + header[0];
        const nlen = @as(usize, header[3]) * 256 + header[2];
        if (nlen != (len ^ 0xffff)) return a.s.err("zlib corrupt") != 0;
        if (a.zpos + len > a.zbuffer.len) return a.s.err("read past buffer") != 0;
        if (a.zout.items.len + len > a.limit) {
            if (!a.zexpand(len)) return false;
        }
        a.zout.appendSliceAssumeCapacity(a.zbuffer[a.zpos..][0..len]);
        a.zpos += len;
        return true;
    }

    /// Ports `stbi__parse_zlib_header` (stb_image.h:4438 @5266f24da).
    fn parseZlibHeader(a: *Zbuf) bool {
        const cmf: c_int = a.zget8();
        const cm = cmf & 15;
        const flg: c_int = a.zget8();
        if (a.zeof()) return a.s.err("bad zlib header") != 0; // zlib spec
        if (@rem(cmf * 256 + flg, 31) != 0) return a.s.err("bad zlib header") != 0; // zlib spec
        if (flg & 32 != 0) return a.s.err("no preset dict") != 0; // preset dictionary not allowed in png
        if (cm != 8) return a.s.err("bad compression") != 0; // DEFLATE required for png
        return true;
    }

    /// Ports `stbi__parse_zlib` (stb_image.h:4481 @5266f24da).
    fn parseZlib(a: *Zbuf, parse_header: bool) bool {
        if (parse_header) {
            if (!a.parseZlibHeader()) return false;
        }
        a.num_bits = 0;
        a.code_buffer = 0;
        a.hit_zeof_once = false;
        while (true) {
            const final = a.zreceive(1);
            const kind = a.zreceive(2);
            if (kind == 0) {
                if (!a.parseUncompressedBlock()) return false;
            } else if (kind == 3) {
                return false;
            } else {
                if (kind == 1) {
                    // use fixed code lengths
                    if (!zbuildHuffman(a.s, &a.z_length, &zdefault_length)) return false;
                    if (!zbuildHuffman(a.s, &a.z_distance, &zdefault_distance)) return false;
                } else {
                    if (!a.computeHuffmanCodes()) return false;
                }
                if (!a.parseHuffmanBlock()) return false;
            }
            if (final != 0) break;
        }
        return true;
    }
};

/// `stbi__zlength_base`, `stbi__zlength_extra`, `stbi__zdist_base` and `stbi__zdist_extra` (stb_image.h:4295, 4300, 4303, 4306 @5266f24da).
const zlength_base = [31]c_int{ 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0, 0 };
const zlength_extra = [31]c_int{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0 };
const zdist_base = [32]c_int{ 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0, 0 };
const zdist_extra = [32]c_int{ 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 0, 0 };

/// `stbi__zdefault_length` and `stbi__zdefault_distance` (stb_image.h:4452,
/// 4464 @5266f24da).
const zdefault_length: [znsyms]u8 = blk: {
    var t: [znsyms]u8 = undefined;
    for (0..144) |i| t[i] = 8;
    for (144..256) |i| t[i] = 9;
    for (256..280) |i| t[i] = 7;
    for (280..288) |i| t[i] = 8;
    break :blk t;
};
const zdefault_distance: [32]u8 = @splat(5);

/// Ports `stbi_zlib_decode_malloc_guesssize_headerflag` (stb_image.h:4541
/// @5266f24da). Return: the inflated bytes, owned by the caller.
fn zlibDecodeGuesssizeHeaderflag(s: *Context, buffer: []const u8, initial_size: usize, parse_header: bool) ?[]u8 {
    var a: Zbuf = .{ .s = s, .zbuffer = buffer, .limit = initial_size };
    a.zout.ensureTotalCapacityPrecise(s.gpa, initial_size) catch return null;
    if (a.parseZlib(parse_header)) return a.zout.toOwnedSlice(s.gpa) catch null;
    a.zout.deinit(s.gpa);
    return null;
}

// -----------------------------------------------------------------------------
// PNG

/// The PNG decoder's state, `stbi__png` (stb_image.h:4635 @5266f24da).
const Png = struct {
    s: *Context,
    idata: ?[]u8 = null,
    expanded: ?[]u8 = null,
    out: ?[]u8 = null,
    depth: c_int = 0,
};

/// The filter types from `STBI__F_none` (stb_image.h:4639 @5266f24da).
const f_none = 0;
const f_sub = 1;
const f_up = 2;
const f_avg = 3;
const f_paeth = 4;
const f_avg_first = 5; // synthetic filter used for first scanline to avoid needing a dummy row of 0s

/// `first_row_filter` (stb_image.h:4648 @5266f24da).
const first_row_filter = [5]u8{ f_none, f_sub, f_none, f_avg_first, f_sub };

/// Ports `stbi__paeth` (stb_image.h:4657 @5266f24da).
fn paeth(a: c_int, b: c_int, c: c_int) c_int {
    const thresh = c * 3 - (a + b);
    const lo = if (a < b) a else b;
    const hi = if (a < b) b else a;
    const t0 = if (hi <= thresh) lo else c;
    const t1 = if (thresh <= lo) hi else t0;
    return t1;
}

/// `stbi__depth_scale_table` (stb_image.h:4670 @5266f24da).
const depth_scale_table = [9]u8{ 0, 0xff, 0x55, 0, 0x11, 0, 0, 0, 0x01 };

/// Ports `stbi__create_png_alpha_expand8` (stb_image.h:4675 @5266f24da): add
/// an all-255 alpha channel; `dest` may be `src`, so it runs backwards.
fn createPngAlphaExpand8(dest: []u8, src: []const u8, x: usize, img_n: c_int) void {
    var i = x;
    if (img_n == 1) {
        while (i > 0) {
            i -= 1;
            dest[i * 2 + 1] = 255;
            dest[i * 2 + 0] = src[i];
        }
    } else {
        while (i > 0) {
            i -= 1;
            dest[i * 4 + 3] = 255;
            dest[i * 4 + 2] = src[i * 3 + 2];
            dest[i * 4 + 1] = src[i * 3 + 1];
            dest[i * 4 + 0] = src[i * 3 + 0];
        }
    }
}

/// Ports `stbi__create_png_image_raw` (stb_image.h:4696 @5266f24da).
fn createPngImageRaw(a: *Png, raw_in: []const u8, out_n: c_int, x: u32, y: u32, depth: c_int, color: c_int) bool {
    const s = a.s;
    const bytes: u32 = if (depth == 16) 2 else 1;
    const stride: usize = x * @as(u32, @intCast(out_n)) * bytes;
    const img_n = s.img_n; // copy it into a local for later

    const output_bytes: c_int = out_n * @as(c_int, @intCast(bytes));
    var filter_bytes: usize = @as(usize, @intCast(img_n)) * bytes;
    var width: usize = x;

    a.out = s.mallocMad3(@bitCast(x), @bitCast(y), output_bytes, 0) orelse return s.err("outofmem") != 0;

    if (!mad3sizesValid(img_n, @bitCast(x), depth, 7)) return s.err("too large") != 0;
    const img_width_bytes: u32 = ((@as(u32, @intCast(img_n)) * x * @as(u32, @intCast(depth))) + 7) >> 3;
    if (!mad2sizesValid(@bitCast(img_width_bytes), @bitCast(y), @bitCast(img_width_bytes))) return s.err("too large") != 0;
    const img_len: u32 = (img_width_bytes + 1) * y;

    // just check for raw_len < img_len always
    if (raw_in.len < img_len) return s.err("not enough pixels") != 0;

    // Allocate two scan lines worth of filter workspace buffer.
    const filter_buf = s.mallocMad2(@bitCast(img_width_bytes), 2, 0) orelse return s.err("outofmem") != 0;
    defer s.gpa.free(filter_buf);

    // Filtering for low-bit-depth images
    if (depth < 8) {
        filter_bytes = 1;
        width = img_width_bytes;
    }

    var raw: usize = 0;
    var j: u32 = 0;
    while (j < y) : (j += 1) {
        // cur/prior filter buffers alternate
        const cur = filter_buf[(j & 1) * img_width_bytes ..][0..img_width_bytes];
        const prior = filter_buf[(~j & 1) * img_width_bytes ..][0..img_width_bytes];
        const dest = a.out.?[stride * j ..];
        const nk = width * filter_bytes;
        var filter: u8 = raw_in[raw];
        raw += 1;

        // check filter type
        if (filter > 4) return s.err("invalid filter") != 0;

        // if first row, use special filter that doesn't sample previous row
        if (j == 0) filter = first_row_filter[filter];

        const r = raw_in[raw..];
        // perform actual filtering
        switch (filter) {
            f_none => @memcpy(cur[0..nk], r[0..nk]),
            f_sub => {
                @memcpy(cur[0..filter_bytes], r[0..filter_bytes]);
                for (filter_bytes..nk) |k| cur[k] = r[k] +% cur[k - filter_bytes];
            },
            f_up => {
                for (0..nk) |k| cur[k] = r[k] +% prior[k];
            },
            f_avg => {
                for (0..filter_bytes) |k| cur[k] = r[k] +% (prior[k] >> 1);
                for (filter_bytes..nk) |k| cur[k] = r[k] +% @as(u8, @truncate((@as(c_uint, prior[k]) + cur[k - filter_bytes]) >> 1));
            },
            f_paeth => {
                for (0..filter_bytes) |k| cur[k] = r[k] +% prior[k]; // prior[k] == stbi__paeth(0,prior[k],0)
                for (filter_bytes..nk) |k| cur[k] = r[k] +% @as(u8, @truncate(@as(c_uint, @bitCast(paeth(cur[k - filter_bytes], prior[k], prior[k - filter_bytes])))));
            },
            f_avg_first => {
                @memcpy(cur[0..filter_bytes], r[0..filter_bytes]);
                for (filter_bytes..nk) |k| cur[k] = r[k] +% (cur[k - filter_bytes] >> 1);
            },
            else => {},
        }

        raw += nk;

        // expand decoded bits in cur to dest, also adding an extra alpha channel if desired
        if (depth < 8) {
            const scale: u8 = if (color == 0) depth_scale_table[@intCast(depth)] else 1; // scale grayscale values to 0..255 range
            var in: usize = 0;
            var inb: u8 = 0;
            const nsmp: usize = x * @as(u32, @intCast(img_n));

            // expand bits to bytes first
            if (depth == 4) {
                for (0..nsmp) |i| {
                    if (i & 1 == 0) {
                        inb = cur[in];
                        in += 1;
                    }
                    dest[i] = scale *% (inb >> 4);
                    inb <<= 4;
                }
            } else if (depth == 2) {
                for (0..nsmp) |i| {
                    if (i & 3 == 0) {
                        inb = cur[in];
                        in += 1;
                    }
                    dest[i] = scale *% (inb >> 6);
                    inb <<= 2;
                }
            } else {
                for (0..nsmp) |i| {
                    if (i & 7 == 0) {
                        inb = cur[in];
                        in += 1;
                    }
                    dest[i] = scale *% (inb >> 7);
                    inb <<= 1;
                }
            }

            // insert alpha=255 values if desired
            if (img_n != out_n) createPngAlphaExpand8(dest, dest, x, img_n);
        } else if (depth == 8) {
            if (img_n == out_n) {
                @memcpy(dest[0 .. x * @as(u32, @intCast(img_n))], cur[0 .. x * @as(u32, @intCast(img_n))]);
            } else {
                createPngAlphaExpand8(dest, cur, x, img_n);
            }
        } else if (depth == 16) {
            // convert the image data from big-endian to platform-native
            const nsmp: usize = x * @as(u32, @intCast(img_n));
            if (img_n == out_n) {
                for (0..nsmp) |i| wr16(dest, i, (@as(u16, cur[i * 2]) << 8) | cur[i * 2 + 1]);
            } else if (img_n == 1) {
                for (0..x) |i| {
                    wr16(dest, i * 2, (@as(u16, cur[i * 2]) << 8) | cur[i * 2 + 1]);
                    wr16(dest, i * 2 + 1, 0xffff);
                }
            } else {
                for (0..x) |i| {
                    wr16(dest, i * 4, (@as(u16, cur[i * 6]) << 8) | cur[i * 6 + 1]);
                    wr16(dest, i * 4 + 1, (@as(u16, cur[i * 6 + 2]) << 8) | cur[i * 6 + 3]);
                    wr16(dest, i * 4 + 2, (@as(u16, cur[i * 6 + 4]) << 8) | cur[i * 6 + 5]);
                    wr16(dest, i * 4 + 3, 0xffff);
                }
            }
        }
    }
    return true;
}

/// Ports `stbi__create_png_image` (stb_image.h:4861 @5266f24da), with the
/// de-interlacing.
fn createPngImage(a: *Png, image_data_in: []const u8, out_n: c_int, depth: c_int, color: c_int, interlaced: bool) bool {
    const s = a.s;
    const bytes: c_int = if (depth == 16) 2 else 1;
    const out_bytes: usize = @intCast(out_n * bytes);
    if (!interlaced) return createPngImageRaw(a, image_data_in, out_n, s.img_x, s.img_y, depth, color);

    // de-interlacing
    const final = s.mallocMad3(@bitCast(s.img_x), @bitCast(s.img_y), @intCast(out_bytes), 0) orelse return s.err("outofmem") != 0;
    var image_data = image_data_in;
    const xorig = [7]u32{ 0, 4, 0, 2, 0, 1, 0 };
    const yorig = [7]u32{ 0, 0, 4, 0, 2, 0, 1 };
    const xspc = [7]u32{ 8, 8, 4, 4, 2, 2, 1 };
    const yspc = [7]u32{ 8, 8, 8, 4, 4, 2, 2 };
    for (0..7) |p| {
        const x: u32 = (s.img_x -% xorig[p] +% xspc[p] -% 1) / xspc[p];
        const y: u32 = (s.img_y -% yorig[p] +% yspc[p] -% 1) / yspc[p];
        if (x != 0 and y != 0) {
            const img_len: u32 = ((((@as(u32, @intCast(s.img_n)) * x * @as(u32, @intCast(depth))) + 7) >> 3) + 1) * y;
            if (!createPngImageRaw(a, image_data, out_n, x, y, depth, color)) {
                s.gpa.free(final);
                return false;
            }
            for (0..y) |j| {
                for (0..x) |i| {
                    const out_y = j * yspc[p] + yorig[p];
                    const out_x = i * xspc[p] + xorig[p];
                    @memcpy(final[out_y * s.img_x * out_bytes + out_x * out_bytes ..][0..out_bytes], a.out.?[(j * x + i) * out_bytes ..][0..out_bytes]);
                }
            }
            s.gpa.free(a.out.?);
            a.out = null;
            image_data = image_data[@min(img_len, image_data.len)..];
        }
    }
    a.out = final;
    return true;
}

/// Ports `stbi__compute_transparency` and `stbi__compute_transparency16`
/// (stb_image.h:4906, 4931 @5266f24da).
fn computeTransparency(z: *Png, tc: [3]u16, out_n: c_int, sixteen: bool) void {
    const s = z.s;
    const pixel_count: usize = @as(usize, s.img_x) * s.img_y;
    const p = z.out.?;
    const full: u16 = if (sixteen) 65535 else 255;
    const get = struct {
        fn f(buf: []const u8, i: usize, w: bool) u16 {
            return if (w) rd16(buf, i) else buf[i];
        }
    }.f;
    const set = struct {
        fn f(buf: []u8, i: usize, v: u16, w: bool) void {
            if (w) wr16(buf, i, v) else buf[i] = @truncate(v);
        }
    }.f;
    if (out_n == 2) {
        for (0..pixel_count) |i| set(p, i * 2 + 1, if (get(p, i * 2, sixteen) == tc[0]) 0 else full, sixteen);
    } else {
        for (0..pixel_count) |i| {
            if (get(p, i * 4, sixteen) == tc[0] and get(p, i * 4 + 1, sixteen) == tc[1] and get(p, i * 4 + 2, sixteen) == tc[2])
                set(p, i * 4 + 3, 0, sixteen);
        }
    }
}

/// Ports `stbi__expand_png_palette` (stb_image.h:4956 @5266f24da).
fn expandPngPalette(a: *Png, palette: *const [1024]u8, pal_img_n: c_int) bool {
    const s = a.s;
    const pixel_count: u32 = s.img_x *% s.img_y;
    const orig = a.out.?;
    const p = s.mallocMad2(@bitCast(pixel_count), pal_img_n, 0) orelse return s.err("outofmem") != 0;
    const pn: usize = @intCast(pal_img_n);
    for (0..pixel_count) |i| {
        const n = @as(usize, orig[i]) * 4;
        for (0..pn) |c| p[i * pn + c] = palette[n + c];
    }
    s.gpa.free(orig);
    a.out = p;
    return true;
}

fn pngType(comptime t: *const [4]u8) u32 {
    return (@as(u32, t[0]) << 24) + (@as(u32, t[1]) << 16) + (@as(u32, t[2]) << 8) + t[3];
}

/// Ports `stbi__parse_png_file` (stb_image.h:5078 @5266f24da), for loading.
fn parsePngFile(z: *Png, req_comp: c_int) bool {
    const s = z.s;
    var palette: [1024]u8 = undefined;
    var pal_img_n: c_int = 0;
    var has_trans = false;
    var tc: [3]u16 = .{ 0, 0, 0 };
    var tc16: [3]u16 = .{ 0, 0, 0 };
    var ioff: u32 = 0;
    var idata_limit: u32 = 0;
    var pal_len: u32 = 0;
    var first = true;
    var interlace: c_int = 0;
    var color: c_int = 0;
    var is_iphone = false;

    z.expanded = null;
    z.idata = null;
    z.out = null;

    if (!checkPngHeader(s)) return false;

    while (true) {
        const length = s.get32be();
        const ctype = s.get32be();
        switch (ctype) {
            pngType("CgBI") => {
                is_iphone = true;
                s.skip(@bitCast(length));
            },
            pngType("IHDR") => {
                if (!first) return s.err("multiple IHDR") != 0;
                first = false;
                if (length != 13) return s.err("bad IHDR len") != 0;
                s.img_x = s.get32be();
                s.img_y = s.get32be();
                if (s.img_y > max_dimensions) return s.err("too large") != 0;
                if (s.img_x > max_dimensions) return s.err("too large") != 0;
                z.depth = s.get8();
                if (z.depth != 1 and z.depth != 2 and z.depth != 4 and z.depth != 8 and z.depth != 16) return s.err("1/2/4/8/16-bit only") != 0;
                color = s.get8();
                if (color > 6) return s.err("bad ctype") != 0;
                if (color == 3 and z.depth == 16) return s.err("bad ctype") != 0;
                if (color == 3) {
                    pal_img_n = 3;
                } else if (color & 1 != 0) return s.err("bad ctype") != 0;
                const comp = s.get8();
                if (comp != 0) return s.err("bad comp method") != 0;
                const filter = s.get8();
                if (filter != 0) return s.err("bad filter method") != 0;
                interlace = s.get8();
                if (interlace > 1) return s.err("bad interlace method") != 0;
                if (s.img_x == 0 or s.img_y == 0) return s.err("0-pixel image") != 0;
                if (pal_img_n == 0) {
                    s.img_n = (if (color & 2 != 0) @as(c_int, 3) else 1) + (if (color & 4 != 0) @as(c_int, 1) else 0);
                    if ((1 << 30) / s.img_x / @as(u32, @intCast(s.img_n)) < s.img_y) return s.err("too large") != 0;
                } else {
                    // if paletted, then pal_n is our final components, and
                    // img_n is # components to decompress/filter.
                    s.img_n = 1;
                    if ((1 << 30) / s.img_x / 4 < s.img_y) return s.err("too large") != 0;
                }
            },
            pngType("PLTE") => {
                if (first) return s.err("first not IHDR") != 0;
                if (length > 256 * 3) return s.err("invalid PLTE") != 0;
                pal_len = length / 3;
                if (pal_len * 3 != length) return s.err("invalid PLTE") != 0;
                for (0..pal_len) |i| {
                    palette[i * 4 + 0] = s.get8();
                    palette[i * 4 + 1] = s.get8();
                    palette[i * 4 + 2] = s.get8();
                    palette[i * 4 + 3] = 255;
                }
            },
            pngType("tRNS") => {
                if (first) return s.err("first not IHDR") != 0;
                if (z.idata != null) return s.err("tRNS after IDAT") != 0;
                if (pal_img_n != 0) {
                    if (pal_len == 0) return s.err("tRNS before PLTE") != 0;
                    if (length > pal_len) return s.err("bad tRNS len") != 0;
                    pal_img_n = 4;
                    for (0..length) |i| palette[i * 4 + 3] = s.get8();
                } else {
                    if (s.img_n & 1 == 0) return s.err("tRNS with alpha") != 0;
                    if (length != @as(u32, @intCast(s.img_n)) * 2) return s.err("bad tRNS len") != 0;
                    has_trans = true;
                    var k: usize = 0;
                    if (z.depth == 16) {
                        while (k < s.img_n and k < 3) : (k += 1) tc16[k] = @truncate(@as(c_uint, @bitCast(s.get16be()))); // copy the values as-is
                    } else {
                        while (k < s.img_n and k < 3) : (k += 1) tc[k] = @as(u8, @truncate(@as(c_uint, @bitCast(s.get16be())) & 255)) *% depth_scale_table[@intCast(z.depth)]; // non 8-bit images will be larger
                    }
                }
            },
            pngType("IDAT") => {
                if (first) return s.err("first not IHDR") != 0;
                if (pal_img_n != 0 and pal_len == 0) return s.err("no PLTE") != 0;
                if (length > (1 << 30)) return s.err("IDAT size limit") != 0;
                if (@as(i32, @bitCast(ioff +% length)) < @as(i32, @bitCast(ioff))) return false;
                if (ioff + length > idata_limit) {
                    if (idata_limit == 0) idata_limit = if (length > 4096) length else 4096;
                    while (ioff + length > idata_limit) idata_limit *%= 2;
                    z.idata = if (z.idata) |old|
                        s.gpa.realloc(old, idata_limit) catch return s.err("outofmem") != 0
                    else
                        s.gpa.alloc(u8, idata_limit) catch return s.err("outofmem") != 0;
                }
                // a zero-length IDAT before any data leaves `idata` null, as in the C
                const dst: []u8 = if (z.idata) |d| d[ioff..][0..length] else &.{};
                if (!s.getn(dst)) return s.err("outofdata") != 0;
                ioff += length;
            },
            pngType("IEND") => {
                if (first) return s.err("first not IHDR") != 0;
                if (z.idata == null) return s.err("no IDAT") != 0;
                // initial guess for decoded data size to avoid unnecessary reallocs
                const bpl: u32 = (s.img_x * @as(u32, @intCast(z.depth)) + 7) / 8; // bytes per line, per component
                const raw_len: u32 = bpl *% s.img_y *% @as(u32, @intCast(s.img_n)) +% s.img_y;
                z.expanded = zlibDecodeGuesssizeHeaderflag(s, z.idata.?[0..ioff], raw_len, !is_iphone);
                if (z.expanded == null) return false; // zlib should set error
                s.gpa.free(z.idata.?);
                z.idata = null;
                if ((req_comp == s.img_n + 1 and req_comp != 3 and pal_img_n == 0) or has_trans) {
                    s.img_out_n = s.img_n + 1;
                } else {
                    s.img_out_n = s.img_n;
                }
                if (!createPngImage(z, z.expanded.?, s.img_out_n, z.depth, color, interlace != 0)) return false;
                if (has_trans) computeTransparency(z, if (z.depth == 16) tc16 else tc, s.img_out_n, z.depth == 16);
                // `stbi__de_iphone` only under a flag nothing sets
                if (pal_img_n != 0) {
                    // pal_img_n == 3 or 4
                    s.img_n = pal_img_n; // record the actual colors we had
                    s.img_out_n = pal_img_n;
                    if (req_comp >= 3) s.img_out_n = req_comp;
                    if (!expandPngPalette(z, &palette, s.img_out_n)) return false;
                } else if (has_trans) {
                    // non-paletted image with tRNS -> source image has (constant) alpha
                    s.img_n += 1;
                }
                s.gpa.free(z.expanded.?);
                z.expanded = null;
                // end of PNG chunk, read and skip CRC
                _ = s.get32be();
                return true;
            },
            else => {
                // if critical, fail
                if (first) return s.err("first not IHDR") != 0;
                if ((ctype & (1 << 29)) == 0) {
                    invalid_chunk[0] = @truncate(ctype >> 24);
                    invalid_chunk[1] = @truncate(ctype >> 16);
                    invalid_chunk[2] = @truncate(ctype >> 8);
                    invalid_chunk[3] = @truncate(ctype >> 0);
                    // a C string: a zero byte in the chunk type ends it
                    return s.err(invalid_chunk[0 .. std.mem.indexOfScalar(u8, &invalid_chunk, 0) orelse invalid_chunk.len]) != 0;
                }
                s.skip(@bitCast(length));
            },
        }
        // end of PNG chunk, read and skip CRC
        _ = s.get32be();
    }
}

/// `stbi__parse_png_file`'s static `invalid_chunk` (stb_image.h:5247
/// @5266f24da): the failure reason points into it, so it outlives the call.
threadlocal var invalid_chunk: [24]u8 = "XXXX PNG chunk not known".*;

/// Ports `stbi__check_png_header` (stb_image.h:4621 @5266f24da).
fn checkPngHeader(s: *Context) bool {
    const png_sig = [8]u8{ 137, 80, 78, 71, 13, 10, 26, 10 };
    for (png_sig) |b| {
        if (s.get8() != b) return s.err("bad png sig") != 0;
    }
    return true;
}

// -----------------------------------------------------------------------------
// the format tests and loaders, as `Context` methods

/// Ports `stbi__png_test` (stb_image.h:5302 @5266f24da).
fn pngTestImpl(s: *Context) bool {
    const r = checkPngHeader(s);
    s.rewind();
    return r;
}

/// Ports `stbi__do_png` and `stbi__png_load` (stb_image.h:5263, 5295 @5266f24da).
fn pngLoadImpl(s: *Context, x: *c_int, y: *c_int, n: *c_int, req_comp: c_int, bits: *c_int) ?[]u8 {
    var p: Png = .{ .s = s };
    var result: ?[]u8 = null;
    if (req_comp < 0 or req_comp > 4) return s.errpuc("bad req_comp");
    if (parsePngFile(&p, req_comp)) {
        if (p.depth <= 8) {
            bits.* = 8;
        } else if (p.depth == 16) {
            bits.* = 16;
        } else return s.errpuc("bad bits_per_channel");
        result = p.out;
        p.out = null;
        if (req_comp != 0 and req_comp != s.img_out_n) {
            if (bits.* == 8) {
                result = s.convertFormat(result.?, s.img_out_n, req_comp, s.img_x, s.img_y);
            } else {
                result = s.convertFormat16(result.?, s.img_out_n, req_comp, s.img_x, s.img_y);
            }
            s.img_out_n = req_comp;
            if (result == null) return result;
        }
        x.* = @bitCast(s.img_x);
        y.* = @bitCast(s.img_y);
        n.* = s.img_n;
    }
    s.free(p.out);
    s.free(p.expanded);
    s.free(p.idata);
    return result;
}

/// Ports `stbi__bmp_test_raw` and `stbi__bmp_test` (stb_image.h:5346, 5361 @5266f24da).
fn bmpTestImpl(s: *Context) bool {
    const r = blk: {
        if (s.get8() != 'B') break :blk false;
        if (s.get8() != 'M') break :blk false;
        _ = s.get32le(); // discard filesize
        _ = s.get16le(); // discard reserved
        _ = s.get16le(); // discard reserved
        _ = s.get32le(); // discard data offset
        const sz = s.get32le();
        break :blk sz == 12 or sz == 40 or sz == 56 or sz == 108 or sz == 124;
    };
    s.rewind();
    return r;
}

/// Ports `stbi__high_bit` (stb_image.h:5370 @5266f24da).
fn highBit(z_in: c_uint) c_int {
    var z = z_in;
    var n: c_int = 0;
    if (z == 0) return -1;
    if (z >= 0x10000) {
        n += 16;
        z >>= 16;
    }
    if (z >= 0x00100) {
        n += 8;
        z >>= 8;
    }
    if (z >= 0x00010) {
        n += 4;
        z >>= 4;
    }
    if (z >= 0x00004) {
        n += 2;
        z >>= 2;
    }
    if (z >= 0x00002) n += 1;
    return n;
}

/// Ports `stbi__bitcount` (stb_image.h:5382 @5266f24da).
fn bitcount(a_in: c_uint) c_int {
    var a = a_in;
    a = (a & 0x55555555) + ((a >> 1) & 0x55555555); // max 2
    a = (a & 0x33333333) + ((a >> 2) & 0x33333333); // max 4
    a = (a + (a >> 4)) & 0x0f0f0f0f; // max 8 per 4, now 8 bits
    a = (a + (a >> 8)); // max 16 per 8 bits
    a = (a + (a >> 16)); // max 32 per 8 bits
    return @intCast(a & 0xff);
}

/// Ports `stbi__shiftsigned` (stb_image.h:5395 @5266f24da): extract an
/// N-bit value and extend it to the full 8-bit range.
fn shiftsigned(v_in: c_uint, shift: c_int, bits: c_int) c_int {
    const mul_table = [9]c_uint{ 0, 0xff, 0x55, 0x49, 0x11, 0x21, 0x41, 0x81, 0x01 };
    const shift_table = [9]u5{ 0, 0, 0, 1, 0, 2, 4, 6, 0 };
    var v = v_in;
    if (shift < 0) {
        v = if (-shift >= 32) 0 else v << @intCast(-shift);
    } else {
        v = if (shift >= 32) 0 else v >> @intCast(shift);
    }
    v = if (8 - bits >= 32) 0 else v >> @intCast(8 - bits);
    const b: usize = @intCast(bits);
    return @bitCast((v *% mul_table[b]) >> shift_table[b]);
}

/// Ports `stbi__bmp_data` (stb_image.h:5420 @5266f24da).
const BmpData = struct {
    bpp: c_int = 0,
    offset: c_int = 0,
    hsz: c_int = 0,
    mr: c_uint = 0,
    mg: c_uint = 0,
    mb: c_uint = 0,
    ma: c_uint = 0,
    all_a: c_uint = 0,
    extra_read: c_int = 0,
};

/// Ports `stbi__bmp_set_mask_defaults` (stb_image.h:5422 @5266f24da).
fn bmpSetMaskDefaults(info: *BmpData, compress: c_int) void {
    // BI_BITFIELDS specifies masks explicitly, don't override
    if (compress == 3) return;
    if (compress == 0) {
        if (info.bpp == 16) {
            info.mr = 31 << 10;
            info.mg = 31 << 5;
            info.mb = 31 << 0;
        } else if (info.bpp == 32) {
            info.mr = 0xff << 16;
            info.mg = 0xff << 8;
            info.mb = 0xff << 0;
            info.ma = 0xff << 24;
            info.all_a = 0; // if all_a is 0 at end, then we loaded alpha channel but it was all 0
        } else {
            // otherwise, use defaults, which is all-0
            info.mr = 0;
            info.mg = 0;
            info.mb = 0;
            info.ma = 0;
        }
    }
}

/// Ports `stbi__bmp_parse_header` (stb_image.h:5448 @5266f24da).
fn bmpParseHeader(s: *Context, info: *BmpData) bool {
    if (s.get8() != 'B' or s.get8() != 'M') return s.err("not BMP") != 0;
    _ = s.get32le(); // discard filesize
    _ = s.get16le(); // discard reserved
    _ = s.get16le(); // discard reserved
    info.offset = @bitCast(s.get32le());
    const hsz: c_int = @bitCast(s.get32le());
    info.hsz = hsz;
    info.mr = 0;
    info.mg = 0;
    info.mb = 0;
    info.ma = 0;
    info.extra_read = 14;

    if (info.offset < 0) return s.err("bad BMP") != 0;

    if (hsz != 12 and hsz != 40 and hsz != 56 and hsz != 108 and hsz != 124) return s.err("unknown BMP") != 0;
    if (hsz == 12) {
        s.img_x = @bitCast(s.get16le());
        s.img_y = @bitCast(s.get16le());
    } else {
        s.img_x = s.get32le();
        s.img_y = s.get32le();
    }
    if (s.get16le() != 1) return s.err("bad BMP") != 0;
    info.bpp = s.get16le();
    if (hsz != 12) {
        const compress: c_int = @bitCast(s.get32le());
        if (compress == 1 or compress == 2) return s.err("BMP RLE") != 0;
        if (compress >= 4) return s.err("BMP JPEG/PNG") != 0; // this includes PNG/JPEG modes
        if (compress == 3 and info.bpp != 16 and info.bpp != 32) return s.err("bad BMP") != 0; // bitfields requires 16 or 32 bits/pixel
        _ = s.get32le(); // discard sizeof
        _ = s.get32le(); // discard hres
        _ = s.get32le(); // discard vres
        _ = s.get32le(); // discard colorsused
        _ = s.get32le(); // discard max important
        if (hsz == 40 or hsz == 56) {
            if (hsz == 56) {
                _ = s.get32le();
                _ = s.get32le();
                _ = s.get32le();
                _ = s.get32le();
            }
            if (info.bpp == 16 or info.bpp == 32) {
                if (compress == 0) {
                    bmpSetMaskDefaults(info, compress);
                } else if (compress == 3) {
                    info.mr = s.get32le();
                    info.mg = s.get32le();
                    info.mb = s.get32le();
                    info.extra_read += 12;
                    // not documented, but generated by photoshop and handled by mspaint
                    if (info.mr == info.mg and info.mg == info.mb) return s.err("bad BMP") != 0;
                } else return s.err("bad BMP") != 0;
            }
        } else {
            // V4/V5 header
            info.mr = s.get32le();
            info.mg = s.get32le();
            info.mb = s.get32le();
            info.ma = s.get32le();
            if (compress != 3) bmpSetMaskDefaults(info, compress); // override mr/mg/mb unless in BI_BITFIELDS mode, as per docs
            _ = s.get32le(); // discard color space
            for (0..12) |_| _ = s.get32le(); // discard color space parameters
            if (hsz == 124) {
                _ = s.get32le(); // discard rendering intent
                _ = s.get32le(); // discard offset of profile data
                _ = s.get32le(); // discard size of profile data
                _ = s.get32le(); // discard reserved
            }
        }
    }
    return true;
}

/// Ports `stbi__bmp_load` (stb_image.h:5531 @5266f24da).
fn bmpLoadImpl(s: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
    var pal: [256][4]u8 = @splat(@splat(0));
    var psize: c_int = 0;
    var info: BmpData = .{ .all_a = 255 };

    if (!bmpParseHeader(s, &info)) return null; // error code already set

    const flip_vertically = @as(c_int, @bitCast(s.img_y)) > 0;
    s.img_y = @intCast(@abs(@as(c_int, @bitCast(s.img_y))));

    if (s.img_y > max_dimensions) return s.errpuc("too large");
    if (s.img_x > max_dimensions) return s.errpuc("too large");

    const mr = info.mr;
    const mg = info.mg;
    const mb = info.mb;
    const ma = info.ma;
    var all_a = info.all_a;

    if (info.hsz == 12) {
        if (info.bpp < 24) psize = @divTrunc(info.offset -% info.extra_read -% 24, 3);
    } else {
        if (info.bpp < 16) psize = (info.offset -% info.extra_read -% info.hsz) >> 2;
    }
    if (psize == 0) {
        // accept some number of extra bytes after the header, but if the offset points either to before
        // the header ends or implies a large amount of extra data, reject the file as malformed
        const bytes_read_so_far: c_int = @intCast(@min(s.pos, std.math.maxInt(c_int)));
        const header_limit = 1024; // max we actually read is below 256 bytes currently.
        const extra_data_limit = 256 * 4; // what ordinarily goes here is a palette; 256 entries*4 bytes is its max size.
        if (bytes_read_so_far <= 0 or bytes_read_so_far > header_limit) return s.errpuc("bad header");
        if (info.offset < bytes_read_so_far or info.offset - bytes_read_so_far > extra_data_limit) {
            return s.errpuc("bad offset");
        } else {
            s.skip(info.offset - bytes_read_so_far);
        }
    }

    if (info.bpp == 24 and ma == 0xff000000) {
        s.img_n = 3;
    } else {
        s.img_n = if (ma != 0) 4 else 3;
    }
    // we can directly decode 3 or 4
    const target: c_int = if (req_comp != 0 and req_comp >= 3) req_comp else s.img_n; // if they want monochrome, we'll post-convert

    // sanity-check size
    if (!mad3sizesValid(target, @bitCast(s.img_x), @bitCast(s.img_y), 0)) return s.errpuc("too large");

    const out = s.mallocMad3(target, @bitCast(s.img_x), @bitCast(s.img_y), 0) orelse return s.errpuc("outofmem");
    const tu: usize = @intCast(target);
    const img_x: usize = s.img_x;
    const img_y: usize = s.img_y;
    if (info.bpp < 16) {
        var z: usize = 0;
        if (psize == 0 or psize > 256) {
            s.gpa.free(out);
            return s.errpuc("invalid");
        }
        for (0..@intCast(@max(psize, 0))) |i| { // a negative size passes the check above and reads none
            pal[i][2] = s.get8();
            pal[i][1] = s.get8();
            pal[i][0] = s.get8();
            if (info.hsz != 12) _ = s.get8();
            pal[i][3] = 255;
        }
        s.skip(info.offset -% info.extra_read -% info.hsz -% psize * (if (info.hsz == 12) @as(c_int, 3) else 4));
        const width: c_int = if (info.bpp == 1) @intCast((s.img_x + 7) >> 3) else if (info.bpp == 4) @intCast((s.img_x + 1) >> 1) else if (info.bpp == 8) @intCast(s.img_x) else {
            s.gpa.free(out);
            return s.errpuc("bad bpp");
        };
        const pad = (-width) & 3;
        if (info.bpp == 1) {
            for (0..img_y) |_| {
                var bit_offset: c_int = 7;
                var v: c_int = s.get8();
                for (0..img_x) |i| {
                    const color: usize = @intCast((v >> @intCast(bit_offset)) & 0x1);
                    if (color >= psize) read_indeterminate = true;
                    out[z] = pal[color][0];
                    out[z + 1] = pal[color][1];
                    out[z + 2] = pal[color][2];
                    z += 3;
                    if (target == 4) {
                        out[z] = 255;
                        z += 1;
                    }
                    if (i + 1 == img_x) break;
                    bit_offset -= 1;
                    if (bit_offset < 0) {
                        bit_offset = 7;
                        v = s.get8();
                    }
                }
                s.skip(pad);
            }
        } else {
            for (0..img_y) |_| {
                var i: usize = 0;
                while (i < img_x) : (i += 2) {
                    var v: usize = s.get8();
                    var v2: usize = 0;
                    if (info.bpp == 4) {
                        v2 = v & 15;
                        v >>= 4;
                    }
                    if (v >= psize) read_indeterminate = true;
                    out[z] = pal[v][0];
                    out[z + 1] = pal[v][1];
                    out[z + 2] = pal[v][2];
                    z += 3;
                    if (target == 4) {
                        out[z] = 255;
                        z += 1;
                    }
                    if (i + 1 == img_x) break;
                    v = if (info.bpp == 8) s.get8() else v2;
                    if (v >= psize) read_indeterminate = true;
                    out[z] = pal[v][0];
                    out[z + 1] = pal[v][1];
                    out[z + 2] = pal[v][2];
                    z += 3;
                    if (target == 4) {
                        out[z] = 255;
                        z += 1;
                    }
                }
                s.skip(pad);
            }
        }
    } else {
        var rshift: c_int = 0;
        var gshift: c_int = 0;
        var bshift: c_int = 0;
        var ashift: c_int = 0;
        var rcount: c_int = 0;
        var gcount: c_int = 0;
        var bcount: c_int = 0;
        var acount: c_int = 0;
        var z: usize = 0;
        var easy: c_int = 0;
        s.skip(info.offset -% info.extra_read -% info.hsz);
        const width: c_int = if (info.bpp == 24) 3 * @as(c_int, @bitCast(s.img_x)) else if (info.bpp == 16) 2 * @as(c_int, @bitCast(s.img_x)) else 0; // bpp = 32 and pad = 0
        const pad = (-%width) & 3;
        if (info.bpp == 24) {
            easy = 1;
        } else if (info.bpp == 32) {
            if (mb == 0xff and mg == 0xff00 and mr == 0x00ff0000 and ma == 0xff000000) easy = 2;
        }
        if (easy == 0) {
            if (mr == 0 or mg == 0 or mb == 0) {
                s.gpa.free(out);
                return s.errpuc("bad masks");
            }
            // right shift amt to put high bit in position #7
            rshift = highBit(mr) - 7;
            rcount = bitcount(mr);
            gshift = highBit(mg) - 7;
            gcount = bitcount(mg);
            bshift = highBit(mb) - 7;
            bcount = bitcount(mb);
            ashift = highBit(ma) - 7;
            acount = bitcount(ma);
            if (rcount > 8 or gcount > 8 or bcount > 8 or acount > 8) {
                s.gpa.free(out);
                return s.errpuc("bad masks");
            }
        }
        for (0..img_y) |_| {
            if (easy != 0) {
                for (0..img_x) |_| {
                    out[z + 2] = s.get8();
                    out[z + 1] = s.get8();
                    out[z + 0] = s.get8();
                    z += 3;
                    const a: u8 = if (easy == 2) s.get8() else 255;
                    all_a |= a;
                    if (target == 4) {
                        out[z] = a;
                        z += 1;
                    }
                }
            } else {
                const bpp = info.bpp;
                for (0..img_x) |_| {
                    const v: u32 = if (bpp == 16) @bitCast(s.get16le()) else s.get32le();
                    out[z] = @truncate(@as(c_uint, @bitCast(shiftsigned(v & mr, rshift, rcount))));
                    out[z + 1] = @truncate(@as(c_uint, @bitCast(shiftsigned(v & mg, gshift, gcount))));
                    out[z + 2] = @truncate(@as(c_uint, @bitCast(shiftsigned(v & mb, bshift, bcount))));
                    z += 3;
                    const a: c_uint = if (ma != 0) @bitCast(shiftsigned(v & ma, ashift, acount)) else 255;
                    all_a |= a;
                    if (target == 4) {
                        out[z] = @truncate(a);
                        z += 1;
                    }
                }
            }
            s.skip(pad);
        }
    }

    // if alpha channel is all 0s, replace with all 255s
    if (target == 4 and all_a == 0) {
        var i: usize = 4 * img_x * img_y;
        while (i >= 4) : (i -= 4) out[i - 1] = 255;
    }

    if (flip_vertically) {
        for (0..img_y >> 1) |j| {
            const p1 = out[j * img_x * tu ..][0 .. img_x * tu];
            const p2 = out[(img_y - 1 - j) * img_x * tu ..][0 .. img_x * tu];
            for (p1, p2) |*a, *b| std.mem.swap(u8, a, b);
        }
    }

    var result = out;
    if (req_comp != 0 and req_comp != target) {
        result = s.convertFormat(out, target, req_comp, s.img_x, s.img_y) orelse return null; // convertFormat frees input on failure
    }

    x.* = @bitCast(s.img_x);
    y.* = @bitCast(s.img_y);
    comp.* = s.img_n;
    return result;
}

/// Ports `stbi__tga_get_comp` (stb_image.h:5739 @5266f24da).
fn tgaGetComp(bits_per_pixel: c_int, is_grey: bool, is_rgb16: ?*bool) c_int {
    // only RGB or RGBA (incl. 16bit) or grey allowed
    if (is_rgb16) |p| p.* = false;
    switch (bits_per_pixel) {
        8 => return 1,
        16, 15 => {
            if (bits_per_pixel == 16 and is_grey) return 2;
            if (is_rgb16) |p| p.* = true;
            return 3;
        },
        24, 32 => return @divTrunc(bits_per_pixel, 8),
        else => return 0,
    }
}

/// Ports `stbi__tga_test` (stb_image.h:5820 @5266f24da).
fn tgaTestImpl(s: *Context) bool {
    const res = blk: {
        _ = s.get8(); // discard Offset
        const tga_color_type = s.get8(); // color type
        if (tga_color_type > 1) break :blk false; // only RGB or indexed allowed
        var sz: c_int = s.get8(); // image type
        if (tga_color_type == 1) { // colormapped (paletted) image
            if (sz != 1 and sz != 9) break :blk false; // colortype 1 demands image type 1 or 9
            s.skip(4); // skip index of first colormap entry and number of entries
            sz = s.get8(); // check bits per palette color entry
            if (sz != 8 and sz != 15 and sz != 16 and sz != 24 and sz != 32) break :blk false;
            s.skip(4); // skip image x and y origin
        } else { // "normal" image w/o colormap
            if (sz != 2 and sz != 3 and sz != 10 and sz != 11) break :blk false; // only RGB or grey allowed, +/- RLE
            s.skip(9); // skip colormap specification and image x/y origin
        }
        if (s.get16le() < 1) break :blk false; // test width
        if (s.get16le() < 1) break :blk false; // test height
        sz = s.get8(); // bits per pixel
        if (tga_color_type == 1 and sz != 8 and sz != 16) break :blk false; // for colormapped images, bpp is size of an index
        if (sz != 8 and sz != 15 and sz != 16 and sz != 24 and sz != 32) break :blk false;
        break :blk true;
    };
    s.rewind();
    return res;
}

/// Ports `stbi__tga_read_rgb16` (stb_image.h:5852 @5266f24da).
fn tgaReadRgb16(s: *Context, out: []u8) void {
    const px: u16 = @truncate(@as(c_uint, @bitCast(s.get16le())));
    const five_bit_mask: u16 = 31;
    // we have 3 channels with 5bits each
    const r: c_int = (px >> 10) & five_bit_mask;
    const g: c_int = (px >> 5) & five_bit_mask;
    const b: c_int = px & five_bit_mask;
    out[0] = @intCast(@divTrunc(r * 255, 31));
    out[1] = @intCast(@divTrunc(g * 255, 31));
    out[2] = @intCast(@divTrunc(b * 255, 31));
}

/// Ports `stbi__tga_load` (stb_image.h:5871 @5266f24da).
fn tgaLoadImpl(s: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
    //   read in the TGA header stuff
    const tga_offset: c_int = s.get8();
    const tga_indexed: c_int = s.get8();
    var tga_image_type: c_int = s.get8();
    var tga_is_rle = false;
    const tga_palette_start = s.get16le();
    const tga_palette_len = s.get16le();
    const tga_palette_bits: c_int = s.get8();
    _ = s.get16le(); // x origin
    _ = s.get16le(); // y origin
    const tga_width = s.get16le();
    const tga_height = s.get16le();
    const tga_bits_per_pixel: c_int = s.get8();
    var tga_rgb16 = false;
    var tga_inverted: c_int = s.get8();
    var raw_data: [4]u8 = .{ 0, 0, 0, 0 };
    var rle_count: c_int = 0;
    var rle_repeating = false;
    var read_next_pixel = true;

    if (tga_height > max_dimensions) return s.errpuc("too large");
    if (tga_width > max_dimensions) return s.errpuc("too large");

    //   do a tiny bit of precessing
    if (tga_image_type >= 8) {
        tga_image_type -= 8;
        tga_is_rle = true;
    }
    tga_inverted = 1 - ((tga_inverted >> 5) & 1);

    //   If I'm paletted, then I'll use the number of bits from the palette
    const tga_comp: c_int = if (tga_indexed != 0) tgaGetComp(tga_palette_bits, false, &tga_rgb16) else tgaGetComp(tga_bits_per_pixel, tga_image_type == 3, &tga_rgb16);

    if (tga_comp == 0) return s.errpuc("bad format"); // shouldn't really happen

    //   tga info
    x.* = tga_width;
    y.* = tga_height;
    comp.* = tga_comp;

    if (!mad3sizesValid(tga_width, tga_height, tga_comp, 0)) return s.errpuc("too large");

    const tga_data = s.mallocMad3(tga_width, tga_height, tga_comp, 0) orelse return s.errpuc("outofmem");
    const cu: usize = @intCast(tga_comp);
    const wu: usize = @intCast(tga_width);
    const hu: usize = @intCast(tga_height);

    // skip to the data's starting position (offset usually = 0)
    s.skip(tga_offset);

    if (tga_indexed == 0 and !tga_is_rle and !tga_rgb16) {
        for (0..hu) |i| {
            const row = if (tga_inverted != 0) hu - i - 1 else i;
            _ = s.getn(tga_data[row * wu * cu ..][0 .. wu * cu]);
        }
    } else {
        var tga_palette: ?[]u8 = null;
        defer if (tga_palette) |p| s.gpa.free(p);
        //   do I need to load a palette?
        if (tga_indexed != 0) {
            if (tga_palette_len == 0) { // you have to have at least one entry!
                s.gpa.free(tga_data);
                return s.errpuc("bad palette");
            }

            //   any data to skip? (offset usually = 0)
            s.skip(tga_palette_start);
            //   load the palette
            tga_palette = s.mallocMad2(tga_palette_len, tga_comp, 0) orelse {
                s.gpa.free(tga_data);
                return s.errpuc("outofmem");
            };
            if (tga_rgb16) {
                var pe: usize = 0;
                for (0..@intCast(tga_palette_len)) |_| {
                    tgaReadRgb16(s, tga_palette.?[pe..]);
                    pe += cu;
                }
            } else if (!s.getn(tga_palette.?[0 .. @as(usize, @intCast(tga_palette_len)) * cu])) {
                s.gpa.free(tga_data);
                return s.errpuc("bad palette");
            }
        }
        //   load the data
        for (0..wu * hu) |i| {
            //   if I'm in RLE mode, do I need to get a RLE stbi__pngchunk?
            if (tga_is_rle) {
                if (rle_count == 0) {
                    //   yep, get the next byte as a RLE command
                    const rle_cmd: c_int = s.get8();
                    rle_count = 1 + (rle_cmd & 127);
                    rle_repeating = (rle_cmd >> 7) != 0;
                    read_next_pixel = true;
                } else if (!rle_repeating) {
                    read_next_pixel = true;
                }
            } else {
                read_next_pixel = true;
            }
            //   OK, if I need to read a pixel, do it now
            if (read_next_pixel) {
                //   load however much data we did have
                if (tga_indexed != 0) {
                    // read in index, then perform the lookup
                    var pal_idx: c_int = if (tga_bits_per_pixel == 8) s.get8() else s.get16le();
                    if (pal_idx >= tga_palette_len) pal_idx = 0; // invalid index
                    const pi: usize = @as(usize, @intCast(pal_idx)) * cu;
                    for (0..cu) |j| raw_data[j] = tga_palette.?[pi + j];
                } else if (tga_rgb16) {
                    tgaReadRgb16(s, &raw_data);
                } else {
                    //   read in the data raw
                    for (0..cu) |j| raw_data[j] = s.get8();
                }
                //   clear the reading flag for the next pixel
                read_next_pixel = false;
            } // end of reading a pixel

            // copy data
            for (0..cu) |j| tga_data[i * cu + j] = raw_data[j];

            //   in case we're in RLE mode, keep counting down
            rle_count -= 1;
        }
        //   do I need to invert the image?
        if (tga_inverted != 0) {
            var j: usize = 0;
            while (j * 2 < hu) : (j += 1) {
                const index1 = j * wu * cu;
                const index2 = (hu - 1 - j) * wu * cu;
                for (0..wu * cu) |k| std.mem.swap(u8, &tga_data[index1 + k], &tga_data[index2 + k]);
            }
        }
    }

    // swap RGB - if the source data was RGB16, it already is in the right order
    if (tga_comp >= 3 and !tga_rgb16) {
        for (0..wu * hu) |i| std.mem.swap(u8, &tga_data[i * cu], &tga_data[i * cu + 2]);
    }

    // convert to target component count
    if (req_comp != 0 and req_comp != tga_comp)
        return s.convertFormat(tga_data, tga_comp, req_comp, @intCast(tga_width), @intCast(tga_height));
    return tga_data;
}

/// Ports `stbi__psd_test` (stb_image.h:6081 @5266f24da).
fn psdTestImpl(s: *Context) bool {
    const r = s.get32be() == 0x38425053;
    s.rewind();
    return r;
}

/// Ports `stbi__psd_decode_rle` (stb_image.h:6088 @5266f24da).
fn psdDecodeRle(s: *Context, out: []u8, start: usize, pixel_count: c_int) bool {
    var count: c_int = 0;
    var p = start;
    while (true) {
        const nleft = pixel_count - count;
        if (nleft <= 0) break;
        var len: c_int = s.get8();
        if (len == 128) {
            // No-op.
        } else if (len < 128) {
            // Copy next len+1 bytes literally.
            len += 1;
            if (len > nleft) return false; // corrupt data
            count += len;
            while (len != 0) : (len -= 1) {
                out[p] = s.get8();
                p += 4;
            }
        } else {
            // Next -len+1 bytes in the dest are replicated from next source byte.
            len = 257 - len;
            if (len > nleft) return false; // corrupt data
            const val = s.get8();
            count += len;
            while (len != 0) : (len -= 1) {
                out[p] = val;
                p += 4;
            }
        }
    }
    return true;
}

/// Ports `stbi__psd_load` (stb_image.h:6126 @5266f24da), at `bpc` 8.
fn psdLoadImpl(s: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
    // Check identifier
    if (s.get32be() != 0x38425053) return s.errpuc("not PSD"); // "8BPS"

    // Check file type version.
    if (s.get16be() != 1) return s.errpuc("wrong version");

    // Skip 6 reserved bytes.
    s.skip(6);

    // Read the number of channels (R, G, B, A, etc).
    const channel_count = s.get16be();
    if (channel_count < 0 or channel_count > 16) return s.errpuc("wrong channel count");

    // Read the rows and columns of the image.
    const h: c_int = @bitCast(s.get32be());
    const w: c_int = @bitCast(s.get32be());

    if (h > max_dimensions) return s.errpuc("too large");
    if (w > max_dimensions) return s.errpuc("too large");

    // Make sure the depth is 8 bits.
    const bitdepth = s.get16be();
    if (bitdepth != 8 and bitdepth != 16) return s.errpuc("unsupported bit depth");

    // Make sure the color mode is RGB.
    if (s.get16be() != 3) return s.errpuc("wrong color format");

    // Skip the Mode Data.  (It's the palette for indexed color; other info for other modes.)
    s.skip(@bitCast(s.get32be()));

    // Skip the image resources.  (resolution, pen tool paths, etc)
    s.skip(@bitCast(s.get32be()));

    // Skip the reserved data.
    s.skip(@bitCast(s.get32be()));

    // Find out if the data is compressed.
    const compression = s.get16be();
    if (compression > 1) return s.errpuc("bad compression");

    // Check size
    if (!mad3sizesValid(4, w, h, 0)) return s.errpuc("too large");

    // Create the destination image (bpc 8: never the 16-bit buffer).
    const out = s.alloc(@intCast(4 * w * h)) orelse return s.errpuc("outofmem");
    const pixel_count = w * h;
    const pc: usize = @intCast(pixel_count);

    // Finally, the image data.
    if (compression != 0) {
        // The RLE-compressed data is preceded by a 2-byte data count for each row in the data,
        // which we're going to just skip.
        s.skip(h *% channel_count *% 2);

        // Read the RLE data by channel.
        for (0..4) |channel| {
            if (channel >= channel_count) {
                // Fill this channel with default data.
                for (0..pc) |i| out[channel + i * 4] = if (channel == 3) 255 else 0;
            } else {
                // Read the RLE data.
                if (!psdDecodeRle(s, out, channel, pixel_count)) {
                    s.gpa.free(out);
                    return s.errpuc("corrupt");
                }
            }
        }
    } else {
        // We're at the raw image data.  It's each channel in order (Red, Green, Blue, Alpha, ...)
        // where each channel consists of an 8-bit (or 16-bit) value for each pixel in the image.
        for (0..4) |channel| {
            if (channel >= channel_count) {
                // Fill this channel with default data.
                const val: u8 = if (channel == 3) 255 else 0;
                for (0..pc) |i| out[channel + i * 4] = val;
            } else {
                if (bitdepth == 16) { // input bpc
                    for (0..pc) |i| out[channel + i * 4] = @truncate(@as(c_uint, @bitCast(s.get16be() >> 8)));
                } else {
                    for (0..pc) |i| out[channel + i * 4] = s.get8();
                }
            }
        }
    }

    // remove weird white matte from PSD
    if (channel_count >= 4) {
        for (0..pc) |i| {
            const pixel = out[4 * i ..][0..4];
            if (pixel[3] != 0 and pixel[3] != 255) {
                const a = @as(f32, @floatFromInt(pixel[3])) / 255.0;
                const ra = 1.0 / a;
                const inv_a = 255.0 * (1 - ra);
                pixel[0] = toU8(@mulAdd(f32, @floatFromInt(pixel[0]), ra, inv_a));
                pixel[1] = toU8(@mulAdd(f32, @floatFromInt(pixel[1]), ra, inv_a));
                pixel[2] = toU8(@mulAdd(f32, @floatFromInt(pixel[2]), ra, inv_a));
            }
        }
    }

    var result = out;
    // convert to desired output format
    if (req_comp != 0 and req_comp != 4) {
        result = s.convertFormat(out, 4, req_comp, @intCast(w), @intCast(h)) orelse return null;
    }

    comp.* = 4;
    y.* = h;
    x.* = w;
    return result;
}

/// `(unsigned char) f`: clang converts through a saturating 32-bit
/// `fcvtzs` and keeps the low byte, so a negative `f` wraps and NaN is 0.
fn toU8(f: f32) u8 {
    if (std.math.isNan(f)) return 0;
    if (f >= 2147483648.0) return 0xff;
    if (f <= -2147483648.0) return 0;
    return @truncate(@as(c_uint, @bitCast(@as(c_int, @intFromFloat(f)))));
}

/// Ports `stbi__pic_is4` (stb_image.h:6336 @5266f24da).
fn picIs4(s: *Context, str: *const [4]u8) bool {
    for (str) |ch| {
        if (s.get8() != ch) return false;
    }
    return true;
}

/// Ports `stbi__pic_test_core` and `stbi__pic_test` (stb_image.h:6346, 6541 @5266f24da).
fn picTestImpl(s: *Context) bool {
    const r = blk: {
        if (!picIs4(s, "\x53\x80\xF6\x34")) break :blk false;
        for (0..84) |_| _ = s.get8();
        if (!picIs4(s, "PICT")) break :blk false;
        break :blk true;
    };
    s.rewind();
    return r;
}

/// Ports `stbi__pic_packet` (stb_image.h:6365 @5266f24da).
const PicPacket = struct { size: u8 = 0, type: u8 = 0, channel: u8 = 0 };

/// Ports `stbi__readval` (stb_image.h:6367 @5266f24da).
fn readval(s: *Context, channel: u8, dest: []u8) bool {
    var mask: u8 = 0x80;
    for (0..4) |i| {
        if (channel & mask != 0) {
            if (s.atEof()) return s.err("bad file") != 0;
            dest[i] = s.get8();
        }
        mask >>= 1;
    }
    return true;
}

/// Ports `stbi__copyval` (stb_image.h:6381 @5266f24da).
fn copyval(channel: u8, dest: []u8, src: []const u8) void {
    var mask: u8 = 0x80;
    for (0..4) |i| {
        if (channel & mask != 0) dest[i] = src[i];
        mask >>= 1;
    }
}

/// Ports `stbi__pic_load_core` (stb_image.h:6390 @5266f24da).
fn picLoadCore(s: *Context, width: c_int, height: c_int, comp: *c_int, result: []u8) bool {
    var act_comp: u8 = 0;
    var num_packets: usize = 0;
    var packets: [10]PicPacket = @splat(.{});

    // this will (should...) cater for even some bizarre stuff like having data
    // for the same channel in multiple packets.
    while (true) {
        if (num_packets == packets.len) return s.err("bad format") != 0;
        const packet = &packets[num_packets];
        num_packets += 1;

        const chained = s.get8();
        packet.size = s.get8();
        packet.type = s.get8();
        packet.channel = s.get8();

        act_comp |= packet.channel;

        if (s.atEof()) return s.err("bad file") != 0;
        if (packet.size != 8) return s.err("bad format") != 0;
        if (chained == 0) break;
    }

    comp.* = if (act_comp & 0x10 != 0) 4 else 3; // has alpha channel?

    const wu: usize = @intCast(width);
    for (0..@intCast(height)) |y| {
        for (packets[0..num_packets]) |*packet| {
            var dest: usize = y * wu * 4;

            switch (packet.type) {
                0 => { //uncompressed
                    for (0..wu) |_| {
                        if (!readval(s, packet.channel, result[dest..])) return false;
                        dest += 4;
                    }
                },
                1 => { //Pure RLE
                    var left: c_int = width;
                    while (left > 0) {
                        var value: [4]u8 = undefined;
                        var count = s.get8();
                        if (s.atEof()) return s.err("bad file") != 0;

                        if (count > left) count = @intCast(left);

                        if (!readval(s, packet.channel, &value)) return false;

                        for (0..count) |_| {
                            copyval(packet.channel, result[dest..], &value);
                            dest += 4;
                        }
                        left -= count;
                    }
                },
                2 => { //Mixed RLE
                    var left: c_int = width;
                    while (left > 0) {
                        var count: c_int = s.get8();
                        if (s.atEof()) return s.err("bad file") != 0;

                        if (count >= 128) { // Repeated
                            var value: [4]u8 = undefined;

                            if (count == 128) {
                                count = s.get16be();
                            } else {
                                count -= 127;
                            }
                            if (count > left) return s.err("bad file") != 0;

                            if (!readval(s, packet.channel, &value)) return false;

                            for (0..@intCast(count)) |_| {
                                copyval(packet.channel, result[dest..], &value);
                                dest += 4;
                            }
                        } else { // Raw
                            count += 1;
                            if (count > left) return s.err("bad file") != 0;

                            for (0..@intCast(count)) |_| {
                                if (!readval(s, packet.channel, result[dest..])) return false;
                                dest += 4;
                            }
                        }
                        left -= count;
                    }
                },
                else => return s.err("bad format") != 0,
            }
        }
    }
    return true;
}

/// Ports `stbi__pic_load` (stb_image.h:6500 @5266f24da).
fn picLoadImpl(s: *Context, px: *c_int, py: *c_int, comp: *c_int, req_comp_in: c_int) ?[]u8 {
    for (0..92) |_| _ = s.get8();

    const x = s.get16be();
    const y = s.get16be();

    if (y > max_dimensions) return s.errpuc("too large");
    if (x > max_dimensions) return s.errpuc("too large");

    if (s.atEof()) return s.errpuc("bad file");
    if (!mad3sizesValid(x, y, 4, 0)) return s.errpuc("too large");

    _ = s.get32be(); //skip `ratio'
    _ = s.get16be(); //skip `fields'
    _ = s.get16be(); //skip `pad'

    // intermediate buffer is RGBA
    var result: ?[]u8 = s.mallocMad3(x, y, 4, 0) orelse return s.errpuc("outofmem");
    @memset(result.?, 0xff);

    if (!picLoadCore(s, x, y, comp, result.?)) {
        s.gpa.free(result.?);
        result = null;
    }
    px.* = x;
    py.* = y;
    var req_comp = req_comp_in;
    if (req_comp == 0) req_comp = comp.*;
    if (result) |r| return s.convertFormat(r, 4, req_comp, @intCast(x), @intCast(y));
    // The C passes the null buffer of a failed core load on to
    // `stbi__convert_format`. With `req_comp` 4 that returns null; with a
    // combination it has no case for it fails "unsupported" once there is a
    // row; with one it has, it returns an empty image when there are no
    // pixels and otherwise **reads through null** -- upstream crashes on a
    // truncated PIC at the helper's `req_comp` 3. Here that case fails.
    if (req_comp == 4) return null;
    const good = s.mallocMad3(req_comp, x, y, 0) orelse return s.errpuc("outofmem");
    if (y > 0 and !validCombo(4 * 8 + req_comp)) {
        s.gpa.free(good);
        return s.errpuc("unsupported");
    }
    if (x == 0 or y == 0) return good;
    s.gpa.free(good);
    return null;
}

/// Ports `stbi__gif_lzw` (stb_image.h:6558 @5266f24da).
const GifLzw = struct { prefix: i16 = 0, first: u8 = 0, suffix: u8 = 0 };

/// Ports `stbi__gif` (stb_image.h:6578 @5266f24da).
const Gif = struct {
    w: c_int = 0,
    h: c_int = 0,
    out: ?[]u8 = null, // output buffer (always 4 components)
    background: ?[]u8 = null, // The current "background" as far as a gif is concerned
    history: ?[]u8 = null,
    flags: c_int = 0,
    bgindex: c_int = 0,
    ratio: c_int = 0,
    transparent: c_int = 0,
    eflags: c_int = 0,
    pal: [256][4]u8 = @splat(@splat(0)),
    lpal: [256][4]u8 = @splat(@splat(0)),
    codes: [8192]GifLzw = @splat(.{}),
    /// `color_table`: the local palette or the global one.
    use_lpal: bool = false,
    parse: c_int = 0,
    step: c_int = 0,
    lflags: c_int = 0,
    start_x: c_int = 0,
    start_y: c_int = 0,
    max_x: c_int = 0,
    max_y: c_int = 0,
    cur_x: c_int = 0,
    cur_y: c_int = 0,
    line_size: c_int = 0,
    delay: c_int = 0,

    fn colorTable(g: *Gif) *[256][4]u8 {
        return if (g.use_lpal) &g.lpal else &g.pal;
    }
};

/// Ports `stbi__gif_test_raw` and `stbi__gif_test` (stb_image.h:6580, 6590 @5266f24da).
fn gifTestImpl(s: *Context) bool {
    const r = blk: {
        if (s.get8() != 'G' or s.get8() != 'I' or s.get8() != 'F' or s.get8() != '8') break :blk false;
        const sz = s.get8();
        if (sz != '9' and sz != '7') break :blk false;
        if (s.get8() != 'a') break :blk false;
        break :blk true;
    };
    s.rewind();
    return r;
}

/// Ports `stbi__gif_parse_colortable` (stb_image.h:6597 @5266f24da).
fn gifParseColortable(s: *Context, pal: *[256][4]u8, num_entries: c_int, transp: c_int) void {
    for (0..@intCast(num_entries)) |i| {
        pal[i][2] = s.get8();
        pal[i][1] = s.get8();
        pal[i][0] = s.get8();
        pal[i][3] = if (transp == @as(c_int, @intCast(i))) 0 else 255;
    }
}

/// Ports `stbi__gif_header` (stb_image.h:6608 @5266f24da), not for info.
fn gifHeader(s: *Context, g: *Gif, comp: *c_int) bool {
    if (s.get8() != 'G' or s.get8() != 'I' or s.get8() != 'F' or s.get8() != '8') return s.err("not GIF") != 0;

    const version = s.get8();
    if (version != '7' and version != '9') return s.err("not GIF") != 0;
    if (s.get8() != 'a') return s.err("not GIF") != 0;

    s.failure = "";
    g.w = s.get16le();
    g.h = s.get16le();
    g.flags = s.get8();
    g.bgindex = s.get8();
    g.ratio = s.get8();
    g.transparent = -1;

    if (g.w > max_dimensions) return s.err("too large") != 0;
    if (g.h > max_dimensions) return s.err("too large") != 0;

    comp.* = 4; // can't actually tell whether it's 3 or 4 until we parse the comments

    if (g.flags & 0x80 != 0) gifParseColortable(s, &g.pal, @as(c_int, 2) << @intCast(g.flags & 7), -1);
    return true;
}

/// Ports `stbi__out_gif_code` (stb_image.h:6654 @5266f24da), with the
/// prefix recursion as a loop over a stack of codes.
fn outGifCode(g: *Gif, code_in: u16) void {
    var stack: [8192]u16 = undefined;
    var n: usize = 0;
    var code = code_in;
    // recurse to decode the prefixes, since the linked-list is backwards
    while (true) {
        stack[n] = code;
        n += 1;
        const prefix = g.codes[code].prefix;
        if (prefix < 0) break;
        code = @intCast(prefix);
    }
    const ct = g.colorTable();
    while (n > 0) {
        n -= 1;
        const c_code = stack[n];
        if (g.cur_y >= g.max_y) continue;

        const idx: usize = @intCast(g.cur_x + g.cur_y);
        const p = g.out.?[idx..];
        g.history.?[idx / 4] = 1;

        const c = &ct[g.codes[c_code].suffix];
        if (c[3] > 128) { // don't render transparent pixels;
            p[0] = c[2];
            p[1] = c[1];
            p[2] = c[0];
            p[3] = c[3];
        }
        g.cur_x += 4;

        if (g.cur_x >= g.max_x) {
            g.cur_x = g.start_x;
            g.cur_y += g.step;

            while (g.cur_y >= g.max_y and g.parse > 0) {
                g.step = (@as(c_int, 1) << @intCast(g.parse)) * g.line_size;
                g.cur_y = g.start_y + (g.step >> 1);
                g.parse -= 1;
            }
        }
    }
}

/// Ports `stbi__process_gif_raster` (stb_image.h:6691 @5266f24da).
fn processGifRaster(s: *Context, g: *Gif) bool {
    const lzw_cs = s.get8();
    if (lzw_cs > 12) return false;
    const clear: c_int = @as(c_int, 1) << @intCast(lzw_cs);
    var first = true;
    var codesize: c_int = lzw_cs + 1;
    var codemask: c_int = (@as(c_int, 1) << @intCast(codesize)) - 1;
    var bits: c_int = 0;
    var valid_bits: c_int = 0;
    for (0..@intCast(clear)) |init_code| {
        g.codes[init_code].prefix = -1;
        g.codes[init_code].first = @truncate(init_code);
        g.codes[init_code].suffix = @truncate(init_code);
    }

    // support no starting clear code
    var avail: c_int = clear + 2;
    var oldcode: c_int = -1;

    var len: c_int = 0;
    while (true) {
        if (valid_bits < codesize) {
            if (len == 0) {
                len = s.get8(); // start new block
                if (len == 0) return true;
            }
            len -= 1;
            bits |= @as(c_int, s.get8()) << @intCast(valid_bits);
            valid_bits += 8;
        } else {
            const code = bits & codemask;
            bits >>= @intCast(codesize);
            valid_bits -= codesize;
            if (code == clear) { // clear code
                codesize = lzw_cs + 1;
                codemask = (@as(c_int, 1) << @intCast(codesize)) - 1;
                avail = clear + 2;
                oldcode = -1;
                first = false;
            } else if (code == clear + 1) { // end of stream code
                s.skip(len);
                while (true) {
                    len = s.get8();
                    if (len <= 0) break;
                    s.skip(len);
                }
                return true;
            } else if (code <= avail) {
                if (first) return s.err("no clear code") != 0;

                if (oldcode >= 0) {
                    const p = &g.codes[@intCast(avail)];
                    avail += 1;
                    if (avail > 8192) return s.err("too many codes") != 0;

                    p.prefix = @intCast(oldcode);
                    p.first = g.codes[@intCast(oldcode)].first;
                    p.suffix = if (code == avail) p.first else g.codes[@intCast(code)].first;
                } else if (code == avail) return s.err("illegal code in raster") != 0;

                outGifCode(g, @intCast(code));

                if ((avail & codemask) == 0 and avail <= 0x0FFF) {
                    codesize += 1;
                    codemask = (@as(c_int, 1) << @intCast(codesize)) - 1;
                }

                oldcode = code;
            } else {
                return s.err("illegal code in raster") != 0;
            }
        }
    }
}

/// Ports `stbi__gif_load_next` (stb_image.h:6778 @5266f24da), for the first
/// frame. Return: `.frame`, `.end` for the stream's terminator, or `.fail`.
fn gifLoadFirst(s: *Context, g: *Gif, comp: *c_int) enum { frame, end, fail } {
    // on first frame, any non-written pixels get the background colour (non-transparent)
    if (!gifHeader(s, g, comp)) return .fail; // stbi__g_failure_reason set by stbi__gif_header
    if (!mad3sizesValid(4, g.w, g.h, 0)) {
        _ = s.err("too large");
        return .fail;
    }
    const pcount: usize = @intCast(g.w * g.h);
    g.out = s.alloc(4 * pcount);
    g.background = s.alloc(4 * pcount);
    g.history = s.alloc(pcount);
    if (g.out == null or g.background == null or g.history == null) {
        _ = s.err("outofmem");
        return .fail;
    }

    while (true) {
        const tag = s.get8();
        switch (tag) {
            0x2C => { // Image Descriptor
                const x = s.get16le();
                const y = s.get16le();
                const w = s.get16le();
                const h = s.get16le();
                if (((x + w) > g.w) or ((y + h) > g.h)) {
                    _ = s.err("bad Image Descriptor");
                    return .fail;
                }

                g.line_size = g.w * 4;
                g.start_x = x * 4;
                g.start_y = y * g.line_size;
                g.max_x = g.start_x + w * 4;
                g.max_y = g.start_y + h * g.line_size;
                g.cur_x = g.start_x;
                g.cur_y = g.start_y;

                // if the width of the specified rectangle is 0, that means
                // we may not see *any* pixels or the image is malformed;
                // to make sure this is caught, move the current y down to
                // max_y (which is what out_gif_code checks).
                if (w == 0) g.cur_y = g.max_y;

                g.lflags = s.get8();

                if (g.lflags & 0x40 != 0) {
                    g.step = 8 * g.line_size; // first interlaced spacing
                    g.parse = 3;
                } else {
                    g.step = g.line_size;
                    g.parse = 0;
                }

                if (g.lflags & 0x80 != 0) {
                    gifParseColortable(s, &g.lpal, @as(c_int, 2) << @intCast(g.lflags & 7), if (g.eflags & 0x01 != 0) g.transparent else -1);
                    g.use_lpal = true;
                } else if (g.flags & 0x80 != 0) {
                    g.use_lpal = false;
                } else {
                    _ = s.err("missing color table");
                    return .fail;
                }

                if (!processGifRaster(s, g)) return .fail;

                // if this was the first frame,
                if (g.bgindex > 0) {
                    // if first frame, any pixel not drawn to gets the background color
                    for (0..pcount) |pi| {
                        if (g.history.?[pi] == 0) {
                            const bg: usize = @intCast(g.bgindex);
                            g.pal[bg][3] = 255; // just in case it was made transparent, undo that
                            @memcpy(g.out.?[pi * 4 ..][0..4], &g.pal[bg]);
                        }
                    }
                }
                return .frame;
            },
            0x21 => { // Comment Extension.
                const ext = s.get8();
                if (ext == 0xF9) { // Graphic Control Extension.
                    const len: c_int = s.get8();
                    if (len == 4) {
                        g.eflags = s.get8();
                        g.delay = 10 * s.get16le(); // delay - 1/100th of a second, saving as 1/1000ths.

                        // unset old transparent
                        if (g.transparent >= 0) g.pal[@intCast(g.transparent)][3] = 255;
                        if (g.eflags & 0x01 != 0) {
                            g.transparent = s.get8();
                            if (g.transparent >= 0) g.pal[@intCast(g.transparent)][3] = 0;
                        } else {
                            // don't need transparent
                            s.skip(1);
                            g.transparent = -1;
                        }
                    } else {
                        s.skip(len);
                        continue;
                    }
                }
                while (true) {
                    const len: c_int = s.get8();
                    if (len == 0) break;
                    s.skip(len);
                }
            },
            0x3B => return .end, // gif stream termination code
            else => {
                _ = s.err("unknown code");
                return .fail;
            },
        }
    }
}

/// Ports `stbi__gif_load` (stb_image.h:7048 @5266f24da).
fn gifLoadImpl(s: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
    const g = s.gpa.create(Gif) catch return s.errpuc("outofmem");
    defer s.gpa.destroy(g);
    g.* = .{};
    defer s.free(g.history);
    defer s.free(g.background);

    var u: ?[]u8 = null;
    if (gifLoadFirst(s, g, comp) == .frame) {
        u = g.out;
        x.* = g.w;
        y.* = g.h;

        // moved conversion to after successful load so that the same
        // can be done for multiple frames.
        if (req_comp != 0 and req_comp != 4) u = s.convertFormat(u.?, 4, req_comp, @intCast(g.w), @intCast(g.h));
    } else {
        // if there was an error and we allocated an image buffer, free it!
        s.free(g.out);
    }
    return u;
}

/// Ports `stbi__hdr_test_core` and `stbi__hdr_test` (stb_image.h:7087, 7097 @5266f24da).
fn hdrTestImpl(s: *Context) bool {
    const sigs = [_][]const u8{ "#?RADIANCE\n", "#?RGBE\n" };
    for (sigs) |sig| {
        var ok = true;
        for (sig) |ch| {
            if (s.get8() != ch) {
                ok = false;
                break;
            }
        }
        s.rewind();
        if (ok) return true;
    }
    return false;
}

const hdr_buflen = 1024;

/// Ports `stbi__hdr_gettoken` (stb_image.h:7109 @5266f24da). Return: the
/// token, borrowed from `buffer`.
fn hdrGettoken(z: *Context, buffer: *[hdr_buflen]u8) []const u8 {
    var len: usize = 0;
    var c = z.get8();

    while (!z.atEof() and c != '\n') {
        buffer[len] = c;
        len += 1;
        if (len == hdr_buflen - 1) {
            // flush to end of line
            while (!z.atEof() and z.get8() != '\n') {}
            break;
        }
        c = z.get8();
    }

    buffer[len] = 0;
    // a NUL inside the line ends the C string
    return buffer[0 .. std.mem.indexOfScalar(u8, buffer[0..len], 0) orelse len];
}

/// Ports `stbi__hdr_convert` (stb_image.h:7131 @5266f24da).
fn hdrConvert(output: []f32, input: *const [4]u8, req_comp: c_int) void {
    if (input[3] != 0) {
        // Exponent
        const f1 = ldexpf(1.0, @as(c_int, input[3]) - (128 + 8));
        if (req_comp <= 2) {
            output[0] = @as(f32, @floatFromInt(@as(c_int, input[0]) + input[1] + input[2])) * f1 / 3;
        } else {
            output[0] = @as(f32, @floatFromInt(input[0])) * f1;
            output[1] = @as(f32, @floatFromInt(input[1])) * f1;
            output[2] = @as(f32, @floatFromInt(input[2])) * f1;
        }
        if (req_comp == 2) output[1] = 1;
        if (req_comp == 4) output[3] = 1;
    } else {
        switch (req_comp) {
            4 => {
                output[3] = 1;
                output[0] = 0;
                output[1] = 0;
                output[2] = 0;
            },
            3 => {
                output[0] = 0;
                output[1] = 0;
                output[2] = 0;
            },
            2 => {
                output[1] = 1;
                output[0] = 0;
            },
            1 => output[0] = 0,
            else => {},
        }
    }
}

/// `strtol(token, &end, 10)` truncated to `int`, as the C uses it.
fn strtolInt(tok: []const u8, end: *usize) c_int {
    var i: usize = 0;
    while (i < tok.len and (tok[i] == ' ' or (tok[i] >= 9 and tok[i] <= 13))) i += 1;
    var neg = false;
    if (i < tok.len and (tok[i] == '+' or tok[i] == '-')) {
        neg = tok[i] == '-';
        i += 1;
    }
    const digits_start = i;
    var v: i64 = 0;
    var overflow = false;
    while (i < tok.len and tok[i] >= '0' and tok[i] <= '9') : (i += 1) {
        if (!overflow) {
            v = v * 10 + (tok[i] - '0');
            if (v > std.math.maxInt(i64) / 10 - 10) overflow = true;
        }
    }
    if (i == digits_start) {
        end.* = 0;
        return 0;
    }
    end.* = i;
    var r: i64 = if (overflow) (if (neg) std.math.minInt(i64) else std.math.maxInt(i64)) else if (neg) -v else v;
    if (overflow and !neg) r = std.math.maxInt(i64);
    return @truncate(r);
}

/// Ports `stbi__hdr_load` (stb_image.h:7158 @5266f24da).
fn hdrLoadImpl(s: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp_in: c_int) ?[]f32 {
    var buffer: [hdr_buflen]u8 = undefined;
    var valid = false;

    // Check identifier
    const header_token = hdrGettoken(s, &buffer);
    if (!std.mem.eql(u8, header_token, "#?RADIANCE") and !std.mem.eql(u8, header_token, "#?RGBE")) {
        s.failure = "not HDR";
        return null;
    }

    // Parse header
    while (true) {
        const token = hdrGettoken(s, &buffer);
        if (token.len == 0) break;
        if (std.mem.eql(u8, token, "FORMAT=32-bit_rle_rgbe")) valid = true;
    }

    if (!valid) {
        s.failure = "unsupported format";
        return null;
    }

    // Parse width and height
    var token = hdrGettoken(s, &buffer);
    if (!std.mem.startsWith(u8, token, "-Y ")) {
        s.failure = "unsupported data layout";
        return null;
    }
    token = token[3..];
    var end: usize = 0;
    const height = strtolInt(token, &end);
    token = token[end..];
    while (token.len > 0 and token[0] == ' ') token = token[1..];
    if (!std.mem.startsWith(u8, token, "+X ")) {
        s.failure = "unsupported data layout";
        return null;
    }
    token = token[3..];
    const width = strtolInt(token, &end);

    if (height > max_dimensions) {
        s.failure = "too large";
        return null;
    }
    if (width > max_dimensions) {
        s.failure = "too large";
        return null;
    }

    x.* = width;
    y.* = height;

    comp.* = 3;
    const req_comp: c_int = if (req_comp_in == 0) 3 else req_comp_in;

    if (!mad4sizesValid(width, height, req_comp, @sizeOf(f32), 0)) {
        s.failure = "too large";
        return null;
    }

    // Read data
    const rc: usize = @intCast(req_comp);
    const n: usize = @as(usize, @intCast(width)) * @as(usize, @intCast(height)) * rc;
    const hdr_data = s.gpa.alloc(f32, n) catch {
        s.failure = "outofmem";
        return null;
    };
    @memset(hdr_data, 0);
    const wu: usize = @intCast(width);
    const hu: usize = @intCast(height);

    // Load image data
    // image data is stored as some number of sca
    var flat_from: ?struct { j: usize, i: usize } = null;
    if (width < 8 or width >= 32768) {
        flat_from = .{ .j = 0, .i = 0 };
    } else {
        // Read RLE-encoded data
        var scanline: ?[]u8 = null;
        defer s.free(scanline);

        var j: usize = 0;
        rle: while (j < hu) : (j += 1) {
            const c1 = s.get8();
            const c2 = s.get8();
            var len: c_int = s.get8();
            if (c1 != 2 or c2 != 2 or (len & 0x80) != 0) {
                // not run-length encoded, so we have to actually use THIS data as a decoded
                // pixel (note this can't be a valid pixel--one of RGB must be >= 128)
                const rgbe = [4]u8{ c1, c2, @intCast(len), s.get8() };
                hdrConvert(hdr_data, &rgbe, req_comp);
                flat_from = .{ .j = 0, .i = 1 };
                break :rle; // yes, this makes no sense
            }
            len <<= 8;
            len |= s.get8();
            if (len != width) {
                s.gpa.free(hdr_data);
                s.failure = "invalid decoded scanline length";
                return null;
            }
            if (scanline == null) {
                scanline = s.mallocMad2(width, 4, 0) orelse {
                    s.gpa.free(hdr_data);
                    s.failure = "outofmem";
                    return null;
                };
            }
            const sl = scanline.?;

            for (0..4) |k| {
                var i: usize = 0;
                while (true) {
                    const nleft: c_int = width - @as(c_int, @intCast(i));
                    if (nleft <= 0) break;
                    var count = s.get8();
                    if (count > 128) {
                        // Run
                        const value = s.get8();
                        count -= 128;
                        if (count == 0 or count > nleft) {
                            s.gpa.free(hdr_data);
                            s.failure = "corrupt";
                            return null;
                        }
                        for (0..count) |_| {
                            sl[i * 4 + k] = value;
                            i += 1;
                        }
                    } else {
                        // Dump
                        if (count == 0 or count > nleft) {
                            s.gpa.free(hdr_data);
                            s.failure = "corrupt";
                            return null;
                        }
                        for (0..count) |_| {
                            sl[i * 4 + k] = s.get8();
                            i += 1;
                        }
                    }
                }
            }
            for (0..wu) |i| hdrConvert(hdr_data[(j * wu + i) * rc ..], sl[i * 4 ..][0..4], req_comp);
        }
    }
    if (flat_from) |start| {
        // Read flat data
        // A short read leaves `rgbe` holding what it held: the C's is an
        // uninitialized local, which the reference harness meets on a
        // zeroed stack (see `mtmd_decode_ref.cpp`), so the last pixel read
        // repeats and the first is zero.
        var rgbe: [4]u8 = .{ 0, 0, 0, 0 };
        var col0 = start.i;
        for (start.j..hu) |j| {
            for (col0..wu) |i| {
                if (!s.getn(&rgbe)) read_indeterminate = true;
                hdrConvert(hdr_data[j * wu * rc + i * rc ..], &rgbe, req_comp);
            }
            col0 = 0;
        }
    }
    return hdr_data;
}

/// Ports `stbi__pnm_test` (stb_image.h:7494 @5266f24da).
fn pnmTestImpl(s: *Context) bool {
    const p = s.get8();
    const t = s.get8();
    if (p != 'P' or (t != '5' and t != '6')) {
        s.rewind();
        return false;
    }
    return true;
}

/// Ports `stbi__pnm_isspace` (stb_image.h:7543 @5266f24da).
fn pnmIsspace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == 0x0b or c == 0x0c or c == '\r';
}

/// Ports `stbi__pnm_skip_whitespace` (stb_image.h:7548 @5266f24da).
fn pnmSkipWhitespace(s: *Context, c: *u8) void {
    while (true) {
        while (!s.atEof() and pnmIsspace(c.*)) c.* = s.get8();

        if (s.atEof() or c.* != '#') break;

        while (!s.atEof() and c.* != '\n' and c.* != '\r') c.* = s.get8();
    }
}

/// Ports `stbi__pnm_getinteger` (stb_image.h:7567 @5266f24da). `char` is
/// signed on Apple aarch64 and x86, unsigned on aarch64 Linux; the
/// comparison `*c > '7'` only differs for bytes 0x80 and up, which are
/// above '7' unsigned and below it signed.
fn pnmGetinteger(s: *Context, c: *u8) c_int {
    var value: c_int = 0;

    while (!s.atEof() and c.* >= '0' and c.* <= '9') {
        value = value * 10 + (c.* - '0');
        c.* = s.get8();
        const cv: c_int = if (char_signed) @as(i8, @bitCast(c.*)) else c.*;
        if (value > 214748364 or (value == 214748364 and cv > '7')) return s.err("integer parse overflow");
    }

    return value;
}

const char_signed = @typeInfo(c_char).int.signedness == .signed;

/// Ports `stbi__pnm_info` (stb_image.h:7581 @5266f24da).
fn pnmInfo(s: *Context, x: *c_int, y: *c_int, comp: *c_int) c_int {
    s.rewind();

    // Get identifier
    const p = s.get8();
    const t = s.get8();
    if (p != 'P' or (t != '5' and t != '6')) {
        s.rewind();
        return 0;
    }

    comp.* = if (t == '6') 3 else 1; // '5' is 1-component .pgm; '6' is 3-component .ppm

    var c = s.get8();
    pnmSkipWhitespace(s, &c);

    x.* = pnmGetinteger(s, &c); // read width
    if (x.* == 0) return s.err("invalid width");
    pnmSkipWhitespace(s, &c);

    y.* = pnmGetinteger(s, &c); // read height
    if (y.* == 0) return s.err("invalid width");
    pnmSkipWhitespace(s, &c);

    const maxv = pnmGetinteger(s, &c); // read max value
    if (maxv > 65535) return s.err("max value > 65535");
    if (maxv > 255) return 16;
    return 8;
}

/// Ports `stbi__pnm_load` (stb_image.h:7506 @5266f24da).
fn pnmLoadImpl(s: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int, bits: *c_int) ?[]u8 {
    var ix: c_int = 0;
    var iy: c_int = 0;
    var in_: c_int = 0;
    bits.* = pnmInfo(s, &ix, &iy, &in_);
    s.img_x = @bitCast(ix);
    s.img_y = @bitCast(iy);
    s.img_n = in_;
    if (bits.* == 0) return null;

    if (s.img_y > max_dimensions) return s.errpuc("too large");
    if (s.img_x > max_dimensions) return s.errpuc("too large");

    x.* = @bitCast(s.img_x);
    y.* = @bitCast(s.img_y);
    comp.* = s.img_n;

    const bpc = @divTrunc(bits.*, 8);
    if (!mad4sizesValid(s.img_n, @bitCast(s.img_x), @bitCast(s.img_y), bpc, 0)) return s.errpuc("too large");

    const out = s.mallocMad4(s.img_n, @bitCast(s.img_x), @bitCast(s.img_y), bpc, 0) orelse return s.errpuc("outofmem");
    if (!s.getn(out)) {
        s.gpa.free(out);
        return s.errpuc("bad PNM");
    }

    if (req_comp != 0 and req_comp != s.img_n) {
        if (bits.* == 16) {
            // the file's samples are big-endian and stb converts them as
            // native, unswapped: the bytes pass through untouched
            return s.convertFormat16(out, s.img_n, req_comp, s.img_x, s.img_y);
        }
        return s.convertFormat(out, s.img_n, req_comp, s.img_x, s.img_y);
    }
    return out;
}

/// Ports `stbi__jpeg_load` (stb_image.h:4028 @5266f24da).
fn jpegLoadImpl(s: *Context, x: *c_int, y: *c_int, comp: *c_int, req_comp: c_int) ?[]u8 {
    const j = jpegNew(s) orelse return s.errpuc("outofmem");
    defer s.gpa.destroy(j);
    return loadJpegImage(j, x, y, comp, req_comp);
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "an unknown buffer fails with stb's reason" {
    const r = loadFromMemory(std.testing.allocator, "hello, world", 3);
    try std.testing.expectEqualStrings("unknown image type", r.failed.?);
}

test "a binary PPM decodes to its own bytes" {
    const ppm = "P6\n2 1\n255\n\x01\x02\x03\x04\x05\x06";
    const r = loadFromMemory(std.testing.allocator, ppm, 3);
    defer std.testing.allocator.free(r.ok.pixels);
    try std.testing.expectEqual(@as(c_int, 2), r.ok.x);
    try std.testing.expectEqualSlices(u8, "\x01\x02\x03\x04\x05\x06", r.ok.pixels);
}
