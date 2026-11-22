//! Barrel for the vendored decoders mtmd-helper reaches: `stb_image` and
//! miniaudio's decoder.
//!
//! **Not a port.** It gathers the files under `vendor/`.

const std = @import("std");

pub const stb_image = @import("stb_image.zig");
pub const miniaudio = @import("miniaudio.zig");

pub const audio_io = @import("audio_io.zig");

pub const dr_wav = @import("dr_wav.zig");

test {
    std.testing.refAllDecls(@This());
}
