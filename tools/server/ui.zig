//! The web UI's embedded assets, as `server-http.cpp` asks for them.
//!
//! # Provenance
//!
//! Ports `llama.cpp/tools/ui/ui.cpp.in` and `llama.cpp/tools/ui/ui.h.in`
//! (v0.5.0, `7fe450e19`) -- the templates upstream's
//! `llama.cpp/scripts/ui-assets.cmake` fills with the built SvelteKit bundle.
//! The bundle itself is never ported: it is JavaScript, CSS and images the
//! browser runs, and travels as data, as the Metal kernels do.
//!
//! The table is the `ui_assets` module, generated at build time by
//! `build/ui_assets.zig` from `-Dui=<dist>` (an empty one without it): one
//! entry per file of the bundle, sorted by name, each gzip-compressed when
//! `use_gzip`, its ETag the quoted SHA-256 of the embedded bytes. Upstream's
//! `#cmakedefine LLAMA_UI_HAS_ASSETS` is `has_assets`, a comptime constant,
//! so a build without a UI compiles the no-assets path alone, as upstream's
//! does.

const std = @import("std");
const generated = @import("ui_assets");

/// Ports `llama_ui_asset` (ui.h.in:11 @7fe450e19): one embedded file. `data`
/// carries `size`.
pub const Asset = generated.Asset;

/// `LLAMA_UI_HAS_ASSETS` (ui.h.in:9 @7fe450e19): defined only when assets
/// were embedded.
pub const has_assets = generated.assets.len > 0;

/// Ports `llama_ui_find_asset` (ui.cpp.in:12 @7fe450e19): the asset named
/// `name`, by a linear search as the C's.
///
/// Parameters:
/// - `name`: the asset's path in the bundle, with no leading `/`.
///
/// Return: the table's entry, static; null when there is none.
pub fn findAsset(name: []const u8) ?*const Asset {
    if (!has_assets) return null;
    for (&generated.assets) |*a| {
        if (std.mem.eql(u8, a.name, name)) return a;
    }
    return null;
}

/// Ports `llama_ui_get_assets` (ui.cpp.in:25 @7fe450e19).
///
/// Return: the whole table, static; empty without a UI.
pub fn getAssets() []const Asset {
    return &generated.assets;
}

/// Ports `llama_ui_use_gzip` (ui.cpp.in:34 @7fe450e19): whether every
/// asset is stored gzip-compressed and served with `Content-Encoding: gzip`.
pub fn useGzip() bool {
    return generated.use_gzip;
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "findAsset agrees with the table" {
    for (getAssets()) |*a| try std.testing.expectEqual(a, findAsset(a.name).?);
    try std.testing.expectEqual(@as(?*const Asset, null), findAsset("no/such/asset"));
}
