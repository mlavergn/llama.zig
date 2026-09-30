//! The backend registry: which backends exist, which devices they expose, and
//! how a shared-library backend is found and loaded.
//!
//! # Provenance
//!
//! Ported from, in the reference checkout:
//!
//! - `llama.cpp/ggml/src/ggml-backend-reg.cpp` — the registry itself
//! - `llama.cpp/ggml/src/ggml-backend-dl.h`    — the `dl_*` wrappers, whose
//!                                               non-Windows bodies are three
//!                                               lines of `dlopen`/`dlsym`
//!
//! Both at v0.3.0 (`c1d0e7a00`). Each declaration below names the C++ it
//! replaces and the line it began at. This file exports the same C symbols
//! with the same signatures, so the C++ that calls into it links unchanged.
//!
//! # Why `ggml-backend-dl.cpp` comes along
//!
//! It is one of the four ggml translation units whose external contract is not
//! a C ABI: it exports `dl_load_library`, `dl_get_sym` and `dl_error` with C++
//! linkage, one of them taking a `std::filesystem::path &`. Zig emits no
//! mangled names, so those three cannot be provided from here.
//!
//! They do not have to be. `ggml-backend-reg.cpp` is their only caller in the
//! whole build, so porting the registry removes the last reference and the file
//! leaves the build with it. The three wrappers become the direct `dlopen`,
//! `dlsym` and `dlerror` calls they always were.
//!
//! # What this build actually registers
//!
//! Metal then CPU, in that order, and nothing else: every other
//! `register_backend` call in the C++ constructor sits behind a `GGML_USE_*`
//! define that `build/llamacpp.zig` does not set. The order is load-bearing —
//! `ggml_backend_init_best` prefers the first GPU device it finds, and device
//! indices are what `llama.cpp` passes around.
//!
//! The dynamic-loading path is ported in full even though this build is static
//! and finds nothing. It is reachable from the public API
//! (`ggml_backend_load`), and a path that silently did nothing would be worse
//! than one that looks for files that are not there.

const std = @import("std");
const builtin = @import("builtin");
const config = @import("config");
const impl = @import("impl.zig");
const c = impl.c;

/// Mirrors `GGML_USE_METAL`, which `build/llamacpp.zig` defines for the
/// library and not for the ported test root. The C++ constructor registers
/// Metal behind `#ifdef GGML_USE_METAL`; this is that `#ifdef`, and it has to
/// be a `comptime` branch rather than a runtime one so the reference to
/// `ggml_backend_metal_reg` is not emitted where nothing defines it.
const use_metal = config.use_metal;

/// True when the C++ build would leave `NDEBUG` unset, which gates the
/// registry's per-backend debug logging and the `silent` flag in
/// `ggml_backend_load_all_from_path`.
const debug_build = builtin.mode == .Debug;

/// The registry's vectors are private to this file and never cross the C ABI,
/// but there is no allocator to be handed one at a C entry point. libc's is
/// the same heap the rest of ggml uses.
const allocator = std.heap.c_allocator;

/// Decision 39, applied to the filesystem as well as to locks: Zig 0.16 puts
/// `getcwd` and directory iteration behind `std.Io`, which takes an `Io` on
/// every call, and a function entered through a C ABI has none to thread
/// through. libc's `opendir`/`readdir`/`getcwd` are what `std::filesystem`
/// calls underneath anyway, so this is the closer translation rather than a
/// substitution. `std.fs` still supplies `max_path_bytes`, which is only a
/// constant.
const Libc = struct {
    extern fn access(path: [*:0]const u8, mode: c_int) c_int;
    const F_OK: c_int = 0;
};

// -----------------------------------------------------------------------------
// Dynamic library loading
//
// Ports the non-Windows arm of `ggml-backend-dl.h` (ggml-backend-dl.h:29
// @c1d0e7a00) together with the bodies in `ggml-backend-dl.cpp`. Kept as
// file-private helpers rather than exported symbols: the C++ exported them
// with C++ linkage, which nothing outside `ggml-backend-reg.cpp` ever called.

/// Ports `dl_load_library` (ggml-backend-dl.cpp:34 @c1d0e7a00), the `#else` arm.
fn dlLoadLibrary(path: [*:0]const u8) ?*anyopaque {
    return std.c.dlopen(path, .{ .NOW = true, .LOCAL = true });
}

/// Ports `dl_get_sym` (ggml-backend-dl.cpp:39 @c1d0e7a00), the `#else` arm.
fn dlGetSym(handle: *anyopaque, name: [*:0]const u8) ?*anyopaque {
    return std.c.dlsym(handle, name);
}

/// Ports `dl_error` (ggml-backend-dl.cpp:43 @c1d0e7a00), the `#else` arm.
///
/// Return: the pending dynamic-linker error, or `""` when there is none.
/// Borrowed from libc and valid only until the next `dlopen`/`dlsym`.
fn dlError() [*:0]const u8 {
    return std.c.dlerror() orelse "";
}

// -----------------------------------------------------------------------------
// The registry

/// Mirrors `struct ggml_backend_reg_entry` (src/ggml-backend-reg.cpp:110 @c1d0e7a00).
///
/// The C++ holds the handle in a `dl_handle_ptr`, a `unique_ptr` whose deleter
/// calls `dlclose`. That deleter never runs: the destructor `release()`s every
/// entry, deliberately, because backend threads may still be inside the
/// library. So the handle is recorded and never closed, which is what this
/// plain pointer does without pretending otherwise.
/// `ggml_backend_metal_reg`, from `ggml-metal.h`. Declared rather than
/// imported: `src/ggml/impl.zig`'s `@cImport` does not take that header, and
/// this is its only use. `ggml_backend_cpu_reg` comes through the import,
/// since `ggml-cpu.h` is already in it.
extern fn ggml_backend_metal_reg() c.ggml_backend_reg_t;

const Entry = struct {
    reg: c.ggml_backend_reg_t,
    handle: ?*anyopaque,
};

/// Mirrors `struct ggml_backend_registry` (src/ggml-backend-reg.cpp:115 @c1d0e7a00).
const Registry = struct {
    backends: std.ArrayList(Entry) = .empty,
    devices: std.ArrayList(c.ggml_backend_dev_t) = .empty,

    /// Ports `ggml_backend_registry`'s constructor
    /// (src/ggml-backend-reg.cpp:119 @c1d0e7a00).
    ///
    /// Only the `GGML_USE_METAL` and `GGML_USE_CPU` arms are compiled here, in
    /// the C++'s order: Metal first, CPU last.
    ///
    /// Return: nothing. Registration failures are not reportable through the
    /// C++ constructor either.
    fn init(self: *Registry) void {
        if (use_metal) self.registerBackend(ggml_backend_metal_reg(), null);
        self.registerBackend(c.ggml_backend_cpu_reg(), null);
    }

    /// Ports `ggml_backend_registry`'s `register_backend`
    /// (src/ggml-backend-reg.cpp:186 @c1d0e7a00).
    ///
    /// Parameters:
    /// - `reg`: the backend to add. A null or already-present `reg` is ignored,
    ///   as in the C++.
    /// - `handle`: the shared library it came from, or null for a built-in.
    ///
    /// Return: nothing.
    fn registerBackend(self: *Registry, reg: c.ggml_backend_reg_t, handle: ?*anyopaque) void {
        if (reg == null) return;

        for (self.backends.items) |entry| {
            if (entry.reg == reg) return;
        }

        if (debug_build) {
            impl.logDebug("%s: registered backend %s (%zu devices)\n", .{
                "register_backend",
                c.ggml_backend_reg_name(reg),
                c.ggml_backend_reg_dev_count(reg),
            });
        }

        self.backends.append(allocator, .{ .reg = reg, .handle = handle }) catch
            impl.abort("failed to grow the backend registry");

        var i: usize = 0;
        const n = c.ggml_backend_reg_dev_count(reg);
        while (i < n) : (i += 1) {
            self.registerDevice(c.ggml_backend_reg_dev_get(reg, i));
        }
    }

    /// Ports `ggml_backend_registry`'s `register_device`
    /// (src/ggml-backend-reg.cpp:207 @c1d0e7a00).
    ///
    /// Parameters:
    /// - `device`: the device to add; ignored if already present.
    ///
    /// Return: nothing.
    fn registerDevice(self: *Registry, device: c.ggml_backend_dev_t) void {
        for (self.devices.items) |dev| {
            if (dev == device) return;
        }

        if (debug_build) {
            impl.logDebug("%s: registered device %s (%s)\n", .{
                "register_device",
                c.ggml_backend_dev_name(device),
                c.ggml_backend_dev_description(device),
            });
        }

        self.devices.append(allocator, device) catch
            impl.abort("failed to grow the device registry");
    }

    /// Ports `ggml_backend_registry`'s `load_backend`
    /// (src/ggml-backend-reg.cpp:220 @c1d0e7a00).
    ///
    /// Parameters:
    /// - `path`: the shared library to open.
    /// - `silent`: suppress the error paths' logging, as the C++ flag does.
    ///
    /// Return: the registered backend, or null if the library would not load,
    /// scored zero, exposed no `ggml_backend_init`, or reported an
    /// incompatible API version.
    fn loadBackend(self: *Registry, path: [*:0]const u8, silent: bool) c.ggml_backend_reg_t {
        const handle = dlLoadLibrary(path) orelse {
            if (!silent) {
                impl.logError("%s: failed to load %s: %s\n", .{ "load_backend", path, dlError() });
            }
            return null;
        };

        // The C++ lets `handle` fall out of scope on every early return, and
        // its deleter closes the library. Ours has to say so.
        var adopted = false;
        defer if (!adopted) {
            _ = std.c.dlclose(handle);
        };

        if (dlGetSym(handle, "ggml_backend_score")) |sym| {
            const score_fn: c.ggml_backend_score_t = @ptrCast(@alignCast(sym));
            if (score_fn.?() == 0) {
                if (!silent) {
                    impl.logInfo("%s: backend %s is not supported on this system\n", .{ "load_backend", path });
                }
                return null;
            }
        }

        const init_sym = dlGetSym(handle, "ggml_backend_init") orelse {
            if (!silent) {
                impl.logError("%s: failed to find ggml_backend_init in %s\n", .{ "load_backend", path });
            }
            return null;
        };
        const backend_init_fn: c.ggml_backend_init_t = @ptrCast(@alignCast(init_sym));

        const reg = backend_init_fn.?();
        if (reg == null or reg.*.api_version != c.GGML_BACKEND_API_VERSION) {
            if (!silent) {
                if (reg == null) {
                    impl.logError(
                        "%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        .{ "load_backend", path },
                    );
                } else {
                    impl.logError(
                        "%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        .{ "load_backend", path, reg.*.api_version, @as(c_int, c.GGML_BACKEND_API_VERSION) },
                    );
                }
            }
            return null;
        }

        impl.logInfo("%s: loaded %s backend from %s\n", .{
            "load_backend",
            c.ggml_backend_reg_name(reg),
            path,
        });

        adopted = true;
        self.registerBackend(reg, handle);
        return reg;
    }

    /// Ports `ggml_backend_registry`'s `unload_backend`
    /// (src/ggml-backend-reg.cpp:266 @c1d0e7a00).
    ///
    /// Parameters:
    /// - `reg`: the backend to drop, along with every device it owns.
    /// - `silent`: suppress the "not found" message.
    ///
    /// Return: nothing. As in the C++, the library itself is never `dlclose`d.
    fn unloadBackend(self: *Registry, reg: c.ggml_backend_reg_t, silent: bool) void {
        const found = for (self.backends.items, 0..) |entry, i| {
            if (entry.reg == reg) break i;
        } else {
            if (!silent) impl.logError("%s: backend not found\n", .{"unload_backend"});
            return;
        };

        if (!silent) {
            impl.logDebug("%s: unloading %s backend\n", .{ "unload_backend", c.ggml_backend_reg_name(reg) });
        }

        // `std::remove_if` then `erase`: drop every device belonging to `reg`
        // while keeping the rest in order.
        var kept: usize = 0;
        for (self.devices.items) |dev| {
            if (c.ggml_backend_dev_backend_reg(dev) == reg) continue;
            self.devices.items[kept] = dev;
            kept += 1;
        }
        self.devices.shrinkRetainingCapacity(kept);

        _ = self.backends.orderedRemove(found);
    }
};

/// Ports the function-local static in `get_reg` (src/ggml-backend-reg.cpp:292
/// @c1d0e7a00).
///
/// C++ magic statics are initialised exactly once under `__cxa_guard_acquire`;
/// this is that guard, written out. The fast path is a single acquire load, so
/// the lock is touched only on the first call — matching the C++, where every
/// call after the first is lock-free.
var registry: Registry = .{};
var registry_ready = std.atomic.Value(bool).init(false);
var registry_mutex: std.c.pthread_mutex_t = .{};

/// Ports `get_reg` (src/ggml-backend-reg.cpp:292 @c1d0e7a00).
///
/// Return: the process-wide registry, constructed on first use. Borrowed for
/// the life of the process.
fn getReg() *Registry {
    if (registry_ready.load(.acquire)) return &registry;

    _ = std.c.pthread_mutex_lock(&registry_mutex);
    defer {
        _ = std.c.pthread_mutex_unlock(&registry_mutex);
    }

    if (!registry_ready.load(.acquire)) {
        registry.init();
        registry_ready.store(true, .release);
    }
    return &registry;
}

// -----------------------------------------------------------------------------
// Internal API

/// Ports `ggml_backend_register` (src/ggml-backend-reg.cpp:298 @c1d0e7a00).
export fn ggml_backend_register(reg: c.ggml_backend_reg_t) callconv(.c) void {
    getReg().registerBackend(reg, null);
}

/// Ports `ggml_backend_device_register` (src/ggml-backend-reg.cpp:302 @c1d0e7a00).
export fn ggml_backend_device_register(device: c.ggml_backend_dev_t) callconv(.c) void {
    getReg().registerDevice(device);
}

// -----------------------------------------------------------------------------
// Backend enumeration

/// Ports `striequals` (src/ggml-backend-reg.cpp:307 @c1d0e7a00).
///
/// `std::tolower` on a `char`, which this reproduces with ASCII folding. Both
/// stop at the first difference and then require both strings to have ended.
fn striequals(a: [*:0]const u8, b: [*:0]const u8) bool {
    var i: usize = 0;
    while (a[i] != 0 and b[i] != 0) : (i += 1) {
        if (std.ascii.toLower(a[i]) != std.ascii.toLower(b[i])) return false;
    }
    return a[i] == b[i];
}

/// Ports `ggml_backend_reg_count` (src/ggml-backend-reg.cpp:316 @c1d0e7a00).
///
/// Return: how many backends are registered.
export fn ggml_backend_reg_count() callconv(.c) usize {
    return getReg().backends.items.len;
}

/// Ports `ggml_backend_reg_get` (src/ggml-backend-reg.cpp:320 @c1d0e7a00).
///
/// Parameters:
/// - `index`: position in registration order; must be in range.
///
/// Return: the backend at `index`. Borrowed; owned by the registry.
export fn ggml_backend_reg_get(index: usize) callconv(.c) c.ggml_backend_reg_t {
    impl.assert(index < ggml_backend_reg_count(), "index < ggml_backend_reg_count()");
    return getReg().backends.items[index].reg;
}

/// Ports `ggml_backend_reg_by_name` (src/ggml-backend-reg.cpp:325 @c1d0e7a00).
///
/// Parameters:
/// - `name`: backend name, matched case-insensitively.
///
/// Return: the backend, or null if no name matches.
export fn ggml_backend_reg_by_name(name: [*:0]const u8) callconv(.c) c.ggml_backend_reg_t {
    var i: usize = 0;
    while (i < ggml_backend_reg_count()) : (i += 1) {
        const reg = ggml_backend_reg_get(i);
        if (striequals(c.ggml_backend_reg_name(reg), name)) return reg;
    }
    return null;
}

// -----------------------------------------------------------------------------
// Device enumeration

/// Ports `ggml_backend_dev_count` (src/ggml-backend-reg.cpp:336 @c1d0e7a00).
///
/// Return: how many devices every registered backend exposes in total.
export fn ggml_backend_dev_count() callconv(.c) usize {
    return getReg().devices.items.len;
}

/// Ports `ggml_backend_dev_get` (src/ggml-backend-reg.cpp:340 @c1d0e7a00).
///
/// Parameters:
/// - `index`: position in registration order; must be in range.
///
/// Return: the device at `index`. Borrowed; owned by its backend.
export fn ggml_backend_dev_get(index: usize) callconv(.c) c.ggml_backend_dev_t {
    impl.assert(index < ggml_backend_dev_count(), "index < ggml_backend_dev_count()");
    return getReg().devices.items[index];
}

/// Ports `ggml_backend_dev_by_name` (src/ggml-backend-reg.cpp:345 @c1d0e7a00).
///
/// Parameters:
/// - `name`: device name, matched case-insensitively.
///
/// Return: the device, or null if no name matches.
export fn ggml_backend_dev_by_name(name: [*:0]const u8) callconv(.c) c.ggml_backend_dev_t {
    var i: usize = 0;
    while (i < ggml_backend_dev_count()) : (i += 1) {
        const dev = ggml_backend_dev_get(i);
        if (striequals(c.ggml_backend_dev_name(dev), name)) return dev;
    }
    return null;
}

/// Ports `ggml_backend_dev_by_type` (src/ggml-backend-reg.cpp:355 @c1d0e7a00).
///
/// Parameters:
/// - `dev_type`: the class of device wanted.
///
/// Return: the first device of that type in registration order, or null.
export fn ggml_backend_dev_by_type(dev_type: c.enum_ggml_backend_dev_type) callconv(.c) c.ggml_backend_dev_t {
    var i: usize = 0;
    while (i < ggml_backend_dev_count()) : (i += 1) {
        const dev = ggml_backend_dev_get(i);
        if (c.ggml_backend_dev_type(dev) == dev_type) return dev;
    }
    return null;
}

// -----------------------------------------------------------------------------
// Convenience constructors

/// Ports `ggml_backend_init_by_name` (src/ggml-backend-reg.cpp:366 @c1d0e7a00).
///
/// Parameters:
/// - `name`: device name, matched case-insensitively.
/// - `params`: backend-specific initialisation string, may be null.
///
/// Return: a new backend instance, or null if no such device. Owned by the
/// caller, released with `ggml_backend_free`.
export fn ggml_backend_init_by_name(name: [*:0]const u8, params: ?[*:0]const u8) callconv(.c) c.ggml_backend_t {
    const dev = ggml_backend_dev_by_name(name) orelse return null;
    return c.ggml_backend_dev_init(dev, params);
}

/// Ports `ggml_backend_init_by_type` (src/ggml-backend-reg.cpp:374 @c1d0e7a00).
///
/// Parameters:
/// - `dev_type`: the class of device wanted.
/// - `params`: backend-specific initialisation string, may be null.
///
/// Return: a new backend instance, or null. Owned by the caller.
export fn ggml_backend_init_by_type(
    dev_type: c.enum_ggml_backend_dev_type,
    params: ?[*:0]const u8,
) callconv(.c) c.ggml_backend_t {
    const dev = ggml_backend_dev_by_type(dev_type) orelse return null;
    return c.ggml_backend_dev_init(dev, params);
}

/// Ports `ggml_backend_init_best` (src/ggml-backend-reg.cpp:382 @c1d0e7a00).
///
/// Return: a backend on the best available device — discrete GPU, then
/// integrated GPU, then CPU — or null if nothing is registered. Owned by the
/// caller.
export fn ggml_backend_init_best() callconv(.c) c.ggml_backend_t {
    var dev = ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == null) dev = ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (dev == null) dev = ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev == null) return null;
    return c.ggml_backend_dev_init(dev, null);
}

// -----------------------------------------------------------------------------
// Dynamic loading
//
// Ported in full though this build is static: the entry points are public API.
// The C++ works in `std::filesystem::path`; paths here are null-terminated
// byte buffers, because every consumer of one is either `dlopen` or a log
// format string.

/// Ports `ggml_backend_load` (src/ggml-backend-reg.cpp:393 @c1d0e7a00).
///
/// Parameters:
/// - `path`: the shared library to load.
///
/// Return: the registered backend, or null on any failure. Borrowed; owned by
/// the registry.
export fn ggml_backend_load(path: [*:0]const u8) callconv(.c) c.ggml_backend_reg_t {
    return getReg().loadBackend(path, false);
}

/// Ports `ggml_backend_unload` (src/ggml-backend-reg.cpp:397 @c1d0e7a00).
///
/// Parameters:
/// - `reg`: the backend to drop from the registry.
///
/// Return: nothing. The library stays mapped, as it does in the C++.
export fn ggml_backend_unload(reg: c.ggml_backend_reg_t) callconv(.c) void {
    getReg().unloadBackend(reg, true);
}

/// `_NSGetExecutablePath`, from `<mach-o/dyld.h>`. Declared rather than
/// imported: the header pulls in the Mach-O structures, and this is its only
/// use.
extern fn _NSGetExecutablePath(buf: [*]u8, bufsize: *u32) c_int;

/// Ports `get_executable_path` (src/ggml-backend-reg.cpp:401 @c1d0e7a00), the
/// `__APPLE__` arm.
///
/// Parameters:
/// - `buf`: receives the directory, null-terminated, with a trailing `/`.
///
/// Return: the slice of `buf` written, or null if the path did not fit. The
/// C++ grows a `std::vector` until the call succeeds; the fixed buffer here is
/// `std.fs.max_path_bytes`, which the call cannot exceed.
fn getExecutablePath(buf: []u8) ?[:0]const u8 {
    var size: u32 = @intCast(buf.len);
    if (_NSGetExecutablePath(buf.ptr, &size) != 0) return null;

    // `_NSGetExecutablePath` null-terminates but only updates `size` when it
    // fails, so the length has to come from the string itself.
    const full = std.mem.sliceTo(buf[0..], 0);

    // "remove executable name", keeping the separator: the C++ appends "/" to
    // the truncated string, so a path with no slash at all becomes "/".
    const cut = if (std.mem.lastIndexOfScalar(u8, full, '/')) |i| i else 0;
    if (cut + 2 > buf.len) return null;
    buf[cut] = '/';
    buf[cut + 1] = 0;
    return buf[0 .. cut + 1 :0];
}

/// Ports `backend_filename_prefix` (src/ggml-backend-reg.cpp:464 @c1d0e7a00) and
/// `backend_filename_extension` (src/ggml-backend-reg.cpp:472 @c1d0e7a00), the
/// non-Windows arms of both.
const backend_filename_prefix = "libggml-";
const backend_filename_extension = ".so";

/// Ports `ggml_backend_load_best` (src/ggml-backend-reg.cpp:480 @c1d0e7a00).
///
/// Scans each search path for `libggml-<name>-*.so`, scores every candidate
/// through its `ggml_backend_score`, and loads the highest. If nothing scores,
/// falls back to a plain `libggml-<name>.so`.
///
/// Parameters:
/// - `name`: the backend family, e.g. `"metal"`.
/// - `silent`: suppress logging on the failure paths.
/// - `user_search_path`: a single directory to search instead of the defaults.
///
/// Return: the loaded backend, or null if none was found. Borrowed; owned by
/// the registry.
fn loadBest(name: [*:0]const u8, silent: bool, user_search_path: ?[*:0]const u8) c.ggml_backend_reg_t {
    var prefix_buf: [std.fs.max_path_bytes]u8 = undefined;
    const file_prefix = std.fmt.bufPrint(
        &prefix_buf,
        backend_filename_prefix ++ "{s}-",
        .{name},
    ) catch return null;

    // The C++ builds `search_paths` as a vector; two entries is the most this
    // configuration ever has, and `GGML_BACKEND_DIR` is not defined here.
    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    var search_paths: [2][]const u8 = undefined;
    var n_paths: usize = 0;

    if (user_search_path) |p| {
        search_paths[0] = std.mem.sliceTo(p, 0);
        n_paths = 1;
    } else {
        if (getExecutablePath(&exe_buf)) |p| {
            search_paths[n_paths] = p;
            n_paths += 1;
        }
        if (std.c.getcwd(&cwd_buf, cwd_buf.len)) |p| {
            search_paths[n_paths] = std.mem.sliceTo(p, 0);
            n_paths += 1;
        }
    }

    var best_score: c_int = 0;
    var best_buf: [std.fs.max_path_bytes]u8 = undefined;
    var best_len: usize = 0;

    for (search_paths[0..n_paths]) |search_path| {
        var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_path = std.fmt.bufPrintZ(&dir_buf, "{s}", .{search_path}) catch continue;

        // `fs::exists` then `directory_iterator`; `opendir` failing covers
        // both, including the C++'s `skip_permission_denied`.
        const dir = std.c.opendir(dir_path.ptr) orelse {
            impl.printDebug("%s: search path %s does not exist\n", .{ "load_best", dir_path.ptr });
            continue;
        };
        defer _ = std.c.closedir(dir);

        while (std.c.readdir(dir)) |ent| {
            // DT_REG. `is_regular_file` in the C++; a `DT_UNKNOWN` filesystem
            // would need a stat, which APFS never reports.
            if (ent.type != 8) continue;
            const entry_name = ent.name[0..ent.namlen];
            if (!std.mem.startsWith(u8, entry_name, file_prefix)) continue;
            if (!std.mem.endsWith(u8, entry_name, backend_filename_extension)) continue;

            var cand_buf: [std.fs.max_path_bytes]u8 = undefined;
            const cand = std.fmt.bufPrintZ(&cand_buf, "{s}/{s}", .{ search_path, entry_name }) catch continue;

            const handle = dlLoadLibrary(cand.ptr) orelse {
                if (!silent) {
                    impl.logError("%s: failed to load %s: %s\n", .{ "load_best", cand.ptr, dlError() });
                }
                continue;
            };
            // The C++ lets the `unique_ptr` close it at the end of the
            // iteration: the winner is re-opened by `load_backend` below.
            defer _ = std.c.dlclose(handle);

            const sym = dlGetSym(handle, "ggml_backend_score") orelse {
                if (!silent) {
                    impl.logInfo("%s: failed to find ggml_backend_score in %s\n", .{ "load_best", cand.ptr });
                }
                continue;
            };
            const score_fn: c.ggml_backend_score_t = @ptrCast(@alignCast(sym));
            const s = score_fn.?();
            impl.printDebug("%s: %s score: %d\n", .{ "load_best", cand.ptr, s });
            if (s > best_score) {
                best_score = s;
                @memcpy(best_buf[0..cand.len], cand);
                best_buf[cand.len] = 0;
                best_len = cand.len;
            }
        }
    }

    if (best_score == 0) {
        // "try to load the base backend": `libggml-<name>.so`, unversioned.
        for (search_paths[0..n_paths]) |search_path| {
            var path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const path = std.fmt.bufPrintZ(
                &path_buf,
                "{s}/" ++ backend_filename_prefix ++ "{s}" ++ backend_filename_extension,
                .{ search_path, name },
            ) catch continue;
            if (Libc.access(path.ptr, Libc.F_OK) != 0) continue;
            return getReg().loadBackend(path.ptr, silent);
        }
        return null;
    }

    best_buf[best_len] = 0;
    const best_path: [*:0]const u8 = @ptrCast(&best_buf);
    return getReg().loadBackend(best_path, silent);
}

/// The families `ggml_backend_load_all_from_path` probes, in the C++'s order.
/// CPU is last so a more capable backend wins the scoring.
const backend_families = [_][*:0]const u8{
    "blas", "zendnn",   "cann",   "cuda",    "hip",    "metal",
    "rpc",  "sycl",     "vulkan", "virtgpu", "opencl", "hexagon",
    "musa", "openvino", "cpu",
};

/// Ports `ggml_backend_load_all` (src/ggml-backend-reg.cpp:562 @c1d0e7a00).
export fn ggml_backend_load_all() callconv(.c) void {
    ggml_backend_load_all_from_path(null);
}

/// Ports `ggml_backend_load_all_from_path` (src/ggml-backend-reg.cpp:566 @c1d0e7a00).
///
/// Parameters:
/// - `dir_path`: a single directory to search, or null for the defaults.
///
/// Return: nothing. Every family that is not present is skipped silently in a
/// release build, as the C++'s `NDEBUG` arm does.
export fn ggml_backend_load_all_from_path(dir_path: ?[*:0]const u8) callconv(.c) void {
    const silent = !debug_build;

    for (backend_families) |family| {
        _ = loadBest(family, silent, dir_path);
    }

    // An out-of-tree backend named outright by the environment.
    if (std.c.getenv("GGML_BACKEND_PATH")) |backend_path| {
        _ = ggml_backend_load(backend_path);
    }
}

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}

test "striequals folds case and requires both strings to end" {
    try std.testing.expect(striequals("Metal", "metal"));
    try std.testing.expect(striequals("CPU", "cpu"));
    try std.testing.expect(striequals("", ""));
    // The C++ compares until one string ends, then requires both to have.
    try std.testing.expect(!striequals("metal", "metalx"));
    try std.testing.expect(!striequals("metalx", "metal"));
    try std.testing.expect(!striequals("metal", "cpu"));
}

test "the registry reports the two backends this build compiles" {
    // Metal first, CPU last -- the order the C++ constructor registers them,
    // and what `ggml_backend_init_best` depends on. The ported test root
    // compiles without Metal, so it sees one.
    try std.testing.expectEqual(@as(usize, if (use_metal) 2 else 1), ggml_backend_reg_count());

    if (use_metal) {
        // The Metal backend registers itself as `GGML_METAL_NAME`, which is
        // "MTL", not "Metal" (ggml-metal/ggml-metal.cpp:14 @c1d0e7a00). Asserting the real names is
        // the point: the first version of this test guessed "Metal" and failed.
        try std.testing.expectEqualStrings("MTL", std.mem.sliceTo(c.ggml_backend_reg_name(ggml_backend_reg_get(0)), 0));
        try std.testing.expect(ggml_backend_reg_by_name("MTL") != null);
        // Lookup folds case, as `striequals` does.
        try std.testing.expect(ggml_backend_reg_by_name("mtl") != null);
    }

    const cpu = ggml_backend_reg_get(ggml_backend_reg_count() - 1);
    try std.testing.expectEqualStrings("CPU", std.mem.sliceTo(c.ggml_backend_reg_name(cpu), 0));
    try std.testing.expect(ggml_backend_reg_by_name("CPU") != null);
    try std.testing.expect(ggml_backend_reg_by_name("cpu") != null);
    try std.testing.expect(ggml_backend_reg_by_name("nonesuch") == null);
}

test "every registered device is reachable by name and by type" {
    try std.testing.expect(ggml_backend_dev_count() >= 1);

    var i: usize = 0;
    while (i < ggml_backend_dev_count()) : (i += 1) {
        const dev = ggml_backend_dev_get(i);
        try std.testing.expectEqual(dev, ggml_backend_dev_by_name(c.ggml_backend_dev_name(dev)));
    }

    // This build always has a CPU device; `init_best` falls back to it.
    try std.testing.expect(ggml_backend_dev_by_type(c.GGML_BACKEND_DEVICE_TYPE_CPU) != null);
}

test "loading a library that does not exist fails rather than registering" {
    const before = ggml_backend_reg_count();
    try std.testing.expect(ggml_backend_load("/nonexistent/libggml-nosuch.so") == null);
    try std.testing.expectEqual(before, ggml_backend_reg_count());
}

test "scanning for absent backends leaves the registry untouched" {
    const before_regs = ggml_backend_reg_count();
    const before_devs = ggml_backend_dev_count();
    // A directory that exists and holds no `libggml-*.so`.
    ggml_backend_load_all_from_path("/tmp");
    try std.testing.expectEqual(before_regs, ggml_backend_reg_count());
    try std.testing.expectEqual(before_devs, ggml_backend_dev_count());
}
