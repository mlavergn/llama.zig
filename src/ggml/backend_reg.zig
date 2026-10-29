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
//! Both at v0.4.1 (`b29c606e2`). Each declaration below names the C++ it
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
//! CUDA (with `-Dcuda`), Metal (on Apple targets), Vulkan (with `-Dvulkan`),
//! then CPU, in that order, and
//! nothing else: every other `register_backend` call in the C++ constructor
//! sits behind a `GGML_USE_*` define that `build/llamacpp.zig` does not set. The order is load-bearing —
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
const win32 = @import("win32.zig");
const c = impl.c;

/// Mirrors `GGML_USE_METAL`, which `build/llamacpp.zig` defines for the
/// library and not for the ported test root. The C++ constructor registers
/// Metal behind `#ifdef GGML_USE_METAL`; this is that `#ifdef`, and it has to
/// be a `comptime` branch rather than a runtime one so the reference to
/// `ggml_backend_metal_reg` is not emitted where nothing defines it.
const use_metal = config.use_metal;

/// Mirrors `GGML_USE_CUDA`, which `build/llamacpp.zig` defines when built with
/// `-Dcuda`. The CUDA backend is upstream's `ggml-cuda`, compiled unmodified by
/// `nvcc`; see `build/llamacpp.zig`'s `addCuda`. Comptime for the same reason as
/// `use_metal`.
const use_cuda = config.use_cuda;

/// Mirrors `GGML_USE_VULKAN`, which `build/llamacpp.zig` defines when built
/// with `-Dvulkan`. Comptime for the same reason as `use_metal`.
const use_vulkan = config.use_vulkan;

/// Mirrors `GGML_USE_BLAS`, which `build/llamacpp.zig` defines when built
/// with `-Dblas`. Comptime for the same reason as `use_metal`.
const use_blas = config.use_blas;

/// True when the C++ build would leave `NDEBUG` unset, which gates the
/// registry's per-backend debug logging and the `silent` flag in
/// `ggml_backend_load_all_from_path`.
const debug_build = builtin.mode == .debug;

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
    extern fn strerror(errnum: c_int) [*:0]const u8;
    const F_OK: c_int = 0;
};

// -----------------------------------------------------------------------------
// Dynamic library loading
//
// Ports both arms of `ggml-backend-dl.h`, whose `dl_handle`, `dl_handle`
// (ggml-backend-dl.h:20, 30 @b29c606e2) is an `HMODULE`'s pointee on Windows
// and a plain `void` elsewhere, together with the bodies in
// `ggml-backend-dl.cpp`. Kept as file-private helpers rather than exported
// symbols: the C++ exported them with C++ linkage, which nothing outside
// `ggml-backend-reg.cpp` ever called.
//
// Paths stay UTF-8 here on every platform. The C++ holds them as `fs::path`,
// which on Windows is a wide string made by `fs::u8path` and turned back with
// `path_str`'s `u8string()` for every log line, so the port widens a path only
// where it reaches a Win32 call.

/// `_WIN32`.
const is_windows = builtin.os.tag == .windows;

/// `fs::u8path(path).wstring()`: a UTF-8 path as the wide string a Win32 call
/// takes. Null for a path that is not UTF-8. Owned by the caller; free with
/// `allocator.free`.
fn widen(path: []const u8) ?[:0]u16 {
    return std.unicode.utf8ToUtf16LeAllocZ(allocator, path) catch null;
}

/// Ports `dl_load_library`, `dl_load_library` (ggml-backend-dl.cpp:5, 34
/// @b29c606e2): the Windows arm, which suppresses the missing-DLL dialog
/// around `LoadLibraryW`, and the `#else` arm.
fn dlLoadLibrary(path: [*:0]const u8) ?*anyopaque {
    if (comptime is_windows) {
        const wpath = widen(std.mem.sliceTo(path, 0)) orelse return null;
        defer allocator.free(wpath);
        // suppress error dialogs for missing DLLs
        const old_mode = win32.SetErrorMode(win32.SEM_FAILCRITICALERRORS);
        _ = win32.SetErrorMode(old_mode | win32.SEM_FAILCRITICALERRORS);
        const handle = win32.LoadLibraryW(wpath.ptr);
        _ = win32.SetErrorMode(old_mode);
        return @ptrCast(handle);
    }
    // `RTLD_LOCAL` is zero on Linux, so Zig's flag struct has no field for it.
    const flags: std.c.RTLD = if (builtin.os.tag == .linux) .{ .NOW = true } else .{ .NOW = true, .LOCAL = true };
    return std.c.dlopen(path, flags);
}

/// Ports `dl_get_sym`, `dl_get_sym` (ggml-backend-dl.cpp:17, 39 @b29c606e2).
fn dlGetSym(handle: *anyopaque, name: [*:0]const u8) ?*anyopaque {
    if (comptime is_windows) {
        const old_mode = win32.SetErrorMode(win32.SEM_FAILCRITICALERRORS);
        _ = win32.SetErrorMode(old_mode | win32.SEM_FAILCRITICALERRORS);
        const p = win32.GetProcAddress(@ptrCast(handle), name);
        _ = win32.SetErrorMode(old_mode);
        return p;
    }
    return std.c.dlsym(handle, name);
}

/// Ports `dl_error`, `dl_error` (ggml-backend-dl.cpp:28, 43 @b29c606e2).
///
/// Return: the pending dynamic-linker error, or `""` when there is none --
/// always `""` on Windows. Borrowed from libc and valid only until the next
/// `dlopen`/`dlsym`.
fn dlError() [*:0]const u8 {
    if (comptime is_windows) return "";
    return std.c.dlerror() orelse "";
}

/// Ports `dl_handle_deleter`, `dl_handle_deleter` (ggml-backend-dl.h:22, 32
/// @b29c606e2): its call operator, `FreeLibrary` or `dlclose`.
fn dlClose(handle: *anyopaque) void {
    if (comptime is_windows) {
        _ = win32.FreeLibrary(@ptrCast(handle));
        return;
    }
    _ = std.c.dlclose(handle);
}

// -----------------------------------------------------------------------------
// The registry

/// Mirrors `struct ggml_backend_reg_entry` (src/ggml-backend-reg.cpp:110 @b29c606e2).
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

/// `ggml_backend_cuda_reg`, from `ggml-cuda.h`. Declared rather than imported,
/// as `ggml_backend_metal_reg` is.
extern fn ggml_backend_cuda_reg() c.ggml_backend_reg_t;

/// `ggml_backend_vk_reg`, from `ggml-vulkan.h`. Declared rather than imported,
/// as `ggml_backend_metal_reg` is.
extern fn ggml_backend_vk_reg() c.ggml_backend_reg_t;

/// `ggml_backend_blas_reg`, from `ggml-blas.h`. Declared rather than
/// imported, as `ggml_backend_metal_reg` is.
extern fn ggml_backend_blas_reg() c.ggml_backend_reg_t;

const Entry = struct {
    reg: c.ggml_backend_reg_t,
    handle: ?*anyopaque,
};

/// Mirrors `struct ggml_backend_registry` (src/ggml-backend-reg.cpp:115 @b29c606e2).
const Registry = struct {
    backends: std.ArrayList(Entry) = .empty,
    devices: std.ArrayList(c.ggml_backend_dev_t) = .empty,

    /// Ports `ggml_backend_registry`'s constructor
    /// (src/ggml-backend-reg.cpp:119 @b29c606e2).
    ///
    /// Only the `GGML_USE_CUDA`, `GGML_USE_METAL`, `GGML_USE_VULKAN`,
    /// `GGML_USE_BLAS` and `GGML_USE_CPU` arms are compiled here, in the
    /// C++'s order: CUDA first, CPU last.
    ///
    /// Return: nothing. Registration failures are not reportable through the
    /// C++ constructor either.
    fn init(self: *Registry) void {
        if (use_cuda) self.registerBackend(ggml_backend_cuda_reg(), null);
        if (use_metal) self.registerBackend(ggml_backend_metal_reg(), null);
        if (use_vulkan) {
            // The C++'s runtime check of `GGML_DISABLE_VULKAN`
            // (src/ggml-backend-reg.cpp:131 @b29c606e2).
            if (std.c.getenv("GGML_DISABLE_VULKAN") == null) {
                self.registerBackend(ggml_backend_vk_reg(), null);
            } else {
                impl.logDebug("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n", .{});
            }
        }
        if (use_blas) self.registerBackend(ggml_backend_blas_reg(), null);
        self.registerBackend(c.ggml_backend_cpu_reg(), null);
    }

    /// Ports `ggml_backend_registry`'s `register_backend`
    /// (src/ggml-backend-reg.cpp:186 @b29c606e2).
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
    /// (src/ggml-backend-reg.cpp:207 @b29c606e2).
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
    /// (src/ggml-backend-reg.cpp:220 @b29c606e2).
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
            dlClose(handle);
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
    /// (src/ggml-backend-reg.cpp:266 @b29c606e2).
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
/// @b29c606e2).
///
/// C++ magic statics are initialised exactly once under `__cxa_guard_acquire`;
/// this is that guard, written out. The fast path is a single acquire load, so
/// the lock is touched only on the first call — matching the C++, where every
/// call after the first is lock-free.
var registry: Registry = .{};
var registry_ready = std.atomic.Value(bool).init(false);
var registry_mutex: impl.CxxMutex = .{};

/// Ports `get_reg` (src/ggml-backend-reg.cpp:292 @b29c606e2).
///
/// Return: the process-wide registry, constructed on first use. Borrowed for
/// the life of the process.
fn getReg() *Registry {
    if (registry_ready.load(.acquire)) return &registry;

    registry_mutex.lock();
    defer {
        registry_mutex.unlock();
    }

    if (!registry_ready.load(.acquire)) {
        registry.init();
        registry_ready.store(true, .release);
    }
    return &registry;
}

// -----------------------------------------------------------------------------
// Internal API

/// Ports `ggml_backend_register` (src/ggml-backend-reg.cpp:298 @b29c606e2).
export fn ggml_backend_register(reg: c.ggml_backend_reg_t) callconv(.c) void {
    getReg().registerBackend(reg, null);
}

/// Ports `ggml_backend_device_register` (src/ggml-backend-reg.cpp:302 @b29c606e2).
export fn ggml_backend_device_register(device: c.ggml_backend_dev_t) callconv(.c) void {
    getReg().registerDevice(device);
}

// -----------------------------------------------------------------------------
// Backend enumeration

/// Ports `striequals` (src/ggml-backend-reg.cpp:307 @b29c606e2).
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

/// Ports `ggml_backend_reg_count` (src/ggml-backend-reg.cpp:316 @b29c606e2).
///
/// Return: how many backends are registered.
export fn ggml_backend_reg_count() callconv(.c) usize {
    return getReg().backends.items.len;
}

/// Ports `ggml_backend_reg_get` (src/ggml-backend-reg.cpp:320 @b29c606e2).
///
/// Parameters:
/// - `index`: position in registration order; must be in range.
///
/// Return: the backend at `index`. Borrowed; owned by the registry.
export fn ggml_backend_reg_get(index: usize) callconv(.c) c.ggml_backend_reg_t {
    impl.assert(index < ggml_backend_reg_count(), "index < ggml_backend_reg_count()");
    return getReg().backends.items[index].reg;
}

/// Ports `ggml_backend_reg_by_name` (src/ggml-backend-reg.cpp:325 @b29c606e2).
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

/// Ports `ggml_backend_dev_count` (src/ggml-backend-reg.cpp:336 @b29c606e2).
///
/// Return: how many devices every registered backend exposes in total.
export fn ggml_backend_dev_count() callconv(.c) usize {
    return getReg().devices.items.len;
}

/// Ports `ggml_backend_dev_get` (src/ggml-backend-reg.cpp:340 @b29c606e2).
///
/// Parameters:
/// - `index`: position in registration order; must be in range.
///
/// Return: the device at `index`. Borrowed; owned by its backend.
export fn ggml_backend_dev_get(index: usize) callconv(.c) c.ggml_backend_dev_t {
    impl.assert(index < ggml_backend_dev_count(), "index < ggml_backend_dev_count()");
    return getReg().devices.items[index];
}

/// Ports `ggml_backend_dev_by_name` (src/ggml-backend-reg.cpp:345 @b29c606e2).
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

/// Ports `ggml_backend_dev_by_type` (src/ggml-backend-reg.cpp:355 @b29c606e2).
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

/// Ports `ggml_backend_init_by_name` (src/ggml-backend-reg.cpp:366 @b29c606e2).
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

/// Ports `ggml_backend_init_by_type` (src/ggml-backend-reg.cpp:374 @b29c606e2).
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

/// Ports `ggml_backend_init_best` (src/ggml-backend-reg.cpp:382 @b29c606e2).
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

/// Ports `ggml_backend_load` (src/ggml-backend-reg.cpp:393 @b29c606e2).
///
/// Parameters:
/// - `path`: the shared library to load.
///
/// Return: the registered backend, or null on any failure. Borrowed; owned by
/// the registry.
export fn ggml_backend_load(path: [*:0]const u8) callconv(.c) c.ggml_backend_reg_t {
    return getReg().loadBackend(path, false);
}

/// Ports `ggml_backend_unload` (src/ggml-backend-reg.cpp:397 @b29c606e2).
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

/// Ports `get_executable_path` (src/ggml-backend-reg.cpp:401 @b29c606e2), the
/// `__APPLE__` arm.
///
/// Parameters:
/// - `buf`: receives the directory, null-terminated, with a trailing `/`.
///
/// Return: the slice of `buf` written, or null if the path did not fit. The
/// C++ grows a `std::vector` until the call succeeds; the fixed buffer here is
/// `std.Io.Dir.max_path_bytes`, which the call cannot exceed.
fn getExecutablePath(buf: []u8) ?[:0]const u8 {
    if (comptime builtin.os.tag == .linux) return getExecutablePathLinux(buf);
    if (comptime is_windows) return getExecutablePathWindows(buf);
    var size: u32 = @intCast(buf.len);
    if (_NSGetExecutablePath(buf.ptr, &size) != 0) return null;

    // `_NSGetExecutablePath` null-terminates but only updates `size` when it
    // fails, so the length has to come from the string itself.
    const full = std.mem.sliceTo(buf[0..], 0);

    // "remove executable name", keeping the separator: the C++ appends "/" to
    // the truncated string, so a path with no slash at all becomes "/".
    const cut = if (std.mem.findScalarLast(u8, full, '/')) |i| i else 0;
    if (cut + 2 > buf.len) return null;
    buf[cut] = '/';
    buf[cut + 1] = 0;
    return buf[0 .. cut + 1 :0];
}

/// Ports `get_executable_path` (src/ggml-backend-reg.cpp:401 @b29c606e2), the
/// `__linux__` arm.
///
/// Parameters:
/// - `buf`: receives the directory, null-terminated, with a trailing `/`.
///
/// Return: the slice of `buf` written, or null if it did not fit. A failed
/// `readlink` gives `./`, as the C++'s `base_path = "."` does; a path that
/// fills the buffer is the C++'s cue to grow it, and here is a failure, since
/// `buf` is already `max_path_bytes`.
fn getExecutablePathLinux(buf: []u8) ?[:0]const u8 {
    const len = std.c.readlink("/proc/self/exe", buf.ptr, buf.len);
    var cut: usize = 1;
    if (len == -1) {
        buf[0] = '.';
    } else {
        const n: usize = @intCast(len);
        if (n >= buf.len) return null;
        // "remove executable name"; with no slash the whole path stays.
        cut = std.mem.findScalarLast(u8, buf[0..n], '/') orelse n;
    }
    if (cut + 2 > buf.len) return null;
    buf[cut] = '/';
    buf[cut + 1] = 0;
    return buf[0 .. cut + 1 :0];
}

/// Ports `get_executable_path` (src/ggml-backend-reg.cpp:401 @b29c606e2), the
/// `_WIN32` arm.
///
/// Parameters:
/// - `buf`: receives the directory as UTF-8, null-terminated, with a trailing
///   `\`.
///
/// Return: the slice of `buf` written, or null if it did not fit. A failed
/// `GetModuleFileNameW` gives the C++'s empty `fs::path`, which no search
/// finds. The C++ asks for `MAX_PATH` characters once and takes what it gets.
fn getExecutablePathWindows(buf: []u8) ?[:0]const u8 {
    var wpath: [win32.MAX_PATH]u16 = undefined;
    const len = win32.GetModuleFileNameW(null, &wpath, wpath.len);
    if (len == 0) {
        buf[0] = 0;
        return buf[0..0 :0];
    }
    // "remove executable name"; with no backslash the whole path stays.
    const full = wpath[0..len];
    const cut = std.mem.findScalarLast(u16, full, '\\') orelse full.len;
    const n = std.unicode.utf16LeToUtf8(buf, full[0..cut]) catch return null;
    if (n + 2 > buf.len) return null;
    buf[n] = '\\';
    buf[n + 1] = 0;
    return buf[0 .. n + 1 :0];
}

/// `fs::current_path()`, as UTF-8: `getcwd` on POSIX, the wide working
/// directory narrowed on Windows. Null on failure, which the C++ would throw.
fn currentPath(buf: []u8) ?[]const u8 {
    if (comptime is_windows) {
        var wbuf: [std.Io.Dir.max_path_bytes]u16 = undefined;
        const len = win32.GetCurrentDirectoryW(wbuf.len, &wbuf);
        if (len == 0 or len >= wbuf.len) return null;
        const n = std.unicode.utf16LeToUtf8(buf, wbuf[0..len]) catch return null;
        return buf[0..n];
    }
    const p = std.c.getcwd(buf.ptr, buf.len) orelse return null;
    return std.mem.sliceTo(p, 0);
}

/// `fs::path::operator/` for the two joins `ggml_backend_load_best` makes:
/// `entry.path()`, which a directory iterator builds as `search_path /
/// name`, and `search_path / filename`. On Windows libc++ inserts the
/// preferred `\` only when the left side is non-empty and does not already
/// end in a separator. The POSIX arm keeps the plain `/` this port has always
/// joined with; `dlopen` and the log treat the two spellings alike.
fn joinPath(buf: []u8, dir: []const u8, name: []const u8) ?[:0]const u8 {
    if (comptime is_windows) {
        const sep: []const u8 = if (dir.len == 0 or dir[dir.len - 1] == '\\' or dir[dir.len - 1] == '/') "" else "\\";
        return std.mem.printSentinel(buf, "{s}{s}{s}", .{ dir, sep, name }, 0) catch null;
    }
    return std.mem.printSentinel(buf, "{s}/{s}", .{ dir, name }, 0) catch null;
}

/// `fs::exists`. Errors read as "does not exist", which is how the C++'s
/// `ec` branch ends too: a debug line and the next path.
/// The text `std::error_code::message()` gives for the call that just
/// failed: `strerror(errno)`, which libc++'s `generic_category` and POSIX
/// `system_category` both return. On Windows the C++ formats the Win32 error
/// through `FormatMessage`; the CRT's `errno` is what is to hand here, so the
/// text of these two debug lines can differ there.
fn lastErrorMessage() [*:0]const u8 {
    return Libc.strerror(std.c._errno().*);
}

fn pathExists(path: [:0]const u8) bool {
    if (comptime is_windows) {
        const wpath = widen(path) orelse return false;
        defer allocator.free(wpath);
        return win32.GetFileAttributesW(wpath.ptr) != win32.INVALID_FILE_ATTRIBUTES;
    }
    return Libc.access(path.ptr, Libc.F_OK) == 0;
}

/// libc's `readdir`. Zig 0.17's `std.c.readdir` binds `readdir$INODE64` on
/// every x86_64 Darwin target, and only macOS's libSystem defines it: the
/// iOS, tvOS and visionOS simulators are 64-bit-inode only
/// (`__DARWIN_ONLY_64_BIT_INO_T` in their `sys/cdefs.h`) and export plain
/// `readdir`, so the xcframework's x86_64 simulator slices failed to link.
const readdir = if (builtin.cpu.arch == .x86_64 and builtin.os.tag.isDarwin() and builtin.os.tag != .macos)
    struct {
        extern "c" fn readdir(dir: *std.c.DIR) ?*std.c.dirent;
    }.readdir
else
    std.c.readdir;

/// `fs::directory_iterator` over one search path, filtered to the entries
/// `is_regular_file` accepts, yielding each one's file name as UTF-8.
///
/// POSIX reads `readdir`'s `d_type`. Windows reads `FindFirstFileW`'s
/// attributes as libc++'s `get_file_type` does, in its `file_descriptor.h`: a
/// symlink reparse point is a symlink, which `is_regular_file` follows, a
/// directory is a directory, and anything else is a regular file.
const DirScan = struct {
    os: if (is_windows) struct {
        find: win32.HANDLE,
        data: win32.WIN32_FIND_DATAW,
        first: bool,
        dir: [:0]const u8,
    } else *std.c.DIR,
    name_buf: [std.Io.Dir.max_path_bytes]u8 = undefined,

    /// Opens `dir`, or returns null when it cannot be listed -- the C++'s
    /// `fs::exists` failing, which this scan stands in for.
    fn open(dir: [:0]const u8) ?DirScan {
        if (comptime is_windows) {
            if (!pathExists(dir)) return null;
            var pattern_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const pattern = joinPath(&pattern_buf, dir, "*") orelse return null;
            const wpattern = widen(pattern) orelse return null;
            defer allocator.free(wpattern);
            var data: win32.WIN32_FIND_DATAW = undefined;
            const find = win32.FindFirstFileW(wpattern.ptr, &data);
            if (find == std.os.windows.INVALID_HANDLE_VALUE) return null;
            return .{ .os = .{ .find = find, .data = data, .first = true, .dir = dir } };
        }
        return .{ .os = std.c.opendir(dir.ptr) orelse return null };
    }

    fn close(self: *DirScan) void {
        if (comptime is_windows) {
            _ = win32.FindClose(self.os.find);
            return;
        }
        _ = std.c.closedir(self.os);
    }

    /// The next regular file's name, borrowed from `self` until the next call.
    fn next(self: *DirScan) ?[]const u8 {
        if (comptime is_windows) {
            while (true) {
                if (!self.os.first and win32.FindNextFileW(self.os.find, &self.os.data) == 0) return null;
                self.os.first = false;
                const wname = std.mem.sliceTo(&self.os.data.cFileName, 0);
                const n = std.unicode.utf16LeToUtf8(&self.name_buf, wname) catch continue;
                const name = self.name_buf[0..n];
                const attrs = self.os.data.dwFileAttributes;
                if (attrs & win32.FILE_ATTRIBUTE_REPARSE_POINT != 0 and self.os.data.dwReserved0 == win32.IO_REPARSE_TAG_SYMLINK) {
                    // `is_regular_file` follows the link.
                    var full_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
                    const full = joinPath(&full_buf, self.os.dir, name) orelse continue;
                    const wfull = widen(full) orelse continue;
                    defer allocator.free(wfull);
                    const target = win32.GetFileAttributesW(wfull.ptr);
                    if (target == win32.INVALID_FILE_ATTRIBUTES or target & win32.FILE_ATTRIBUTE_DIRECTORY != 0) continue;
                    return name;
                }
                if (attrs & win32.FILE_ATTRIBUTE_DIRECTORY != 0) continue;
                return name;
            }
        }
        while (readdir(self.os)) |ent| {
            // DT_REG. `is_regular_file` in the C++; a `DT_UNKNOWN` filesystem
            // would need a stat, which APFS never reports.
            if (ent.type != 8) continue;
            // `d_namlen` is BSD; `d_name` is NUL-terminated everywhere.
            return std.mem.sliceTo(&ent.name, 0);
        }
        return null;
    }
};

/// Ports `backend_filename_prefix`, `backend_filename_prefix`
/// (src/ggml-backend-reg.cpp:464 @b29c606e2) and `backend_filename_extension`,
/// `backend_filename_extension` (src/ggml-backend-reg.cpp:472 @b29c606e2):
/// both arms of each.
const backend_filename_prefix = if (is_windows) "ggml-" else "libggml-";
const backend_filename_extension = if (is_windows) ".dll" else ".so";

/// Ports `ggml_backend_load_best` (src/ggml-backend-reg.cpp:480 @b29c606e2).
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
    var prefix_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const file_prefix = std.mem.print(
        &prefix_buf,
        backend_filename_prefix ++ "{s}-",
        .{name},
    ) catch return null;

    // The C++ builds `search_paths` as a vector; two entries is the most this
    // configuration ever has, and `GGML_BACKEND_DIR` is not defined here.
    var exe_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    var cwd_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
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
        if (currentPath(&cwd_buf)) |p| {
            search_paths[n_paths] = p;
            n_paths += 1;
        } else {
            impl.printDebug("%s: current_path() failure, error-message: %s\n", .{ "load_best", lastErrorMessage() });
        }
    }

    var best_score: c_int = 0;
    var best_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    var best_len: usize = 0;

    for (search_paths[0..n_paths]) |search_path| {
        var dir_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const dir_path = std.mem.printSentinel(&dir_buf, "{s}", .{search_path}, 0) catch continue;

        // `fs::exists`, then `directory_iterator`, whose own failure is
        // logged apart since v0.4.0.
        if (!pathExists(dir_path)) {
            impl.printDebug("%s: search path %s does not exist\n", .{ "load_best", dir_path.ptr });
            continue;
        }
        var dir = DirScan.open(dir_path) orelse {
            impl.printDebug("%s: failed to enumerate %s: %s\n", .{ "load_best", dir_path.ptr, lastErrorMessage() });
            continue;
        };
        defer dir.close();

        while (dir.next()) |entry_name| {
            if (!std.mem.startsWith(u8, entry_name, file_prefix)) continue;
            if (!std.mem.endsWith(u8, entry_name, backend_filename_extension)) continue;

            var cand_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const cand = joinPath(&cand_buf, search_path, entry_name) orelse continue;

            const handle = dlLoadLibrary(cand.ptr) orelse {
                if (!silent) {
                    impl.logError("%s: failed to load %s: %s\n", .{ "load_best", cand.ptr, dlError() });
                }
                continue;
            };
            // The C++ lets the `unique_ptr` close it at the end of the
            // iteration: the winner is re-opened by `load_backend` below.
            defer dlClose(handle);

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
            var file_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const filename = std.mem.print(
                &file_buf,
                backend_filename_prefix ++ "{s}" ++ backend_filename_extension,
                .{name},
            ) catch continue;
            var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const path = joinPath(&path_buf, search_path, filename) orelse continue;
            if (!pathExists(path)) continue;
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

/// Ports `ggml_backend_load_all` (src/ggml-backend-reg.cpp:574 @b29c606e2).
export fn ggml_backend_load_all() callconv(.c) void {
    ggml_backend_load_all_from_path(null);
}

/// Ports `ggml_backend_load_all_from_path` (src/ggml-backend-reg.cpp:578 @b29c606e2).
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
    const gpu_regs: usize = @as(usize, @intFromBool(use_metal)) + @intFromBool(use_cuda) +
        @intFromBool(use_vulkan and std.c.getenv("GGML_DISABLE_VULKAN") == null) + @intFromBool(use_blas);
    try std.testing.expectEqual(gpu_regs + 1, ggml_backend_reg_count());

    if (use_cuda) {
        try std.testing.expectEqualStrings("CUDA", std.mem.sliceTo(c.ggml_backend_reg_name(ggml_backend_reg_get(0)), 0));
    }

    if (use_metal) {
        // The Metal backend registers itself as `GGML_METAL_NAME`, which is
        // "MTL", not "Metal" (ggml-metal/ggml-metal.cpp:15 @b29c606e2). Asserting the real names is
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
