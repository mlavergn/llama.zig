//! A cross-process exclusive lock: the one two downloads of the same file,
//! here and in huggingface_hub, take before touching the cache.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/file-lock.cpp` and `llama.cpp/common/file-lock.h`
//! (v0.8.0, `8642541b2`), their POSIX arm: `llama-server` is not a Windows
//! target. Each declaration names what it ports and the line it began at.

const std = @import("std");
const tc = @import("tools_common");
const hf_cache = @import("hf_cache.zig");

const log = tc.log;
const Allocator = std.mem.Allocator;

extern "c" fn flock(fd: c_int, op: c_int) c_int;
extern "c" fn fchmod(fd: c_int, mode: std.c.mode_t) c_int;
extern "c" fn strerror(e: c_int) [*:0]const u8;

const LOCK_EX: c_int = 2;
const LOCK_NB: c_int = 4;
const LOCK_UN: c_int = 8;

/// `const std::function<bool()> & keep_waiting`.
pub const KeepWaiting = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque) bool,

    pub fn call(self: KeepWaiting) bool {
        return self.func(self.ctx);
    }
};

/// Ports `common_file_lock` (file-lock.h:8 @8642541b2): `flock` on a file
/// opened lazily, so a read-only cache works if the lock is never acquired.
pub const FileLock = struct {
    a: Allocator,
    lock_path: []const u8,
    fd: c_int = -1,
    locked: bool = false,

    /// `status` (file-lock.h:24 @8642541b2).
    const Status = enum {
        acquired,
        busy, // another process holds the lock
        err, // file locking is unavailable: no exclusion is possible
    };

    /// Ports `common_file_lock` (file-lock.cpp:26 @8642541b2), the
    /// constructor.
    ///
    /// Parameters:
    /// - `lock_path`: borrowed; an empty path locks nothing.
    pub fn init(a: Allocator, lock_path: []const u8) FileLock {
        return .{ .a = a, .lock_path = lock_path };
    }

    /// Ports the destructor of `common_file_lock` (file-lock.cpp:177
    /// @8642541b2).
    pub fn deinit(self: *FileLock) void {
        self.close();
    }

    /// Ports `try_lock` (file-lock.cpp:68 @8642541b2), a member of
    /// `common_file_lock`.
    fn tryLock(self: *FileLock) Allocator.Error!Status {
        const func = "try_lock";
        if (self.lock_path.len == 0) {
            return .acquired; // no lock path, nothing to lock
        }
        if (self.fd < 0) {
            const z = try self.a.dupeSentinel(u8, self.lock_path, 0);
            defer self.a.free(z);
            // group-writable, so users sharing a cache can wait on the lock
            self.fd = std.c.open(z.ptr, .{ .ACCMODE = .RDWR, .CREAT = true, .CLOEXEC = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o664));
            if (self.fd < 0) {
                log.err("%s: cannot open lock file: %s\n", .{ func, z.ptr });
                return .err;
            }
            // open() filters the mode through the umask
            _ = fchmod(self.fd, 0o664);
        }

        if (flock(self.fd, LOCK_EX | LOCK_NB) == 0) {
            self.locked = true;
            return .acquired;
        }
        // EACCES is reported as contention on some filesystems
        const e = std.c._errno().*;
        if (e == @backingInt(std.c.E.AGAIN) or e == @backingInt(std.c.E.ACCES)) {
            return .busy;
        }
        // flock is not implemented (some NFS/FUSE mounts): report it instead of racing
        log.err("%s: file locking failed (%s)\n", .{ func, strerror(e) });
        return .err;
    }

    /// Ports `acquire` (file-lock.cpp:124 @8642541b2), a member of
    /// `common_file_lock`: takes the lock, waiting while another process
    /// holds it; `keep_waiting` is polled before each retry, false cancels
    /// the wait.
    ///
    /// Return: false when cancelled, or when locking is unavailable.
    pub fn acquire(self: *FileLock, keep_waiting: ?KeepWaiting) Allocator.Error!bool {
        // the lock file is opened by try_lock, its parent must exist by then
        if (self.lock_path.len != 0) {
            const lock_dir = hf_cache.pathParent(self.lock_path);
            if (lock_dir.len != 0) {
                const ec = hf_cache.createDirectories(self.a, lock_dir);
                if (ec != 0) {
                    const z = try self.a.dupeSentinel(u8, lock_dir, 0);
                    defer self.a.free(z);
                    log.err("%s: unable to create lock directory '%s': %s\n", .{ "acquire", z.ptr, hf_cache.errMessage(ec) });
                    return false;
                }
            }
        }

        while (true) {
            switch (try self.tryLock()) {
                .acquired => return true,
                .busy => {
                    if (keep_waiting) |kw| if (!kw.call()) {
                        return false;
                    };
                    var ts: std.c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
                    while (std.c.nanosleep(&ts, &ts) != 0) {}
                },
                .err => return false,
            }
        }
    }

    /// Ports `close` (file-lock.cpp:156 @8642541b2), a member of
    /// `common_file_lock`. The lock file is never deleted: another process
    /// could then create a fresh file at the same path and lock that.
    pub fn close(self: *FileLock) void {
        if (self.fd >= 0) {
            if (self.locked) {
                _ = flock(self.fd, LOCK_UN);
            }
            _ = std.c.close(self.fd);
        }
        self.fd = -1;
        self.locked = false;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
