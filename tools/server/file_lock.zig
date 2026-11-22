//! A cross-process exclusive lock: the one two downloads of the same file,
//! here and in huggingface_hub, take before touching the cache.
//!
//! # Provenance
//!
//! Ports `llama.cpp/common/file-lock.cpp` and `llama.cpp/common/file-lock.h`
//! (v1.0.0, `b1b770d90`), their POSIX arm: `llama-server` is not a Windows
//! target. Each declaration names what it ports and the line it began at.

const std = @import("std");
const stdlib = @import("stdlib");
const tc = @import("tools_common");

const log = tc.log;
const Allocator = std.mem.Allocator;

const LOCK = std.posix.LOCK;

/// `open(path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0664)` on
/// `std.Io`, which has no create without following a final link: an open
/// that does not follow one, and when there is nothing to open, an
/// exclusive create, which never follows one either; a file created
/// between the two is opened on the next turn.
fn openNoFollow(io: std.Io, path: []const u8) !std.Io.File {
    const cwd = std.Io.Dir.cwd();
    while (true) {
        if (cwd.openFile(io, path, .{ .mode = .read_write, .follow_symlinks = false })) |f| {
            return f;
        } else |e| if (e != error.FileNotFound) return e;
        if (cwd.createFile(io, path, .{ .read = true, .truncate = false, .exclusive = true, .permissions = .fromMode(0o664) })) |f| {
            return f;
        } else |e| if (e != error.PathAlreadyExists) return e;
    }
}

/// `const std::function<bool()> & keep_waiting`.
pub const KeepWaiting = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque) bool,

    pub fn call(self: KeepWaiting) bool {
        return self.func(self.ctx);
    }
};

/// Ports `common_file_lock` (file-lock.h:8 @b1b770d90): `flock` on a file
/// opened lazily, so a read-only cache works if the lock is never acquired.
pub const FileLock = struct {
    a: Allocator,
    lock_path: []const u8,
    file: ?std.Io.File = null,
    locked: bool = false,

    /// `status` (file-lock.h:24 @b1b770d90).
    const Status = enum {
        acquired,
        busy, // another process holds the lock
        err, // file locking is unavailable: no exclusion is possible
    };

    /// Ports `common_file_lock` (file-lock.cpp:26 @b1b770d90), the
    /// constructor.
    ///
    /// Parameters:
    /// - `lock_path`: borrowed; an empty path locks nothing.
    pub fn init(a: Allocator, lock_path: []const u8) FileLock {
        return .{ .a = a, .lock_path = lock_path };
    }

    /// Ports the destructor of `common_file_lock` (file-lock.cpp:177
    /// @b1b770d90).
    pub fn deinit(self: *FileLock) void {
        self.close();
    }

    /// Ports `try_lock` (file-lock.cpp:68 @b1b770d90), a member of
    /// `common_file_lock`.
    fn tryLock(self: *FileLock) Allocator.Error!Status {
        const func = "try_lock";
        if (self.lock_path.len == 0) {
            return .acquired; // no lock path, nothing to lock
        }
        const io = stdlib.file.defaultIo();
        const f = self.file orelse f: {
            // group-writable, so users sharing a cache can wait on the lock
            const f = openNoFollow(io, self.lock_path) catch {
                const z = try self.a.dupeSentinel(u8, self.lock_path, 0);
                defer self.a.free(z);
                log.err("%s: cannot open lock file: %s\n", .{ func, z.ptr });
                return .err;
            };
            // open() filters the mode through the umask
            f.setPermissions(io, .fromMode(0o664)) catch |err| switch (err) {
                else => {}, // fchmod's result unchecked
            };
            self.file = f;
            break :f f;
        };

        // the system's flock, not std's tryLock, which folds EACCES and the
        // errno printed below into errors of its own
        const e = std.posix.errno(std.posix.system.flock(f.handle, LOCK.EX | LOCK.NB));
        if (e == .SUCCESS) {
            self.locked = true;
            return .acquired;
        }
        // EACCES is reported as contention on some filesystems
        if (e == .AGAIN or e == .ACCES) {
            return .busy;
        }
        // flock is not implemented (some NFS/FUSE mounts): report it instead of racing
        log.err("%s: file locking failed (%s)\n", .{ func, stdlib.errno.strerror(@backingInt(e)) });
        return .err;
    }

    /// Ports `acquire` (file-lock.cpp:124 @b1b770d90), a member of
    /// `common_file_lock`: takes the lock, waiting while another process
    /// holds it; `keep_waiting` is polled before each retry, false cancels
    /// the wait.
    ///
    /// Return: false when cancelled, or when locking is unavailable.
    pub fn acquire(self: *FileLock, keep_waiting: ?KeepWaiting) Allocator.Error!bool {
        // the lock file is opened by try_lock, its parent must exist by then
        if (self.lock_path.len != 0) {
            const lock_dir = stdlib.path.parent(self.lock_path);
            if (lock_dir.len != 0) {
                const ec = stdlib.fs.createDirectories(lock_dir);
                if (ec != 0) {
                    const z = try self.a.dupeSentinel(u8, lock_dir, 0);
                    defer self.a.free(z);
                    log.err("%s: unable to create lock directory '%s': %s\n", .{ "acquire", z.ptr, stdlib.errno.strerror(ec) });
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
                    stdlib.clock.sleepMs(100);
                },
                .err => return false,
            }
        }
    }

    /// Ports `close` (file-lock.cpp:156 @b1b770d90), a member of
    /// `common_file_lock`. The lock file is never deleted: another process
    /// could then create a fresh file at the same path and lock that.
    pub fn close(self: *FileLock) void {
        if (self.file) |f| {
            if (self.locked) {
                _ = std.posix.system.flock(f.handle, LOCK.UN);
            }
            f.close(stdlib.file.defaultIo());
        }
        self.file = null;
        self.locked = false;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
}
