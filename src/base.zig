const std = @import("std");
const log = std.log.scoped(.llamazig_base);
const mod = @import("module.zig");

/// The template's placeholder feature: it builds greeting strings.
///
/// Stands in for whatever real work a project cloned from this template does.
/// It is deliberately the smallest thing that still exercises the conventions
/// every other struct here follows: the allocator arrives first and is stored
/// as a field, helpers reach for `self.allocator` rather than re-asking, and
/// what that allocator produces is released by the struct that holds it.
pub const Base = struct {
    const Self = @This();

    /// The subject used when the caller does not name one.
    pub const default_subject: []const u8 = "World";

    allocator: std.mem.Allocator,

    /// The most recent message. Owned here, not by the caller: this struct
    /// holds the allocator that made it, so this struct is what can free it.
    message: []u8 = &.{},

    /// Constructs a `Base` bound to `allocator`.
    ///
    /// Parameters:
    /// - `allocator`: allocator for the strings this produces; must outlive the
    ///   value and every string still held from it.
    ///
    /// Return: an initialized `Base`; the error union is reserved for the
    /// lifecycle convention and yields no error today.
    pub fn init(allocator: std.mem.Allocator) !Self {
        return Self{ .allocator = allocator };
    }

    /// Releases the value and the message it holds.
    ///
    /// Parameters:
    /// - `self`: the value to tear down.
    ///
    /// Return: nothing. Any slice handed out by `getMessage` is invalid after
    /// this returns.
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.message);
    }

    /// Builds the message for `subject`.
    ///
    /// An empty subject falls back to `default_subject`, so a caller with no
    /// argument to pass need not special-case the absence itself.
    ///
    /// Replaces any previously held message, so the memory does not grow with
    /// repeated calls.
    ///
    /// Parameters:
    /// - `self`: the value supplying the allocator and holding the result.
    /// - `subject`: who to greet; borrowed only for the duration of the call.
    ///
    /// Return: the message, borrowed from `self` and valid until the next
    /// `getMessage` or `deinit`; propagates allocation failure.
    pub fn getMessage(self: *Self, subject: []const u8) ![]const u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });

        const who = if (subject.len == 0) default_subject else subject;

        // Built before the old one is released, so a failure here leaves the
        // previous message intact rather than dangling.
        const next = try std.fmt.allocPrint(self.allocator, "Hello {s}", .{who});
        self.allocator.free(self.message);
        self.message = next;

        return self.message;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Base);
}

test "base names the subject it is given" {
    const allocator = std.testing.allocator;

    var base = try Base.init(allocator);
    defer base.deinit();

    try std.testing.expectEqualStrings(mod.test_greeting, try base.getMessage(mod.test_subject));
}

test "base falls back to the default subject" {
    const allocator = std.testing.allocator;

    var base = try Base.init(allocator);
    defer base.deinit();

    try std.testing.expectEqualStrings("Hello " ++ Base.default_subject, try base.getMessage(""));
}

test "base replaces the message it holds" {
    const allocator = std.testing.allocator;

    var base = try Base.init(allocator);
    defer base.deinit();

    // Two calls, one deinit: the first message must not leak. std.testing's
    // allocator fails the test if it does.
    try std.testing.expectEqualStrings(mod.test_greeting, try base.getMessage(mod.test_subject));
    try std.testing.expectEqualStrings("Hello " ++ Base.default_subject, try base.getMessage(""));
}
