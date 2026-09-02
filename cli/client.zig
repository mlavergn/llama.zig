const std = @import("std");
const log = std.log.scoped(.llamazig_client);
const mod = @import("llamazig");

/// The one thing the demo CLI does: greet, and say so on stdout.
///
/// Everything the executable does lives here rather than in `main`, so the
/// whole behavior stays reachable from the unit tests below.
pub const Client = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    io: std.Io,

    /// Constructs a client bound to the process capabilities.
    ///
    /// Parameters:
    /// - `allocator`: allocator for the greeting the client renders.
    /// - `io`: IO interface, held for the operations a real client would need.
    ///
    /// Return: an initialized `Client`; the error union is reserved for the
    /// lifecycle convention and yields no error today.
    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Self {
        return Self{ .allocator = allocator, .io = io };
    }

    /// Releases the client.
    ///
    /// Parameters:
    /// - `self`: the client to tear down.
    ///
    /// Return: nothing. The message belongs to the `mod.Base` that made it and
    /// is released with it.
    pub fn deinit(self: *Self) void {
        _ = self;
    }

    /// Greets the subject named on the command line and writes the result.
    ///
    /// Parameters:
    /// - `self`: the client supplying the allocator.
    /// - `writer`: destination for the greeting; flushed before returning.
    /// - `args`: the full argument vector, program name included.
    ///
    /// Return: nothing on success; propagates allocation and write failures.
    pub fn run(self: *Self, writer: *std.Io.Writer, args: []const [:0]const u8) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });

        var base = try mod.Base.init(self.allocator);
        defer base.deinit();

        const subject: []const u8 = if (args.len > 1) args[1] else "";

        const message = try base.getMessage(subject);

        try writer.print("{s}\n", .{message});
        try writer.flush();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Client);
}

test "client greets the subject from the argument vector" {
    const allocator = std.testing.allocator;

    var client = try Client.init(allocator, std.testing.io);
    defer client.deinit();

    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();

    try client.run(&output.writer, &.{ "llamazig", "Zig" });
    try std.testing.expectEqualStrings(mod.test_greeting ++ "\n", output.written());
}

test "client greets the world when given no subject" {
    const allocator = std.testing.allocator;

    var client = try Client.init(allocator, std.testing.io);
    defer client.deinit();

    var output: std.Io.Writer.Allocating = .init(allocator);
    defer output.deinit();

    try client.run(&output.writer, &.{"llamazig"});
    try std.testing.expectEqualStrings("Hello " ++ mod.Base.default_subject ++ "\n", output.written());
}
