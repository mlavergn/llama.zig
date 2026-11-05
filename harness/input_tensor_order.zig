//! The order-free comparison of `llama_graph_n_input_tensors`'s log lines.
//!
//! **Not a port.** Since v0.5.0 every graph reserve logs each input tensor
//! and the nodes that read it (llama-context.cpp:676 @620d2225c), iterating
//! an `std::unordered_map` keyed on the tensor's address. The order is the
//! addresses', which differ between the C++ and the port -- and between any
//! two runs of either -- so the block is compared as a multiset: every
//! maximal run of consecutive lines carrying the function's prefix is
//! sorted before two transcripts are compared. Its lines, their count and
//! the line after it still have to match.

const std = @import("std");

/// The prefix every line of the block carries.
pub const prefix = "llama_graph_n_input_tensors: ";

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// `text` with every run of `prefix` lines sorted.
///
/// Parameters:
/// - `gpa`: the allocator for the result.
/// - `text`: a transcript, lines separated by `\n`.
///
/// Return: the sorted transcript, owned by the caller; the same bytes
/// when `text` holds no such line.
pub fn sortRuns(gpa: std.mem.Allocator, text: []const u8) std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    var run: std.ArrayList([]const u8) = .empty;
    defer run.deinit(gpa);
    var lines = std.mem.splitScalar(u8, text, '\n');
    var first = true;
    while (true) {
        const next = lines.next();
        if (next) |line| {
            if (std.mem.startsWith(u8, line, prefix)) {
                run.append(gpa, line) catch @panic("OOM");
                continue;
            }
        }
        if (run.items.len != 0) {
            std.mem.sort([]const u8, run.items, {}, lessThan);
            for (run.items) |r| {
                if (!first) out.append(gpa, '\n') catch @panic("OOM");
                first = false;
                out.appendSlice(gpa, r) catch @panic("OOM");
            }
            run.clearRetainingCapacity();
        }
        const line = next orelse break;
        if (!first) out.append(gpa, '\n') catch @panic("OOM");
        first = false;
        out.appendSlice(gpa, line) catch @panic("OOM");
    }
    return out;
}

test "a run is sorted, the lines around it kept in place" {
    const gpa = std.testing.allocator;
    var r = sortRuns(gpa, "a\n" ++ prefix ++ "z\n" ++ prefix ++ "b\nc\n");
    defer r.deinit(gpa);
    try std.testing.expectEqualStrings("a\n" ++ prefix ++ "b\n" ++ prefix ++ "z\nc\n", r.items);
}
