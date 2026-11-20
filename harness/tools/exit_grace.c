// Gives a tool's log worker time to write what is queued before the process
// exits, for a tool gate. Preloaded into both sides by scripts/tools-diff's
// `!preload`.
//
// Upstream's common_log_main() leaks its logger on purpose (a destructor at
// DLL teardown hangs on Windows; llama.cpp issue 22142), so the messages its
// worker thread has not yet written when a tool returns from main are lost:
// a tool that logs an error and exits at once prints all of them, some or
// none. The port logs synchronously and always prints them. The pause runs
// in an atexit handler, while the worker is still alive, and it writes and
// flushes each queued line in that time.
#include <stdlib.h>
#include <time.h>

static void grace(void) {
    struct timespec ts = { 0, 500 * 1000 * 1000 };
    nanosleep(&ts, NULL);
}

__attribute__((constructor)) static void register_grace(void) {
    atexit(grace);
}
