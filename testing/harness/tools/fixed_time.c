// Pins the wall clock for a tool gate: cvector-generator's PCA seeds its
// random engine with std::time(0), so without this the two sides start from
// different vectors whenever a second boundary falls between them.
// Preloaded into both sides by testing/scripts/tools-diff's `!preload`.
//
// Upstream reads time(3); the port reads std.Io's real clock, which is
// clock_gettime(CLOCK_REALTIME) under the C library on Darwin and Linux. Both
// are pinned to the same second; every other clock is passed through, so
// durations still advance.
#define _GNU_SOURCE
#include <time.h>

#define FIXED_TIME ((time_t) 1700000000)

static time_t fixed_time(time_t * t) {
    if (t) *t = FIXED_TIME;
    return FIXED_TIME;
}

#if defined(__APPLE__)
static int fixed_clock_gettime(clockid_t clk, struct timespec * ts) {
    if (clk == CLOCK_REALTIME) {
        ts->tv_sec = FIXED_TIME;
        ts->tv_nsec = 0;
        return 0;
    }
    // inside the interposing image, this is the C library's own
    return clock_gettime(clk, ts);
}

__attribute__((used)) static const struct { const void * replacement; const void * replacee; } interposers[]
    __attribute__((section("__DATA,__interpose"))) = {
    { (const void *) fixed_time, (const void *) time },
    { (const void *) fixed_clock_gettime, (const void *) clock_gettime },
};
#else
#include <dlfcn.h>

time_t time(time_t * t) {
    return fixed_time(t);
}

int clock_gettime(clockid_t clk, struct timespec * ts) {
    if (clk == CLOCK_REALTIME) {
        ts->tv_sec = FIXED_TIME;
        ts->tv_nsec = 0;
        return 0;
    }
    static int (*next)(clockid_t, struct timespec *);
    if (!next) next = (int (*)(clockid_t, struct timespec *)) dlsym(RTLD_NEXT, "clock_gettime");
    return next(clk, ts);
}
#endif
