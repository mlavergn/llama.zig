// Pins time(3) for a tool gate: cvector-generator's PCA seeds its random
// engine with std::time(0), so without this the two sides start from
// different vectors whenever a second boundary falls between them.
// Preloaded into both sides by scripts/tools-diff's `!preload`.
#include <time.h>

#define FIXED_TIME ((time_t) 1700000000)

#if defined(__APPLE__)
static time_t fixed_time(time_t * t) {
    if (t) *t = FIXED_TIME;
    return FIXED_TIME;
}

__attribute__((used)) static const struct { const void * replacement; const void * replacee; } interposers[]
    __attribute__((section("__DATA,__interpose"))) = { { (const void *) fixed_time, (const void *) time } };
#else
time_t time(time_t * t) {
    if (t) *t = FIXED_TIME;
    return FIXED_TIME;
}
#endif
