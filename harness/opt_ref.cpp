// The libc++ half of `make opt-diff`'s shuffle checks.
//
// **Not a port.** This is our own code: two C entry points over the real
// `std::uniform_int_distribution<ptrdiff_t>` and `std::shuffle` templates,
// instantiated from the same libc++ headers `ggml-opt.cpp` is compiled
// against, so `harness/opt_diff.zig` can compare `src/ggml/opt.zig`'s
// reproduction of them over ranges no dataset reaches -- up to 2^63, where
// the bits engine needs two engine draws per value.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>

extern "C" {

// Draws `count` values in [a, b] from one engine seeded with `seed`,
// writing each to `out`, then returns the engine's next raw output so the
// caller can check the draws consumed the same number of words.
uint32_t opt_ref_uniform(uint32_t seed, int64_t a, int64_t b, int64_t * out, size_t count) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<ptrdiff_t> dist;
    using param = std::uniform_int_distribution<ptrdiff_t>::param_type;
    for (size_t i = 0; i < count; ++i) {
        out[i] = dist(rng, param(a, b));
    }
    return rng();
}

// Shuffles `v` `reps` times in place with one engine seeded with `seed`;
// returns the engine's next raw output, as above.
uint32_t opt_ref_shuffle(uint32_t seed, int64_t * v, size_t n, size_t reps) {
    std::mt19937 rng(seed);
    for (size_t r = 0; r < reps; ++r) {
        std::shuffle(v, v + n, rng);
    }
    return rng();
}

}
