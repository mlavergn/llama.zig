// The libc++ templates `src/llama/libcxx.zig` reproduces, instantiated with
// the comparators and types the sampler uses, behind a C ABI.
//
// Compiled by scripts/libcxx-diff against the SDK's `c++/v1` headers -- the
// same headers `build/llamacpp.zig` compiles `llama-sampler.cpp` against -- and
// linked into `harness/libcxx_diff.zig`, which calls these and the Zig side on
// the same inputs and compares the results exactly. This file is the
// reference, so it must stay a direct call of the standard template and do
// nothing else.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

struct token_data {
    int32_t id;
    float logit;
    float p;
};

extern "C" {

// llama-sampler.cpp:136 @b29c606e2 -- the comparator every candidate sort uses.
void ref_sort_td(token_data * d, size_t n) {
    std::sort(d, d + n, [](const token_data & a, const token_data & b) { return a.logit > b.logit; });
}

void ref_partial_sort_td(token_data * d, size_t n, size_t k) {
    std::partial_sort(d, d + k, d + n, [](const token_data & a, const token_data & b) { return a.logit > b.logit; });
}

// llama-sampler.cpp:1938 @b29c606e2 -- typical's index sort.
void ref_sort_idx(size_t * idx, size_t n, const float * scores) {
    std::sort(idx, idx + n, [&](size_t a, size_t b) { return scores[a] < scores[b]; });
}

// The same comparator, through std::partial_sort: `size_t` is arithmetic,
// so libc++'s make_heap takes its `__assume_both_children` path here, and
// not for `token_data`.
void ref_partial_sort_idx(size_t * idx, size_t n, size_t k, const float * scores) {
    std::partial_sort(idx, idx + k, idx + n, [&](size_t a, size_t b) { return scores[a] < scores[b]; });
}

void ref_mt(uint32_t seed, uint32_t * out, size_t n) {
    std::mt19937 g(seed);
    for (size_t i = 0; i < n; i++) out[i] = g();
}

void ref_uniform_f64(uint32_t seed, double * out, size_t n) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; i++) out[i] = dist(g);
}

void ref_uniform_f32(uint32_t seed, float * out, size_t n) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; i++) out[i] = dist(g);
}

// llama-sampler.cpp:217 @b29c606e2 -- a fresh distribution per draw, over
// the `p` fields, as llama_sample_dist builds it.
void ref_discrete(const token_data * d, size_t n, uint32_t seed, int draws, int * out) {
    struct probs_iterator {
        typedef std::input_iterator_tag iterator_category;
        typedef float value_type;
        typedef float * pointer;
        typedef float & reference;
        typedef ptrdiff_t difference_type;
        const token_data * data;
        bool operator==(const probs_iterator & o) const { return data == o.data; }
        bool operator!=(const probs_iterator & o) const { return data != o.data; }
        const float & operator*() const { return data->p; }
        probs_iterator & operator++() { ++data; return *this; }
        probs_iterator operator++(int) { probs_iterator t = *this; ++data; return t; }
    };
    std::mt19937 g(seed);
    for (int i = 0; i < draws; i++) {
        std::discrete_distribution<int> dist(probs_iterator{d}, probs_iterator{d + n});
        out[i] = dist(g);
    }
}


// std::hash<std::string>, which llama-vocab.cpp's token_to_id hashes with.
size_t ref_hash_string(const char * s, size_t n) {
    return std::hash<std::string>()(std::string(s, n));
}

// The dylib's __next_prime, which sizes every unordered_map.
size_t ref_next_prime(size_t n) {
    return std::__next_prime(n);
}

// An unordered_map<string, int> built as llama-vocab.cpp builds token_to_id,
// `m[key] = i` in order, then walked: `out` gets the values in iteration
// order. `n` strings, the i-th at blob[offs[i], offs[i + 1]).
// Return: the map's size; `*bc` gets its bucket count.
size_t ref_umap(const char * blob, const size_t * offs, size_t n, int * out, size_t * bc) {
    std::unordered_map<std::string, int> m;
    for (size_t i = 0; i < n; i++) {
        m[std::string(blob + offs[i], offs[i + 1] - offs[i])] = (int) i;
    }
    size_t k = 0;
    for (const auto & it : m) out[k++] = it.second;
    *bc = m.bucket_count();
    return m.size();
}

struct pq_item {
    float score;
    int   left;
};

// llama-vocab.cpp:99 @b29c606e2 -- llm_bigram_spm::comparator.
struct pq_less {
    bool operator()(const pq_item & l, const pq_item & r) const {
        return (l.score < r.score) || (l.score == r.score && l.left > r.left);
    }
};

// A priority_queue driven by `ops`: op >= 0 pushes items[op], -1 pops into
// `out`. Return: the number of pops written.
size_t ref_pq(const pq_item * items, const int * ops, size_t n_ops, pq_item * out) {
    std::priority_queue<pq_item, std::vector<pq_item>, pq_less> q;
    size_t k = 0;
    for (size_t i = 0; i < n_ops; i++) {
        if (ops[i] >= 0) {
            q.push(items[ops[i]]);
        } else if (!q.empty()) {
            out[k++] = q.top();
            q.pop();
        }
    }
    while (!q.empty()) {
        out[k++] = q.top();
        q.pop();
    }
    return k;
}

}
