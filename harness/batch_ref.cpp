// The reference side of `scripts/batch-diff`.
//
// **Not a port.** Drives the real `llama_batch_allocr` in `libllama.a`
// through a C interface the Zig harness can call: construct, `init`, the
// accessors, every split mode and `ubatch_reserve`; and since v0.6.0 the
// `llama_batch_ext` `init` takes -- built with its test constructor, filled
// by `llama_batch_compat::init` or the `llama_batch_ext_*` C API, and read
// back entry by entry, sequence ids in the `unordered_set`'s order. The memory module it is
// handed is a stub `llama_memory_i` whose only live methods are
// `seq_pos_min` and `seq_pos_max`, answered from a table the harness fills;
// the port's `MemoryPos` reads the same table through `ref_mem_min` and
// `ref_mem_max`, so both sides see one memory.
//
// A returned `llama_ubatch` is copied into a heap holder so its `shared_ptr`
// keeps the data alive until `ref_ubatch_free` -- the harness compares the
// ubatches only after the allocator has moved on, which is part of what is
// checked.

#include "llama-batch.h"
#include "llama-ext.h"
#include "llama-memory.h"

#include <cstdint>
#include <cstring>
#include <map>

namespace {

llama_pos g_mem_min[LLAMA_MAX_SEQ];
llama_pos g_mem_max[LLAMA_MAX_SEQ];

// `llama_batch_compat::init` asks the memory about every sequence below the
// batch's `n_seq_max`, which a test may set past `LLAMA_MAX_SEQ`.
llama_pos mem_min(llama_seq_id s) { return s >= 0 && s < LLAMA_MAX_SEQ ? g_mem_min[s] : -1; }
llama_pos mem_max(llama_seq_id s) { return s >= 0 && s < LLAMA_MAX_SEQ ? g_mem_max[s] : -1; }

struct fake_memory : llama_memory_i {
    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override { return nullptr; }
    llama_memory_context_ptr init_full() override { return nullptr; }
    llama_memory_context_ptr init_update(llama_context *, bool) override { return nullptr; }

    bool get_can_shift() const override { return false; }

    void clear(bool) override {}

    bool seq_rm  (llama_seq_id, llama_pos, llama_pos) override { return false; }
    void seq_cp  (llama_seq_id, llama_seq_id, llama_pos, llama_pos) override {}
    void seq_keep(llama_seq_id) override {}
    void seq_add (llama_seq_id, llama_pos, llama_pos, llama_pos) override {}
    void seq_div (llama_seq_id, llama_pos, llama_pos, int) override {}

    llama_pos seq_pos_min(llama_seq_id s) const override { return mem_min(s); }
    llama_pos seq_pos_max(llama_seq_id s) const override { return mem_max(s); }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return {}; }

    void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override {}
    void state_read (llama_io_read_i  &, llama_seq_id, llama_state_seq_flags) override {}
};

fake_memory g_memory;

} // namespace

// The scalar and pointer fields of `llama_ubatch`, in its order, without
// the shared_ptr.
struct ref_ubatch {
    uint32_t b_equal_seqs;
    uint32_t n_tokens;
    uint32_t n_seq_tokens;
    uint32_t n_seqs;
    uint32_t n_seqs_unq;
    uint32_t n_pos;

    llama_token  *  token;
    float        *  embd;
    float        *  embd_state;
    int32_t      *  embd_row;
    llama_pos    *  pos;
    int32_t      *  n_seq_id;
    llama_seq_id ** seq_id;
    llama_seq_id *  seq_id_unq;
    int32_t      *  seq_idx;
    int8_t       *  output;
    int8_t       *  type;
    int32_t      *  decision_order;

    void * holder; // llama_ubatch *, for ref_ubatch_free
};

// One `llama_batch_ext::token`, flattened.
struct ref_ext_token {
    llama_token  id;
    int32_t      has_embd;
    uint64_t     embd_off;
    int32_t      has_state;
    uint64_t     state_off;
    int32_t      embd_row;
    int32_t      output;
    int32_t      decision_order;
    llama_pos    pos[4];
    int32_t      n_seq;
    llama_seq_id seq[LLAMA_MAX_SEQ + 8];
};

static void export_ubatch(const llama_ubatch & u, ref_ubatch * out) {
    auto * h = new llama_ubatch(u);
    out->b_equal_seqs = h->b_equal_seqs;
    out->n_tokens     = h->n_tokens;
    out->n_seq_tokens = h->n_seq_tokens;
    out->n_seqs       = h->n_seqs;
    out->n_seqs_unq   = h->n_seqs_unq;
    out->n_pos        = h->n_pos;
    out->token        = h->token;
    out->embd         = h->embd;
    out->embd_state   = h->embd_state;
    out->embd_row     = h->embd_row;
    out->pos          = h->pos;
    out->n_seq_id     = h->n_seq_id;
    out->seq_id       = h->seq_id;
    out->seq_id_unq   = h->seq_id_unq;
    out->seq_idx      = h->seq_idx;
    out->output       = h->output;
    out->type         = h->type;
    out->decision_order = h->decision_order;
    out->holder       = h;
}

extern "C" {

void ref_mem_set(int s, llama_pos p_min, llama_pos p_max) {
    g_mem_min[s] = p_min;
    g_mem_max[s] = p_max;
}

llama_pos ref_mem_min(int s) { return mem_min(s); }
llama_pos ref_mem_max(int s) { return mem_max(s); }

void * ref_new(uint32_t n_pos_per_embd, int allow_mixed) {
    return new llama_batch_allocr(n_pos_per_embd, allow_mixed != 0);
}

llama_batch_ext * ref_ext_new(size_t n_tokens_max, size_t n_embd_inp, size_t n_embd_inp_enc, llama_seq_id n_seq_max, int use_memory, llama_token n_vocab, size_t n_pos_per_embd) {
    return new llama_batch_ext(n_tokens_max, n_embd_inp, n_embd_inp_enc, n_seq_max, use_memory ? &g_memory : nullptr, n_vocab, n_pos_per_embd);
}

void ref_ext_free(llama_batch_ext * b) { delete b; }

void ref_compat_init(llama_batch_ext * b, const llama_batch * batch, size_t n_embd_row) {
    llama_batch_compat::init(*b, *batch, n_embd_row);
}

int ref_ext_set_decision_order(llama_batch_ext * b, int32_t idx, int32_t order) {
    return llama_batch_ext_set_decision_order(b, idx, (llama_decision_order) order) ? 1 : 0;
}

// v0.9.0: llama-ext.h declares it with C++ linkage.
int32_t ref_ext_add_embd_row(llama_batch_ext * b, llama_seq_id seq_id, int32_t row) {
    return llama_batch_ext_add_embd_row(b, seq_id, row);
}

size_t ref_ext_n_tokens(const llama_batch_ext * b) { return b->tokens.size(); }

// v0.7.0's state width: the test constructor sets none, the context
// constructor one per context type (llama_batch_ext_select_n_embd_state), so
// the harness sets it as a context would.
void ref_ext_set_n_embd_state(llama_batch_ext * b, size_t n) { const_cast<size_t &>(b->n_embd_state) = n; }
size_t ref_ext_n_embd_state(const llama_batch_ext * b) { return b->n_embd_state; }

size_t ref_ext_state(const llama_batch_ext * b, const float ** data) {
    *data = b->state.data();
    return b->state.size();
}
size_t ref_ext_n_embd(const llama_batch_ext * b) { return b->n_embd; }

size_t ref_ext_embd(const llama_batch_ext * b, const float ** data) {
    *data = b->embd.data();
    return b->embd.size();
}

void ref_ext_token(const llama_batch_ext * b, size_t i, ref_ext_token * out) {
    const auto & t = b->tokens[i];
    out->id = t.id;
    out->has_embd = t.has_embd;
    out->embd_off = t.embd_off;
    out->has_state = t.has_state;
    out->state_off = t.state_off;
    out->embd_row = t.embd_row;
    out->output = t.output;
    out->decision_order = t.decision_order;
    for (int j = 0; j < 4; ++j) out->pos[j] = t.pos[j];
    out->n_seq = 0;
    for (auto s : t.seq_ids) {
        if (out->n_seq < (int32_t) (sizeof(out->seq) / sizeof(out->seq[0]))) out->seq[out->n_seq] = s;
        out->n_seq++;
    }
}

void ref_delete(void * a) {
    delete (llama_batch_allocr *) a;
}

int ref_init(void * a, const llama_batch_ext * batch, const llama_vocab * vocab, int output_all) {
    auto * ba = (llama_batch_allocr *) a;
    return ba->init(*batch, *vocab, output_all != 0) ? 1 : 0;
}

void ref_get_batch(void * a, llama_batch * out) {
    *out = ((llama_batch_allocr *) a)->get_batch();
}

uint32_t ref_n_tokens (void * a) { return ((llama_batch_allocr *) a)->get_n_tokens(); }
uint32_t ref_n_outputs(void * a) { return ((llama_batch_allocr *) a)->get_n_outputs(); }
uint32_t ref_n_used   (void * a) { return ((llama_batch_allocr *) a)->get_n_used(); }

size_t ref_out_ids(void * a, const int32_t ** ids) {
    auto & v = ((llama_batch_allocr *) a)->get_out_ids();
    *ids = v.data();
    return v.size();
}

llama_pos ref_seq_pos_min(void * a, llama_seq_id s) { return ((llama_batch_allocr *) a)->seq_pos_min(s); }
llama_pos ref_seq_pos_max(void * a, llama_seq_id s) { return ((llama_batch_allocr *) a)->seq_pos_max(s); }

void ref_split_reset(void * a) {
    ((llama_batch_allocr *) a)->split_reset();
}

void ref_split_simple(void * a, uint32_t n_ubatch, ref_ubatch * out) {
    export_ubatch(((llama_batch_allocr *) a)->split_simple(n_ubatch), out);
}

void ref_split_equal(void * a, uint32_t n_ubatch, int sequential, uint32_t n_keep_tail, ref_ubatch * out) {
    export_ubatch(((llama_batch_allocr *) a)->split_equal(n_ubatch, sequential != 0, n_keep_tail), out);
}

void ref_split_seq(void * a, uint32_t n_ubatch, ref_ubatch * out) {
    export_ubatch(((llama_batch_allocr *) a)->split_seq(n_ubatch), out);
}

void ref_ubatch_reserve(void * a, uint32_t n_seq_tokens, uint32_t n_seqs, ref_ubatch * out) {
    export_ubatch(((llama_batch_allocr *) a)->ubatch_reserve(n_seq_tokens, n_seqs), out);
}

void ref_ubatch_free(ref_ubatch * u) {
    delete (llama_ubatch *) u->holder;
    u->holder = nullptr;
}

// A ubatch the holder keeps alive: true when its data is the shared_ptr's.
int ref_ubatch_owned(const ref_ubatch * u) {
    return ((llama_ubatch *) u->holder)->data != nullptr;
}

// The original batch index of each token, `data->batch_idxs`.
size_t ref_ubatch_batch_idxs(const ref_ubatch * u, const int32_t ** idxs) {
    const auto & d = ((llama_ubatch *) u->holder)->data;
    if (!d) {
        *idxs = nullptr;
        return 0;
    }
    *idxs = d->batch_idxs.data();
    return d->batch_idxs.size();
}

}
