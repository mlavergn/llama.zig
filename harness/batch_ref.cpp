// The reference side of `scripts/batch-diff`.
//
// **Not a port.** Drives the real `llama_batch_allocr` in `libllama.a`
// through a C interface the Zig harness can call: construct, `init`, the
// accessors, every split mode and `ubatch_reserve`. The memory module it is
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
#include "llama-memory.h"

#include <cstdint>
#include <cstring>
#include <map>

namespace {

llama_pos g_mem_min[LLAMA_MAX_SEQ];
llama_pos g_mem_max[LLAMA_MAX_SEQ];

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

    llama_pos seq_pos_min(llama_seq_id s) const override { return g_mem_min[s]; }
    llama_pos seq_pos_max(llama_seq_id s) const override { return g_mem_max[s]; }

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
    llama_pos    *  pos;
    int32_t      *  n_seq_id;
    llama_seq_id ** seq_id;
    llama_seq_id *  seq_id_unq;
    int32_t      *  seq_idx;
    int8_t       *  output;

    void * holder; // llama_ubatch *, for ref_ubatch_free
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
    out->pos          = h->pos;
    out->n_seq_id     = h->n_seq_id;
    out->seq_id       = h->seq_id;
    out->seq_id_unq   = h->seq_id_unq;
    out->seq_idx      = h->seq_idx;
    out->output       = h->output;
    out->holder       = h;
}

extern "C" {

void ref_mem_set(int s, llama_pos p_min, llama_pos p_max) {
    g_mem_min[s] = p_min;
    g_mem_max[s] = p_max;
}

llama_pos ref_mem_min(int s) { return g_mem_min[s]; }
llama_pos ref_mem_max(int s) { return g_mem_max[s]; }

void * ref_new(uint32_t n_pos_per_embd) {
    return new llama_batch_allocr(n_pos_per_embd);
}

void ref_delete(void * a) {
    delete (llama_batch_allocr *) a;
}

int ref_init(void * a, const llama_batch * batch, const llama_vocab * vocab, int use_memory, uint32_t n_embd, uint32_t n_seq_max, int output_all) {
    auto * ba = (llama_batch_allocr *) a;
    return ba->init(*batch, *vocab, use_memory ? &g_memory : nullptr, n_embd, n_seq_max, output_all != 0) ? 1 : 0;
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

}
