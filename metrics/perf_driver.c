// Performance driver: one instrumented generation, one line of metrics out.
//
// Not a port. This is measurement apparatus, linked twice by metrics/Makefile:
// once against llama.cpp.zmake's libraries (the stock C, built by the Zig
// toolchain at --release=fast) and once against ours. The driver is identical
// on both sides, so the libraries are the only variable -- the same pairing
// scripts/parity-port uses for correctness.
//
// Usage: perf_driver <model.gguf> <prompt> <n_predict> <label> [flags]
//
// `flags` is a comma-separated list:
//   raw      feed the prompt as-is instead of through the model's chat template
//   show     write the generated text to stderr
//   cpu      the CPU device alone (as scripts/parity-port --cpu)
//
// Environment:
//   PERF_IDLE_MS  (default 2000) the idle window measured before loading, which
//                 every power figure is reported against
//   PERF_TAIL_MS  (default 1500) how long system power keeps integrating after
//                 the last token; see "System power" below
//
// The work is fixed, not the text: decoding is greedy and continues past
// end-of-generation, so every run decodes exactly n_predict tokens. A rate over
// a run that stopped early would not be comparable to one that did not.
// `tok_hash` hashes the token ids so the report can confirm both sides did the
// same work.
//
// What each source measures, and its scope:
//   wall clock      CLOCK_UPTIME_RAW, in-process, at phase boundaries
//   CPU time        getrusage, this process, user + system
//   GPU time        IORegistry AppUsage.accumulatedGPUTime for this pid
//   memory          task_vm_info phys_footprint (incl. Metal buffers) + max RSS
//   SoC energy      IOReport "Energy Model": CPU, GPU, DRAM, ANE. Package-wide,
//                   not per-process. Unprivileged, only "GPU Energy" moves on
//                   macOS 27; the rest needs root. A channel that did not move
//                   across the whole run is reported as `na`, never as 0.
//   system power    SMC PSTR, sampled every 20 ms on a thread and integrated.
//                   Whole machine, display included; read it against idle.
//                   Updates about once a second, so whole-run only.

#include "llama.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static uint64_t cpu_ns(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (uint64_t) (ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000000000ull +
           (uint64_t) (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1000ull;
}

static int has_flag(const char *flags, const char *name) {
    const size_t len = strlen(name);
    for (const char *p = flags; *p; ) {
        const char *end = strchr(p, ',');
        const size_t n = end ? (size_t) (end - p) : strlen(p);
        if (n == len && strncmp(p, name, len) == 0) return 1;
        if (!end) break;
        p = end + 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// GPU time, per process
//
// Every Metal client gets an AGXDeviceUserClient under the accelerator, tagged
// "pid N, name"; its AppUsage array accumulates GPU time in nanoseconds. This
// is what Activity Monitor's GPU column reads. A process may hold several
// clients, so they are summed.

static uint64_t gpu_ns(void) {
    io_iterator_t accels;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &accels)) return 0;
    char want[32];
    snprintf(want, sizeof want, "pid %d,", getpid());
    uint64_t total = 0;
    io_object_t accel;
    while ((accel = IOIteratorNext(accels))) {
        io_iterator_t kids;
        if (IORegistryEntryGetChildIterator(accel, kIOServicePlane, &kids) == KERN_SUCCESS) {
            io_object_t kid;
            while ((kid = IOIteratorNext(kids))) {
                CFStringRef creator = IORegistryEntryCreateCFProperty(kid, CFSTR("IOUserClientCreator"), kCFAllocatorDefault, 0);
                char buf[128] = {0};
                if (creator && CFGetTypeID(creator) == CFStringGetTypeID())
                    CFStringGetCString(creator, buf, sizeof buf, kCFStringEncodingUTF8);
                if (creator) CFRelease(creator);
                if (strncmp(buf, want, strlen(want)) == 0) {
                    CFArrayRef usage = IORegistryEntryCreateCFProperty(kid, CFSTR("AppUsage"), kCFAllocatorDefault, 0);
                    if (usage && CFGetTypeID(usage) == CFArrayGetTypeID()) {
                        for (CFIndex i = 0; i < CFArrayGetCount(usage); i++) {
                            CFDictionaryRef d = CFArrayGetValueAtIndex(usage, i);
                            CFNumberRef t = CFDictionaryGetValue(d, CFSTR("accumulatedGPUTime"));
                            int64_t v = 0;
                            if (t && CFNumberGetValue(t, kCFNumberSInt64Type, &v)) total += (uint64_t) v;
                        }
                    }
                    if (usage) CFRelease(usage);
                }
                IOObjectRelease(kid);
            }
            IOObjectRelease(kids);
        }
        IOObjectRelease(accel);
    }
    IOObjectRelease(accels);
    return total;
}

// ---------------------------------------------------------------------------
// SoC energy through IOReport
//
// libIOReport is private but ships a .tbd in the SDK. These declarations are
// the ones macmon, asitop and friends use.

typedef struct IOReportSubscription *IOReportSubscriptionRef;
extern CFDictionaryRef IOReportCopyChannelsInGroup(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
extern IOReportSubscriptionRef IOReportCreateSubscription(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
extern CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
extern CFStringRef IOReportChannelGetUnitLabel(CFDictionaryRef);
extern int IOReportChannelGetFormat(CFDictionaryRef);
extern int64_t IOReportSimpleGetIntegerValue(CFDictionaryRef, int32_t);

static IOReportSubscriptionRef ior_sub;
static CFMutableDictionaryRef ior_chans;

static void ior_init(void) {
    CFDictionaryRef e = IOReportCopyChannelsInGroup(CFSTR("Energy Model"), NULL, 0, 0, 0);
    if (!e) return;
    CFMutableDictionaryRef ch = CFDictionaryCreateMutableCopy(NULL, 0, e);
    CFRelease(e);
    ior_sub = IOReportCreateSubscription(NULL, ch, &ior_chans, 0, NULL);
}

static CFDictionaryRef ior_sample(void) {
    return ior_sub ? IOReportCreateSamples(ior_sub, ior_chans, NULL) : NULL;
}

typedef struct { double cpu, gpu, dram, ane; } Energy; // joules

// Energy between two samples. "GPU Energy" and not "GPU0": the two cover the
// same block, and only the former moves without root.
static Energy ior_delta(CFDictionaryRef a, CFDictionaryRef b) {
    Energy e = {0};
    if (!a || !b) return e;
    CFDictionaryRef d = IOReportCreateSamplesDelta(a, b, NULL);
    if (!d) return e;
    CFArrayRef arr = CFDictionaryGetValue(d, CFSTR("IOReportChannels"));
    for (CFIndex i = 0; arr && i < CFArrayGetCount(arr); i++) {
        CFDictionaryRef c = CFArrayGetValueAtIndex(arr, i);
        if (IOReportChannelGetFormat(c) != 1) continue;
        char name[128] = {0}, unit[16] = {0};
        CFStringRef n = IOReportChannelGetChannelName(c), u = IOReportChannelGetUnitLabel(c);
        if (n) CFStringGetCString(n, name, sizeof name, kCFStringEncodingUTF8);
        if (u) CFStringGetCString(u, unit, sizeof unit, kCFStringEncodingUTF8);
        const double scale = strcmp(unit, "mJ") == 0 ? 1e-3 : strcmp(unit, "uJ") == 0 ? 1e-6 :
                             strcmp(unit, "nJ") == 0 ? 1e-9 : 0.0;
        const double j = (double) IOReportSimpleGetIntegerValue(c, 0) * scale;
        if (strcmp(name, "CPU Energy") == 0) e.cpu += j;
        else if (strcmp(name, "GPU Energy") == 0) e.gpu += j;
        else if (strncmp(name, "DRAM", 4) == 0) e.dram += j;
        else if (strncmp(name, "ANE", 3) == 0) e.ane += j;
    }
    CFRelease(d);
    return e;
}

// ---------------------------------------------------------------------------
// System power through the SMC
//
// PSTR is total system power in watts as a float. Measured on an M5 Max: it
// changes about once a second (0.74 s, then 0.99 s), so it cannot resolve a
// phase -- a 0.55 s generation read 17 W against a 22 W idle. What it can do
// is account for a whole run: a smoothed reading lags but conserves area, so
// the integral above idle from the prompt to PERF_TAIL_MS past the last token
// captures what the run added. That is the only figure derived from it. The
// struct layout is the AppleSMC user-client ABI and must not be reordered.

typedef struct { char major, minor, build, reserved; uint16_t release; } SmcVers;
typedef struct { uint16_t version, length; uint32_t cpu, gpu, mem; } SmcPLimit;
typedef struct { uint32_t size, type; char attrs; } SmcKeyInfo;
typedef struct {
    uint32_t key; SmcVers vers; SmcPLimit plimit; SmcKeyInfo info;
    char result, status, cmd; uint32_t data32; unsigned char bytes[32];
} SmcParam;

static io_connect_t smc_conn;

static int smc_read_float(const char *key, float *out) {
    SmcParam in = {0}, o = {0};
    size_t sz = sizeof o;
    in.key = ((uint32_t) key[0] << 24) | ((uint32_t) key[1] << 16) | ((uint32_t) key[2] << 8) | (uint32_t) key[3];
    in.cmd = 9; // key info
    if (IOConnectCallStructMethod(smc_conn, 2, &in, sizeof in, &o, &sz) || o.result) return -1;
    if (o.info.type != (('f' << 24) | ('l' << 16) | ('t' << 8) | ' ')) return -1;
    in.info.size = o.info.size;
    in.cmd = 5; // read bytes
    memset(&o, 0, sizeof o);
    sz = sizeof o;
    if (IOConnectCallStructMethod(smc_conn, 2, &in, sizeof in, &o, &sz) || o.result) return -1;
    memcpy(out, o.bytes, sizeof *out);
    return 0;
}

static struct {
    pthread_mutex_t mu;
    double joules, last_w;
    uint64_t last_ns;
    atomic_int stop;
    int ok;
} smc = { .mu = PTHREAD_MUTEX_INITIALIZER };

static void *smc_loop(void *arg) {
    (void) arg;
    while (!atomic_load(&smc.stop)) {
        float w;
        if (smc_read_float("PSTR", &w) == 0) {
            const uint64_t t = now_ns();
            pthread_mutex_lock(&smc.mu);
            if (smc.last_ns) smc.joules += 0.5 * (w + smc.last_w) * (double) (t - smc.last_ns) * 1e-9;
            smc.last_w = w;
            smc.last_ns = t;
            pthread_mutex_unlock(&smc.mu);
        }
        usleep(20000);
    }
    return NULL;
}

static pthread_t smc_thread;

static void smc_start(void) {
    io_service_t s = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"));
    if (!s || IOServiceOpen(s, mach_task_self(), 0, &smc_conn)) return;
    float w;
    if (smc_read_float("PSTR", &w)) return;
    smc.ok = 1;
    pthread_create(&smc_thread, NULL, smc_loop, NULL);
    usleep(50000); // two samples in, so the integral has a start
}

// The integral so far, extended at the last reading to now.
static double smc_joules(void) {
    if (!smc.ok) return 0;
    pthread_mutex_lock(&smc.mu);
    const double j = smc.joules + smc.last_w * (double) (now_ns() - smc.last_ns) * 1e-9;
    pthread_mutex_unlock(&smc.mu);
    return j;
}

// ---------------------------------------------------------------------------
// Snapshots at phase boundaries

typedef struct {
    uint64_t wall, cpu, gpu;
    double sys_j;
    CFDictionaryRef ior;
} Snap;

static Snap snap(void) {
    Snap s;
    s.wall = now_ns();
    s.cpu = cpu_ns();
    s.gpu = gpu_ns();
    s.sys_j = smc_joules();
    s.ior = ior_sample();
    return s;
}

static void footprint(uint64_t *now, uint64_t *peak) {
    task_vm_info_data_t vi;
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &vi, &n) != KERN_SUCCESS) { *now = *peak = 0; return; }
    *now = vi.phys_footprint;
    *peak = vi.ledger_phys_footprint_peak;
}

static void quiet(enum ggml_log_level level, const char *text, void *user) { (void) level; (void) text; (void) user; }

// Follows harness/raw_completion.c:warm_up. Without it the first decode
// carries Metal pipeline compilation, and TTFT measures the shader compiler.
static void warm_up(struct llama_model *model, struct llama_context *ctx, const struct llama_vocab *vocab) {
    llama_token tmp[2];
    int n = 0;
    const llama_token bos = llama_vocab_bos(vocab), eos = llama_vocab_eos(vocab);
    if (bos != LLAMA_TOKEN_NULL) tmp[n++] = bos;
    if (eos != LLAMA_TOKEN_NULL) tmp[n++] = eos;
    if (n == 0) { tmp[0] = 0; n = 1; }
    if (llama_model_has_decoder(model)) llama_decode(ctx, llama_batch_get_one(tmp, n));
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_synchronize(ctx);
}

// "na" for a channel that never moved, so a reader cannot mistake a counter
// we could not read for a block that drew no power.
static void watts(const char *key, double j, double s, int live) {
    if (live && s > 0) printf("\t%s=%.3f", key, j / s);
    else printf("\t%s=na", key);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: perf_driver <model.gguf> <prompt> <n_predict> <label> [raw,show,cpu]\n");
        return 2;
    }
    const char *model_path = argv[1], *prompt = argv[2], *label = argv[4];
    const int n_predict = atoi(argv[3]);
    const char *flags = argc > 5 ? argv[5] : "";
    const int idle_ms = getenv("PERF_IDLE_MS") ? atoi(getenv("PERF_IDLE_MS")) : 2000;
    const int tail_ms = getenv("PERF_TAIL_MS") ? atoi(getenv("PERF_TAIL_MS")) : 1500;
    const int cpu_only = has_flag(flags, "cpu"), show = has_flag(flags, "show");

    ior_init();
    smc_start();

    // Idle: nothing of ours loaded yet.
    const Snap s_idle0 = snap();
    usleep((useconds_t) idle_ms * 1000);
    const Snap s_idle1 = snap();

    // Load: backend init, weights, context.
    const uint64_t t_load0 = now_ns();
    llama_log_set(quiet, NULL);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    ggml_backend_dev_t devs[2] = { NULL, NULL };
    if (cpu_only) {
        devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        mp.devices = devs;
        mp.n_gpu_layers = 0;
    }
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path); return 1; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_predict + 1024;
    cp.n_batch = 1024;
    cp.no_perf = false;
    if (cpu_only) { cp.offload_kqv = false; cp.op_offload = false; }
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create the context\n"); return 1; }
    const uint64_t t_load1 = now_ns();
    uint64_t fp_loaded, fp_peak_loaded;
    footprint(&fp_loaded, &fp_peak_loaded);

    const uint64_t t_warm0 = now_ns();
    warm_up(model, ctx, vocab);
    const uint64_t t_warm1 = now_ns();

    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    llama_perf_context_reset(ctx);

    // Prompt: TTFT runs from the prompt string to the first sampled token, so
    // it includes templating and tokenization, as a user would see it.
    const Snap s_p0 = snap();
    const uint64_t t_p0 = now_ns();

    char *text = NULL;
    const char *tmpl = llama_model_chat_template(model, NULL);
    if (!has_flag(flags, "raw") && tmpl) {
        struct llama_chat_message msg = { "user", prompt };
        const int need = llama_chat_apply_template(tmpl, &msg, 1, true, NULL, 0);
        if (need > 0) {
            text = malloc((size_t) need + 1);
            llama_chat_apply_template(tmpl, &msg, 1, true, text, need + 1);
            text[need] = 0;
        }
    }
    const char *input = text ? text : prompt;
    llama_token toks[4096];
    const int n_prompt = llama_tokenize(vocab, input, (int) strlen(input), toks, 4096, true, true);
    if (n_prompt < 0) { fprintf(stderr, "prompt too long\n"); return 1; }

    uint64_t hash = 1469598103934665603ull; // FNV-1a over token ids
    int first_eog = -1;
    llama_token id;
    if (llama_decode(ctx, llama_batch_get_one(toks, n_prompt))) return 1;
    id = llama_sampler_sample(smpl, ctx, -1);
    const uint64_t t_p1 = now_ns();
    const Snap s_p1 = snap();

    // Generation: the remaining n_predict - 1 tokens, one decode each.
    const uint64_t t_g0 = now_ns();
    for (int i = 0; ; i++) {
        hash = (hash ^ (uint64_t) (uint32_t) id) * 1099511628211ull;
        if (first_eog < 0 && llama_vocab_is_eog(vocab, id)) first_eog = i;
        if (show) {
            char buf[256];
            const int k = llama_token_to_piece(vocab, id, buf, sizeof buf, 0, true);
            if (k > 0) fwrite(buf, 1, (size_t) k, stderr);
        }
        if (i + 1 >= n_predict) break;
        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) return 1;
        id = llama_sampler_sample(smpl, ctx, -1);
    }
    const uint64_t t_g1 = now_ns();
    const Snap s_g1 = snap();
    if (show) fputc('\n', stderr);

    uint64_t fp_end, fp_peak;
    footprint(&fp_end, &fp_peak);
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    const struct llama_perf_context_data perf = llama_perf_context(ctx);

    usleep((useconds_t) tail_ms * 1000);
    const uint64_t t_tail = now_ns();
    const double sys_j_tail = smc_joules();
    atomic_store(&smc.stop, 1);
    if (smc.ok) pthread_join(smc_thread, NULL);

    const double idle_s = (double) (s_idle1.wall - s_idle0.wall) * 1e-9;
    const double ttft_s = (double) (t_p1 - t_p0) * 1e-9;
    const double gen_s = (double) (t_g1 - t_g0) * 1e-9;
    const int n_gen = n_predict - 1; // tokens after the first
    const double gen_win_s = (double) (s_g1.wall - s_p1.wall) * 1e-9;
    const double run_win_s = (double) (s_g1.wall - s_p0.wall) * 1e-9;

    const Energy e_idle = ior_delta(s_idle0.ior, s_idle1.ior);
    const Energy e_gen = ior_delta(s_p1.ior, s_g1.ior);
    const Energy e_run = ior_delta(s_p0.ior, s_g1.ior);
    // A channel is live when it moved at all from the first sample to the last.
    const Energy e_all = ior_delta(s_idle0.ior, s_g1.ior);
    const int cpu_live = e_all.cpu > 0, gpu_live = e_all.gpu > 0, dram_live = e_all.dram > 0, ane_live = e_all.ane > 0;

    const double sys_idle_w = smc.ok ? (s_idle1.sys_j - s_idle0.sys_j) / idle_s : 0;
    // Energy the run added to the machine: prompt through tail, above idle.
    const double sys_run_j = smc.ok ? (sys_j_tail - s_p0.sys_j) - sys_idle_w * (double) (t_tail - s_p0.wall) * 1e-9 : 0;

    printf("label=%s", label);
    printf("\tn_prompt=%d\tn_gen=%d\ttok_hash=%016llx\tfirst_eog=%d", n_prompt, n_gen, (unsigned long long) hash, first_eog);
    printf("\tload_ms=%.1f\twarmup_ms=%.1f", (t_load1 - t_load0) * 1e-6, (t_warm1 - t_warm0) * 1e-6);
    printf("\tttft_ms=%.2f\tpp_tps=%.1f", ttft_s * 1e3, n_prompt / ttft_s);
    printf("\ttg_tps=%.2f\ttg_ms_per_tok=%.3f", n_gen / gen_s, gen_s * 1e3 / n_gen);
    printf("\tperf_tg_tps=%.2f", perf.t_eval_ms > 0 ? 1e3 * perf.n_eval / perf.t_eval_ms : 0.0);
    printf("\tfootprint_loaded_mb=%.1f\tfootprint_peak_mb=%.1f\trss_peak_mb=%.1f",
           fp_loaded / 1048576.0, fp_peak / 1048576.0, ru.ru_maxrss / 1048576.0);
    printf("\tcpu_load_s=%.3f\tcpu_gen_s=%.3f\tcpu_gen_cores=%.3f",
           (s_p0.cpu - s_idle1.cpu) * 1e-9, (s_g1.cpu - s_p1.cpu) * 1e-9, (s_g1.cpu - s_p1.cpu) * 1e-9 / gen_win_s);
    printf("\tgpu_gen_s=%.3f\tgpu_gen_util=%.1f", (s_g1.gpu - s_p1.gpu) * 1e-9, 100.0 * (s_g1.gpu - s_p1.gpu) * 1e-9 / gen_win_s);
    printf("\tgpu_ttft_ms=%.2f", (s_p1.gpu - s_p0.gpu) * 1e-6);
    watts("gpu_w_idle", e_idle.gpu, idle_s, gpu_live);
    watts("gpu_w_gen", e_gen.gpu, gen_win_s, gpu_live);
    watts("cpu_w_idle", e_idle.cpu, idle_s, cpu_live);
    watts("cpu_w_gen", e_gen.cpu, gen_win_s, cpu_live);
    watts("dram_w_gen", e_gen.dram, gen_win_s, dram_live);
    watts("ane_w_gen", e_gen.ane, gen_win_s, ane_live);
    if (smc.ok) printf("\tsys_w_idle=%.2f\tsys_j_run=%.3f", sys_idle_w, sys_run_j);
    else printf("\tsys_w_idle=na\tsys_j_run=na");
    // Energy per token. SoC: whichever IOReport channels moved, over the
    // generation phase. System: the whole run above idle, over every token
    // the run produced, since the sensor cannot separate prompt from tokens.
    const double soc_j = (gpu_live ? e_gen.gpu : 0) + (cpu_live ? e_gen.cpu : 0) + (dram_live ? e_gen.dram : 0);
    printf("\tsoc_mj_per_tok=%.2f", 1e3 * soc_j / n_gen);
    if (smc.ok) printf("\tsys_mj_per_tok=%.2f", 1e3 * sys_run_j / n_predict);
    else printf("\tsys_mj_per_tok=na");
    printf("\tgpu_j_run=%.3f\trun_s=%.3f", gpu_live ? e_run.gpu : 0.0, run_win_s);
    printf("\troot=%d\n", geteuid() == 0);

    free(text);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
