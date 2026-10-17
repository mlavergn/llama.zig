/* Exercises the CPU backend's host arms -- thread priority, CPU affinity,
 * NUMA, and the sampler's default seed -- and prints what each thread ended
 * up with, read back from the OS.
 *
 * Built twice by scripts/host-arms-diff -- once against the stock libraries
 * from `llama.cpp.zmake`, once against ours -- and the two transcripts are
 * diffed. It talks to the libraries through `ggml.h`, `ggml-cpu.h` and
 * `llama.h` alone, so the same source links against either side unchanged.
 *
 * Why nothing else can do this
 * ----------------------------
 * These are the arms of `ggml-cpu.c` and `llama-sampler.cpp` that call the
 * OS rather than compute: `ggml_thread_apply_priority`,
 * `ggml_thread_apply_affinity`, `ggml_numa_init` and its
 * `set_numa_thread_affinity`, and `get_rng_seed`. Every value-level gate
 * compares *results*, and a thread at the wrong priority or on the wrong CPU
 * computes the same bits. The only observable is the thread's own state, so
 * a probe op runs on every worker of a real threadpool and reads back its
 * scheduling policy, priority and affinity mask.
 *
 * Each scenario runs in a fresh process (`host_arms run <name>`): the C
 * applies the pool's priority and affinity to the *calling* thread as well,
 * and `ggml_numa_init` takes effect once per process, so one scenario would
 * otherwise leak into the next.
 *
 * Messages: the C writes its warnings with `fprintf(stderr, ...)`, not
 * through ggml's logger, and this installs a log callback that prefixes
 * everything it receives with `log:` on stdout -- so a message that went
 * through the wrong channel lands in the other stream and the diff sees it.
 * The script sorts stderr before comparing, because worker threads print
 * their warnings in whatever order they run.
 *
 * Usage: host_arms list | host_arms run <scenario>
 */

#if defined(__linux__)
#define _GNU_SOURCE
#endif

#include "ggml.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// ---------------------------------------------------------------------------
// Reading a thread's state back
// ---------------------------------------------------------------------------

#define MAX_THREADS 16

struct thread_state {
    int  seen;
    int  policy;
    int  priority;
    char cpus[512];
};

static struct thread_state workers[MAX_THREADS];

static const char * policy_name(int policy) {
    switch (policy) {
        case SCHED_OTHER: return "other";
        case SCHED_FIFO:  return "fifo";
        case SCHED_RR:    return "rr";
#if defined(SCHED_BATCH)
        case SCHED_BATCH: return "batch";
#endif
#if defined(SCHED_IDLE)
        case SCHED_IDLE:  return "idle";
#endif
        default:          return "unknown";
    }
}

#if defined(__linux__)
// "0-3,8,10-11": the affinity mask as ranges.
static void cpus_string(const cpu_set_t * set, char * out, size_t size) {
    size_t len = 0;
    out[0] = '\0';
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (!CPU_ISSET(i, set)) continue;
        int j = i;
        while (j + 1 < CPU_SETSIZE && CPU_ISSET(j + 1, set)) j++;
        int n = j == i ? snprintf(out + len, size - len, "%s%d", len ? "," : "", i)
                       : snprintf(out + len, size - len, "%s%d-%d", len ? "," : "", i, j);
        if (n < 0 || (size_t) n >= size - len) break;
        len += (size_t) n;
        i = j;
    }
}
#endif

static void read_self(struct thread_state * st) {
    struct sched_param sp;
    int policy = -1;
    if (pthread_getschedparam(pthread_self(), &policy, &sp) != 0) {
        policy = -1;
        sp.sched_priority = -1;
    }
    st->policy   = policy;
    st->priority = sp.sched_priority;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (pthread_getaffinity_np(pthread_self(), sizeof(set), &set) == 0) {
        cpus_string(&set, st->cpus, sizeof(st->cpus));
    } else {
        snprintf(st->cpus, sizeof(st->cpus), "error");
    }
#else
    // ggml_thread_apply_affinity is a no-op on Apple platforms.
    snprintf(st->cpus, sizeof(st->cpus), "n/a");
#endif
    st->seen = 1;
}

static void print_state(const char * scenario, const char * who, const struct thread_state * st) {
    printf("%s %s: policy=%s priority=%d cpus=%s\n", scenario, who, policy_name(st->policy), st->priority, st->cpus);
}

// The probe op: every thread of the pool runs it once, `ith` 0 being the
// calling thread.
static void probe(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    (void) dst;
    (void) a;
    (void) userdata;
    if (ith < MAX_THREADS && nth <= MAX_THREADS) read_self(&workers[ith]);
}

// ---------------------------------------------------------------------------
// Running a pool
// ---------------------------------------------------------------------------

// The CPUs this process may run on at the start, which the mask scenarios
// draw from; on Apple platforms, where affinity is a no-op, 0 and 1.
static int allowed[2];

static void find_allowed_cpus(void) {
    allowed[0] = 0;
    allowed[1] = 1;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (pthread_getaffinity_np(pthread_self(), sizeof(set), &set) != 0) return;
    int k = 0;
    for (int i = 0; i < CPU_SETSIZE && k < 2; i++) {
        if (CPU_ISSET(i, &set)) allowed[k++] = i;
    }
#endif
}

static int run_pool(const char * scenario, struct ggml_threadpool_params * tpp) {
    struct thread_state st;

    read_self(&st);
    print_state(scenario, "main.start", &st);

    memset(workers, 0, sizeof(workers));
    struct ggml_threadpool * tp = ggml_threadpool_new(tpp);
    if (tp == NULL) {
        printf("%s threadpool: NULL\n", scenario);
        return 1;
    }
    read_self(&st);
    print_state(scenario, "main.after_new", &st);

    struct ggml_init_params ip = { 1024 * 1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    memset(a->data, 0, ggml_nbytes(a));
    struct ggml_tensor * t = ggml_map_custom1(ctx, a, probe, GGML_N_TASKS_MAX, NULL);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);

    struct ggml_cplan cplan = ggml_graph_plan(gf, tpp->n_threads, tp);
    void * work = cplan.work_size > 0 ? malloc(cplan.work_size) : NULL;
    cplan.work_data = work;
    enum ggml_status status = ggml_graph_compute(gf, &cplan);
    printf("%s compute: status=%d\n", scenario, (int) status);

    for (int i = 0; i < tpp->n_threads && i < MAX_THREADS; i++) {
        char who[32];
        snprintf(who, sizeof(who), "thread.%d", i);
        if (workers[i].seen) {
            print_state(scenario, who, &workers[i]);
        } else {
            printf("%s %s: did not run\n", scenario, who);
        }
    }
    read_self(&st);
    print_state(scenario, "main.after_compute", &st);

    free(work);
    ggml_free(ctx);
    ggml_threadpool_free(tp);
    return 0;
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

#define N_THREADS 3

static int prio_scenario(const char * name, enum ggml_sched_priority prio) {
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(N_THREADS);
    tpp.prio = prio;
    return run_pool(name, &tpp);
}

// mask: the first two allowed CPUs, or `bad` -- a CPU index no machine this
// runs on has, which makes `pthread_setaffinity_np` fail with EINVAL.
static int mask_scenario(const char * name, bool strict, bool paused, enum ggml_sched_priority prio, bool bad) {
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(N_THREADS);
    if (bad) {
        tpp.cpumask[GGML_MAX_N_THREADS - 1] = true;
    } else {
        tpp.cpumask[allowed[0]] = true;
        tpp.cpumask[allowed[1]] = true;
    }
    tpp.strict_cpu = strict;
    tpp.paused     = paused;
    tpp.prio       = prio;
    return run_pool(name, &tpp);
}

static int numa_scenario(const char * name, enum ggml_numa_strategy strategy, bool twice) {
    ggml_numa_init(strategy);
    if (twice) ggml_numa_init(strategy);
    printf("%s is_numa: %d\n", name, (int) ggml_is_numa());
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(N_THREADS);
    return run_pool(name, &tpp);
}

static uint32_t now_us32(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint32_t) ((uint64_t) tv.tv_sec * 1000000u + (uint64_t) tv.tv_usec);
}

// LLAMA_DEFAULT_SEED draws a seed: from `std::random_device` where libc++
// says it is a true RNG, from the system clock in microseconds where it says
// it is not. The value is unrepeatable, so what is compared is which source
// it came from: five draws all within the clock's window are the clock --
// a random seed lands in a one-millisecond window with odds of 2^-22 a draw.
static int seed_scenario(const char * name) {
    int clock_like = 0;
    int not_default = 0;
    for (int i = 0; i < 5; i++) {
        uint32_t before = now_us32();
        struct llama_sampler * s = llama_sampler_init_dist(LLAMA_DEFAULT_SEED);
        uint32_t seed = llama_sampler_get_seed(s);
        uint32_t after = now_us32();
        llama_sampler_free(s);
        if (seed != LLAMA_DEFAULT_SEED) not_default++;
        if ((uint32_t) (seed - before) <= (uint32_t) (after - before) + 1000u) clock_like++;
    }
    printf("%s drawn: %d/5 differ from LLAMA_DEFAULT_SEED, source=%s\n", name, not_default,
           clock_like == 5 ? "clock" : clock_like == 0 ? "random_device" : "mixed");
    return 0;
}

static const char * const scenarios[] = {
    "prio.low", "prio.normal", "prio.medium", "prio.high", "prio.realtime",
    "mask.loose", "mask.strict", "mask.strict_paused", "mask.low", "mask.bad", "mask.bad_strict",
    "numa.disabled", "numa.distribute", "numa.isolate", "numa.numactl", "numa.mirror", "numa.twice",
    "seed.default",
};

static void log_to_stdout(enum ggml_log_level level, const char * text, void * user) {
    (void) user;
    printf("log: %d %s", (int) level, text);
}

int main(int argc, char ** argv) {
    if (argc == 2 && strcmp(argv[1], "list") == 0) {
        for (size_t i = 0; i < sizeof(scenarios) / sizeof(scenarios[0]); i++) printf("%s\n", scenarios[i]);
        return 0;
    }
    if (argc != 3 || strcmp(argv[1], "run") != 0) {
        fprintf(stderr, "usage: host_arms list | host_arms run <scenario>\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    ggml_log_set(log_to_stdout, NULL);
    llama_log_set(log_to_stdout, NULL);
    find_allowed_cpus();

    const char * n = argv[2];
    if (strcmp(n, "prio.low") == 0)           return prio_scenario(n, GGML_SCHED_PRIO_LOW);
    if (strcmp(n, "prio.normal") == 0)        return prio_scenario(n, GGML_SCHED_PRIO_NORMAL);
    if (strcmp(n, "prio.medium") == 0)        return prio_scenario(n, GGML_SCHED_PRIO_MEDIUM);
    if (strcmp(n, "prio.high") == 0)          return prio_scenario(n, GGML_SCHED_PRIO_HIGH);
    if (strcmp(n, "prio.realtime") == 0)      return prio_scenario(n, GGML_SCHED_PRIO_REALTIME);
    if (strcmp(n, "mask.loose") == 0)         return mask_scenario(n, false, false, GGML_SCHED_PRIO_NORMAL, false);
    if (strcmp(n, "mask.strict") == 0)        return mask_scenario(n, true,  false, GGML_SCHED_PRIO_NORMAL, false);
    if (strcmp(n, "mask.strict_paused") == 0) return mask_scenario(n, true,  true,  GGML_SCHED_PRIO_NORMAL, false);
    if (strcmp(n, "mask.low") == 0)           return mask_scenario(n, false, false, GGML_SCHED_PRIO_LOW,    false);
    if (strcmp(n, "mask.bad") == 0)           return mask_scenario(n, false, false, GGML_SCHED_PRIO_NORMAL, true);
    if (strcmp(n, "mask.bad_strict") == 0)    return mask_scenario(n, true,  false, GGML_SCHED_PRIO_NORMAL, true);
    if (strcmp(n, "numa.disabled") == 0)      return numa_scenario(n, GGML_NUMA_STRATEGY_DISABLED,   false);
    if (strcmp(n, "numa.distribute") == 0)    return numa_scenario(n, GGML_NUMA_STRATEGY_DISTRIBUTE, false);
    if (strcmp(n, "numa.isolate") == 0)       return numa_scenario(n, GGML_NUMA_STRATEGY_ISOLATE,    false);
    if (strcmp(n, "numa.numactl") == 0)       return numa_scenario(n, GGML_NUMA_STRATEGY_NUMACTL,    false);
    if (strcmp(n, "numa.mirror") == 0)        return numa_scenario(n, GGML_NUMA_STRATEGY_MIRROR,     false);
    if (strcmp(n, "numa.twice") == 0)         return numa_scenario(n, GGML_NUMA_STRATEGY_DISTRIBUTE, true);
    if (strcmp(n, "seed.default") == 0)       return seed_scenario(n);
    fprintf(stderr, "unknown scenario: %s\n", n);
    return 2;
}
