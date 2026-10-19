// Not a port: `harness/ops_dump.c` as `scripts/x86-ops-diff` runs it on the
// Windows laptop.
//
// `inferiseupdater push --run` starts a pushed program with arguments only:
// no environment, no redirection, and no way to bring a file back. So this
// wrapper takes what `scripts/ops-diff` passes through the shell as
// arguments -- the thread count `OPS_THREADS` reads, the file stdout goes
// to -- and keeps every file under `%TEMP%\llamazig`, where
// `harness/win_lines_diff.c` compares them on the same machine.
//
// Usage: win_ops_dump <threads> <out name> <write-q|read-q> <q name>

#define main ops_dump_main
#include "ops_dump.c"
#undef main

#include <direct.h>

static void temp_path(char * buf, size_t len, const char * name) {
    const char * tmp = getenv("TEMP");
    snprintf(buf, len, "%s\\llamazig\\%s", tmp ? tmp : ".", name);
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <threads> <out name> <write-q|read-q> <q name>\n", argv[0]);
        return 2;
    }
    char dir[1024], out[1024], q[1024];
    temp_path(dir, sizeof dir, "");
    _mkdir(dir);
    temp_path(out, sizeof out, argv[2]);
    temp_path(q, sizeof q, argv[4]);

    if (_putenv_s("OPS_THREADS", argv[1]) != 0) { fprintf(stderr, "cannot set OPS_THREADS\n"); return 2; }
    if (!freopen(out, "w", stdout)) { fprintf(stderr, "cannot open %s\n", out); return 2; }

    char * av[] = { argv[0], argv[3], q, NULL };
    const int rc = ops_dump_main(3, av);
    fflush(stdout);
    fprintf(stderr, "ops_dump rc=%d\n", rc);
    return rc;
}
