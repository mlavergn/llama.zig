// Not a port: `harness/node_dump.c` as `scripts/x86-node-diff` runs it on
// the Windows laptop.
//
// As `harness/win_ops_dump.c` does for ops_dump: `push --run` passes
// arguments only, so stdout goes to a file under `%TEMP%\llamazig` named
// here, and `harness/win_lines_diff.c` compares the two sides there. A model
// path that is not absolute is taken under `%USERPROFILE%`, so a script can
// name a model on the laptop without knowing its user's name.
//
// Usage: win_node_dump <out name> <model> <prompt> <threads> [cpu|gpu] [fused <n>]

#define main node_dump_main
#include "node_dump.c"
#undef main

#include <direct.h>

static void temp_path(char * buf, size_t len, const char * name) {
    const char * tmp = getenv("TEMP");
    snprintf(buf, len, "%s\\llamazig\\%s", tmp ? tmp : ".", name);
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <out name> <model> <prompt> <threads> [cpu|gpu] [fused <n>]\n", argv[0]);
        return 2;
    }
    char dir[1024], out[1024], model[1024];
    temp_path(dir, sizeof dir, "");
    _mkdir(dir);
    temp_path(out, sizeof out, argv[1]);
    const int absolute = (argv[2][0] && argv[2][1] == ':') || argv[2][0] == '\\';
    if (absolute) {
        snprintf(model, sizeof model, "%s", argv[2]);
    } else {
        const char * home = getenv("USERPROFILE");
        snprintf(model, sizeof model, "%s\\%s", home ? home : ".", argv[2]);
    }
    if (!freopen(out, "w", stdout)) { fprintf(stderr, "cannot open %s\n", out); return 2; }

    argv[1] = argv[0];
    argv[2] = model;
    const int rc = node_dump_main(argc - 1, argv + 1);
    fflush(stdout);
    fprintf(stderr, "node_dump rc=%d\n", rc);
    return rc;
}
