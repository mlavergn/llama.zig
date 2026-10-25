// Not a port: compares text outputs on the Windows laptop, line by line.
//
// The gates that run there through `zupdater push --run` cannot bring
// a file back to the Mac, so the comparison happens on the laptop and only
// its verdict travels: the line counts, the first few differing lines, and a
// final `PASS:` or `FAIL:` line that `scripts/x86-ops-diff` reads. Lines are
// cut to 76 columns, because the output comes back through ConPTY, which
// wraps at the console's width.
//
// Usage: win_lines_diff <ref name> <port name> [<ref name> <port name>]...
//   Names are under %TEMP%\llamazig, as `harness/win_ops_dump.c` writes them.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void temp_path(char * buf, size_t len, const char * name) {
    const char * tmp = getenv("TEMP");
    snprintf(buf, len, "%s\\llamazig\\%s", tmp ? tmp : ".", name);
}

static void show(const char * tag, const char * line) {
    char cut[77];
    size_t n = strcspn(line, "\r\n");
    if (n > 70) n = 70;
    memcpy(cut, line, n);
    cut[n] = 0;
    printf("  %s %s\n", tag, cut);
}

int main(int argc, char ** argv) {
    if (argc < 3 || (argc - 1) % 2) {
        fprintf(stderr, "usage: %s <ref name> <port name> ...\n", argv[0]);
        return 2;
    }
    long total = 0, differ = 0;
    for (int k = 1; k + 1 < argc; k += 2) {
        char pa[1024], pb[1024];
        temp_path(pa, sizeof pa, argv[k]);
        temp_path(pb, sizeof pb, argv[k + 1]);
        FILE * a = fopen(pa, "r");
        FILE * b = fopen(pb, "r");
        if (!a || !b) {
            printf("FAIL: cannot open %s or %s\n", argv[k], argv[k + 1]);
            return 1;
        }
        static char la[1 << 16], lb[1 << 16];
        long n = 0, bad = 0;
        for (;;) {
            char * ra = fgets(la, sizeof la, a);
            char * rb = fgets(lb, sizeof lb, b);
            if (!ra && !rb) break;
            n++;
            if (!ra || !rb || strcmp(la, lb) != 0) {
                if (bad < 3) {
                    printf("%s line %ld:\n", argv[k], n);
                    show("ref ", ra ? la : "<eof>");
                    show("port", rb ? lb : "<eof>");
                }
                bad++;
            }
        }
        fclose(a);
        fclose(b);
        printf("%s: %ld lines, %ld differ\n", argv[k], n, bad);
        total += n;
        differ += bad;
    }
    if (differ) {
        printf("FAIL: %ld of %ld lines differ\n", differ, total);
        return 1;
    }
    printf("PASS: %ld lines identical\n", total);
    return 0;
}
