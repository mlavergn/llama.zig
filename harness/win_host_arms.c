// Not a port: `harness/host_arms.c` as `scripts/host-arms-diff` runs it on
// the Windows laptop.
//
// The script runs every scenario in a fresh process per side and builds a
// transcript of each: stdout, the exit code, then stderr sorted, with the
// affinity warning's address normalised. `zupdater push --run`
// starts one program with arguments and no redirection, so this does the
// script's half on the laptop: `all <out name>` starts itself once per
// scenario with `run <scenario>`, its stdout and stderr sent to files under
// `%TEMP%\llamazig`, and writes the same transcript there, every scenario
// in order, for `harness/win_lines_diff.c` to compare.
//
// Usage: win_host_arms list | run <scenario> | all <out name>

#define main host_arms_main
#include "host_arms.c"
#undef main

#include <direct.h>

static void temp_path(char * buf, size_t len, const char * name) {
    const char * tmp = getenv("TEMP");
    snprintf(buf, len, "%s\\llamazig\\%s", tmp ? tmp : ".", name);
}

// Reads a whole file into a NUL-terminated buffer the caller frees.
static char * slurp(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    size_t cap = 4096, len = 0;
    char * buf = malloc(cap);
    size_t n;
    while ((n = fread(buf + len, 1, cap - len - 1, f)) > 0) {
        len += n;
        if (cap - len < 2) buf = realloc(buf, cap *= 2);
    }
    fclose(f);
    buf[len] = '\0';
    return buf;
}

// scripts/host-arms-diff's `normalise`: "failed to set affinity mask 0x…"
// prints the mask's address, not the mask.
static void put_normalised(FILE * out, const char * line, size_t n) {
    static const char key[] = "failed to set affinity mask 0x";
    const char * hit = NULL;
    for (size_t i = 0; i + sizeof(key) - 1 <= n; i++) {
        if (memcmp(line + i, key, sizeof(key) - 1) == 0) { hit = line + i; break; }
    }
    if (!hit) { fwrite(line, 1, n, out); return; }
    const char * hex = hit + sizeof(key) - 1;
    const char * end = hex;
    while (end < line + n && ((*end >= '0' && *end <= '9') || (*end >= 'a' && *end <= 'f'))) end++;
    fwrite(line, 1, (size_t) (hex - line), out);
    fputs("ADDR", out);
    fwrite(end, 1, (size_t) (line + n - end), out);
}

static int cmp_lines(const void * a, const void * b) {
    return strcmp(*(const char * const *) a, *(const char * const *) b);
}

// Runs `<self> run <scenario>` with its output in two files; -1 if it could
// not start.
static long run_child(const char * self, const char * scenario, const char * out_path, const char * err_path) {
    SECURITY_ATTRIBUTES sa = { sizeof(sa), NULL, TRUE };
    HANDLE o = CreateFileA(out_path, GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    HANDLE e = CreateFileA(err_path, GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (o == INVALID_HANDLE_VALUE || e == INVALID_HANDLE_VALUE) return -1;
    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = o;
    si.hStdError  = e;
    PROCESS_INFORMATION pi;
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "\"%s\" run %s", self, scenario);
    BOOL ok = CreateProcessA(self, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
    CloseHandle(o);
    CloseHandle(e);
    if (!ok) return -1;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (long) code;
}

int main(int argc, char ** argv) {
    if (argc != 3 || strcmp(argv[1], "all") != 0) return host_arms_main(argc, argv);

    char dir[1024], out_path[1024], so[1024], se[1024], self[1024];
    temp_path(dir, sizeof dir, "");
    _mkdir(dir);
    temp_path(out_path, sizeof out_path, argv[2]);
    snprintf(so, sizeof so, "%s.stdout", out_path);
    snprintf(se, sizeof se, "%s.stderr", out_path);
    if (!GetModuleFileNameA(NULL, self, sizeof self)) { printf("FAIL: cannot find this program\n"); return 2; }

    FILE * out = fopen(out_path, "wb");
    if (!out) { printf("FAIL: cannot open %s\n", out_path); return 2; }
    int n = 0;
    for (size_t k = 0; k < sizeof(scenarios) / sizeof(scenarios[0]); k++) {
        const char * s = scenarios[k];
        const long code = run_child(self, s, so, se);
        if (code == -1) { printf("FAIL: could not run %s\n", s); return 2; }

        char * text = slurp(so);
        for (char * line = text; line && *line;) {
            size_t len = strcspn(line, "\n");
            size_t keep = len > 0 && line[len - 1] == '\r' ? len - 1 : len;
            put_normalised(out, line, keep);
            fputc('\n', out);
            line += len + (line[len] == '\n');
        }
        free(text);
        fprintf(out, "exit=%ld\n-- stderr, sorted\n", code);

        char * err = slurp(se);
        char * lines[4096];
        int nl = 0;
        for (char * line = err; line && *line && nl < 4096;) {
            size_t len = strcspn(line, "\n");
            const int nlc = line[len] == '\n';
            if (len > 0 && line[len - 1] == '\r') line[len - 1] = '\0';
            line[len] = '\0';
            lines[nl++] = line;
            line += len + nlc;
        }
        qsort(lines, (size_t) nl, sizeof(lines[0]), cmp_lines);
        for (int i = 0; i < nl; i++) {
            put_normalised(out, lines[i], strlen(lines[i]));
            fputc('\n', out);
        }
        free(err);
        n++;
    }
    fclose(out);
    printf("host_arms: %d scenarios -> %s\n", n, argv[2]);
    return 0;
}
