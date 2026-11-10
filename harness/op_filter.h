/* test-backend-ops' `-o` filter, for the extra cases `cuda_ops_diff.c` and
 * `vk_ops_diff.c` run beside upstream's list: comma-separated entries, each
 * an op name matched exactly or, if it is not a plain `[A-Z0-9_]+` name, a
 * regex searched for in the name (POSIX extended here, ECMAScript there;
 * the same for the filters the gates pass) -- `op_filter_entries` and
 * `op_filter_entry_matches` (test-backend-ops.cpp:1158, 1180 @cd959bae9).
 * An entry naming a full test case, `OP(...)`, matches no extra case.
 *
 * The extras compared the whole `-o` argument with `strcmp` until v0.8.0, so
 * any list or regex -- `-o 'MUL_MAT,MUL_MAT_ID'`, `-o 'RMS_NORM.*'` -- ran
 * upstream's cases alone and none of the extras written for them. */
#ifndef LLAMAZIG_OP_FILTER_H
#define LLAMAZIG_OP_FILTER_H

#include <regex.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static bool op_filter_entry_matches(const char * entry, size_t n, const char * op) {
    if (strlen(op) == n && strncmp(entry, op, n) == 0) {
        return true;
    }
    bool plain = n > 0;
    for (size_t i = 0; i < n; i++) {
        const char ch = entry[i];
        plain = plain && ((ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_');
    }
    if (plain) {
        return false;
    }
    char pattern[512];
    if (n + 1 > sizeof(pattern)) {
        return false;
    }
    snprintf(pattern, sizeof(pattern), "%.*s", (int) n, entry);
    regex_t re;
    if (regcomp(&re, pattern, REG_EXTENDED | REG_NOSUB) != 0) {
        return false;
    }
    const bool hit = regexec(&re, op, 0, NULL, 0) == 0;
    regfree(&re);
    return hit;
}

/* Whether the extra case for `op` runs under `-o filter`; NULL runs all. */
static bool op_filter_matches(const char * filter, const char * op) {
    if (filter == NULL) {
        return true;
    }
    const char * p = filter;
    while (*p) {
        const char * comma = strchr(p, ',');
        const char * lparen = strchr(p, '(');
        if (lparen != NULL && (comma == NULL || lparen < comma)) {
            const char * rparen = strchr(lparen, ')');
            comma = rparen ? strchr(rparen, ',') : NULL;
            /* a full test case string names an upstream case, not an extra */
        } else if (op_filter_entry_matches(p, comma ? (size_t) (comma - p) : strlen(p), op)) {
            return true;
        }
        if (comma == NULL) {
            break;
        }
        p = comma + 1;
    }
    return false;
}

#endif
