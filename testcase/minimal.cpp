// The whole bug, in six lines. No Zig code, no llamazig, no libc++ sources.
//
//   zig c++ -c -std=c++17 minimal.cpp -o /dev/null   # compiles
//   zig c++ -c -std=c++20 minimal.cpp -o /dev/null   # error: INFINITY undeclared
//
// macOS SDK 27's <math.h> withholds INFINITY when __has_feature(modules) is on,
// deferring to <float.h> where C23 moved it. Clang turns modules on from
// -std=c++20. Zig builds its bundled libc++ with -std=c++23, and libc++'s
// __random/clamp_to_integral.h uses INFINITY without including <float.h>.
#include <math.h>

float f() { return INFINITY; }
