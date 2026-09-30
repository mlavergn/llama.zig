# Zig 0.16.0 cannot link C++ against the macOS 27 SDK.
#
# Zig builds its bundled libc++ with -std=c++23. That turns clang's `modules`
# feature on. SDK 27's <math.h> then declines to define INFINITY, deferring to
# <float.h> where C23 moved it -- but libc++'s own
# __random/clamp_to_integral.h:47 uses INFINITY without including <float.h>.
#
# Nothing below is specific to any project. Recipes are not silenced, so the
# command that shows the problem is the line make prints.
#
#   make cxx17    passes -- modules off, INFINITY defined
#   make cxx20    FAILS  -- modules on,  INFINITY withheld
#   make link     FAILS  -- the same error, reached by linking libc++
#   make sdk      the guard in the SDK header
#   make flags    the -std Zig compiles its libc++ with
#   make modules  what __has_feature(modules) is per -std
#   make c17      proof that -std=c++17 builds libc++ clean (Zig hardcodes c++23)
#   make apple    proof that Apple's own C++ stack links fine through zig

.PHONY: all cxx17 cxx20 link sdk flags modules c17 apple clean

SDK := $(shell xcrun --show-sdk-path)

# The failing cases are prefixed with `-` so make reports them and carries on.
all: sdk flags modules cxx17
	-$(MAKE) cxx20
	-$(MAKE) link

# The whole bug: the same file, two -std values.
cxx17:
	zig c++ -c -std=c++17 minimal.cpp -o minimal.o

cxx20:
	zig c++ -c -std=c++20 minimal.cpp -o minimal.o

# What a real build hits: linking pulls in libc++, which is what fails to build.
# Filtered to the error: libc++ also emits ~3000 nullability warnings, because
# Zig compiles it with -D_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER.
link:
	zig c++ hello.cpp -o hello 2>&1 | grep 'error:' | sort -u

# SDK 26 defined INFINITY unconditionally. SDK 27 added this.
sdk:
	grep -B2 -A1 'define INFINITY' $(SDK)/usr/include/math.h

# Zig compiles libcxx/src/random.cpp with -std=c++23.
flags:
	printf 'test "x" {}\n' > tiny.zig
	zig test --verbose-cc -lc -lc++ -Mroot=tiny.zig 2>&1 | grep -m1 random.cpp | tr ' ' '\n' | grep -e -std= -e -nostdinc | sort -u

# -std=c++20 and later turn the modules feature on, which is what trips the guard.
modules:
	printf '#if __has_feature(modules)\n#error MODULES ON\n#endif\nint x;\n' > modules.c
	-zig c++ -c -std=c++17 -x c++ modules.c -o modules.o
	-zig c++ -c -std=c++20 -x c++ modules.c -o modules.o

# Zig's own libc++ source builds clean at c++17 -- the -std is the whole
# problem. Proven by replaying Zig's *actual* compile command for
# libcxx/src/random.cpp with only -std changed. Zig hardcodes c++23 for it
# inside the compiler binary; no flag, env var or build.zig option reaches it.
c17:
	printf 'test "x" {}\n' > tiny.zig
	zig test --verbose-cc -lc -lc++ -Mroot=tiny.zig 2>&1 | grep -m1 -i 'random\.cpp' > cmd.txt
	grep -o -- '-std=[^ ]*' cmd.txt
	sh cmd.txt 2>&1 | grep -c 'error:' || true
	sed 's/-std=c++23/-std=c++17/' cmd.txt | sh 2>&1 | grep -c 'error:' || true

# The usable workaround: skip Zig's libc++ entirely and use Apple's headers and
# dylib, which is what Apple clang does. Note -I, not -isystem: the C++ headers
# must come before the C ones.
apple:
	zig c++ -nostdinc++ -I$(SDK)/usr/include/c++/v1 -nostdlib++ hello.cpp -o hello $(SDK)/usr/lib/libc++.tbd
	./hello

clean:
	rm -f minimal.o modules.o modules.c random.o hello tiny.zig cmd.txt
