// A C++ program that cannot be linked on this machine, because linking pulls in
// Zig's bundled libc++, which is what fails to build.
//
//   zig c++ -c hello.cpp -o /dev/null   # compiles fine
//   zig c++    hello.cpp -o hello       # error: sub-compilation of libcxx failed
#include <cstdio>
#include <string>

int main() {
    std::string s = "ok";
    std::printf("%s\n", s.c_str());
    return 0;
}
