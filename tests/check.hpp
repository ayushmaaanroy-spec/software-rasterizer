// A ~40 line test harness, so the project stays dependency-free.
#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace test {

using TestFn = void (*)();

struct Case {
    std::string name;
    TestFn fn;
};

std::vector<Case>& registry();

struct Registrar {
    Registrar(const char* name, TestFn fn);
};

void reportFailure(const char* file, int line, const char* expression);
void reportNear(const char* file, int line, const char* expression, double actual, double expected,
                double tolerance);

}  // namespace test

#define TEST(name)                                                  \
    static void name();                                             \
    static const ::test::Registrar registrar_##name(#name, &name);  \
    static void name()

#define CHECK(expr)                                            \
    do {                                                       \
        if (!(expr)) ::test::reportFailure(__FILE__, __LINE__, #expr); \
    } while (false)

#define CHECK_NEAR(actual, expected, tolerance)                                          \
    do {                                                                                 \
        const double a_ = static_cast<double>(actual);                                   \
        const double e_ = static_cast<double>(expected);                                 \
        if (!(std::fabs(a_ - e_) <= (tolerance)))                                        \
            ::test::reportNear(__FILE__, __LINE__, #actual " ~ " #expected, a_, e_,      \
                               static_cast<double>(tolerance));                          \
    } while (false)
