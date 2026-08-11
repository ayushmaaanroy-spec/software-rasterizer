#include "check.hpp"

#include <cstdio>

namespace test {
namespace {
int gFailures = 0;
int gCurrentCaseFailures = 0;
}  // namespace

std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

Registrar::Registrar(const char* name, TestFn fn) { registry().push_back({name, fn}); }

void reportFailure(const char* file, int line, const char* expression) {
    ++gFailures;
    ++gCurrentCaseFailures;
    std::printf("    FAIL %s:%d  %s\n", file, line, expression);
}

void reportNear(const char* file, int line, const char* expression, double actual, double expected,
                double tolerance) {
    ++gFailures;
    ++gCurrentCaseFailures;
    std::printf("    FAIL %s:%d  %s  (got %.9g, want %.9g, tolerance %.3g)\n", file, line,
                expression, actual, expected, tolerance);
}

}  // namespace test

int main() {
    int failedCases = 0;
    for (const test::Case& testCase : test::registry()) {
        test::gCurrentCaseFailures = 0;
        testCase.fn();
        const bool passed = test::gCurrentCaseFailures == 0;
        if (!passed) ++failedCases;
        std::printf("  %-4s %s\n", passed ? "ok" : "FAIL", testCase.name.c_str());
    }

    std::printf("\n%zu tests, %d failed\n", test::registry().size(), failedCases);
    return failedCases == 0 ? 0 : 1;
}
