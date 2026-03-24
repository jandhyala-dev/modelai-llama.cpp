#pragma once

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace llama_kv_compact_test {

inline int fail(const std::string & prefix, const std::string & message) {
    std::fprintf(stderr, "%s: %s\n", prefix.c_str(), message.c_str());
    return 1;
}

inline bool check(const std::string & prefix, bool cond, const std::string & message, int & rc) {
    if (!cond) {
        std::fprintf(stderr, "%s FAIL: %s\n", prefix.c_str(), message.c_str());
        rc = 1;
        return false;
    }
    std::printf("  PASS: %s\n", message.c_str());
    return true;
}

inline float cosine_similarity(const std::vector<float> & a, const std::vector<float> & b) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12f);
}

} // namespace llama_kv_compact_test
