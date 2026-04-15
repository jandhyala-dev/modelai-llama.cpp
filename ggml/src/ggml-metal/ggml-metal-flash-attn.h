#pragma once

#include "ggml.h"

static inline bool ggml_metal_flash_attn_ext_supported_type(enum ggml_type type, bool has_bfloat) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
            return true;
        case GGML_TYPE_BF16:
            return has_bfloat;
        default:
            return false;
    }
}

static inline bool ggml_metal_flash_attn_ext_supported_type_pair(
        enum ggml_type type_k,
        enum ggml_type type_v,
        bool has_bfloat) {
    if (!ggml_metal_flash_attn_ext_supported_type(type_k, has_bfloat) ||
        !ggml_metal_flash_attn_ext_supported_type(type_v, has_bfloat)) {
        return false;
    }

    if (type_k == type_v) {
        return true;
    }

    // Support the practical mixed quant pairs used to keep K at q8_0 while shrinking V more aggressively.
    return
        (type_k == GGML_TYPE_Q8_0 && (type_v == GGML_TYPE_Q4_0 || type_v == GGML_TYPE_Q4_1 ||
                                      type_v == GGML_TYPE_Q5_0 || type_v == GGML_TYPE_Q5_1)) ||
        (type_v == GGML_TYPE_Q8_0 && (type_k == GGML_TYPE_Q4_0 || type_k == GGML_TYPE_Q4_1 ||
                                      type_k == GGML_TYPE_Q5_0 || type_k == GGML_TYPE_Q5_1));
}
