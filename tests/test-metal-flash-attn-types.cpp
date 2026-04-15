#include "../ggml/src/ggml-metal/ggml-metal-flash-attn.h"
#include "testing.h"

#include <iostream>

static void test_supported_same_type_pairs(testing & t) {
    t.assert_true("same q4_0 pair stays supported",
        ggml_metal_flash_attn_ext_supported_type_pair(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false));
    t.assert_true("same q8_0 pair stays supported",
        ggml_metal_flash_attn_ext_supported_type_pair(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false));
    t.assert_true("bf16 pair requires bfloat support",
        ggml_metal_flash_attn_ext_supported_type_pair(GGML_TYPE_BF16, GGML_TYPE_BF16, true));
    t.assert_true("bf16 pair is rejected without bfloat support",
        !ggml_metal_flash_attn_ext_supported_type_pair(GGML_TYPE_BF16, GGML_TYPE_BF16, false));
}

static void test_supported_mixed_quant_pairs(testing & t) {
    t.assert_true("q8_0 K with q4_0 V is supported",
        ggml_metal_flash_attn_ext_supported_type_pair(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, false));
    t.assert_true("q4_0 K with q8_0 V is supported",
        ggml_metal_flash_attn_ext_supported_type_pair(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, false));
    t.assert_true("q8_0 K with q5_1 V is supported",
        ggml_metal_flash_attn_ext_supported_type_pair(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, false));
    t.assert_true("q5_0 K with q8_0 V is supported",
        ggml_metal_flash_attn_ext_supported_type_pair(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, false));
}

static void test_rejected_mixed_pairs(testing & t) {
    t.assert_true("q4_0 K with q5_0 V is still unsupported",
        !ggml_metal_flash_attn_ext_supported_type_pair(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, false));
    t.assert_true("mixed quant and fp16 remains unsupported in this slice",
        !ggml_metal_flash_attn_ext_supported_type_pair(GGML_TYPE_Q8_0, GGML_TYPE_F16, false));
    t.assert_true("unsupported types remain rejected",
        !ggml_metal_flash_attn_ext_supported_type_pair(GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, false));
}

int main(int argc, char ** argv) {
    testing t(std::cout);
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("supported_same_type_pairs", test_supported_same_type_pairs);
    t.test("supported_mixed_quant_pairs", test_supported_mixed_quant_pairs);
    t.test("rejected_mixed_pairs", test_rejected_mixed_pairs);
    return t.summary();
}
