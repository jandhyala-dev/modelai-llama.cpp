#include "../src/unicode.h"
#include "testing.h"

#include <iostream>
#include <string>

static void test_qwen35_combining_marks(testing & t) {
    const std::string regex_qwen35 =
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    const std::string text = "Cafe\xCC\x81 12";
    const auto tokens = unicode_regex_split(text, { regex_qwen35 }, false);

    t.assert_equal("token count", 4u, tokens.size());
    if (tokens.size() == 4u) {
        t.assert_equal("combining mark stays attached to the word", std::string("Cafe\xCC\x81"), tokens[0]);
        t.assert_equal("space token", std::string(" "), tokens[1]);
        t.assert_equal("first digit token", std::string("1"), tokens[2]);
        t.assert_equal("second digit token", std::string("2"), tokens[3]);
    }
}

static void test_qwen35_long_combining_mark_run(testing & t) {
    const std::string regex_qwen35 =
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    std::string text;
    text.reserve(3 * 4096);
    for (int i = 0; i < 4096; ++i) {
        text += "a\xCC\x81";
    }

    const auto tokens = unicode_regex_split(text, { regex_qwen35 }, false);
    t.assert_equal("long combining-mark run stays in one token", 1u, tokens.size());
    if (!tokens.empty()) {
        t.assert_equal("token length matches input length", text.size(), tokens[0].size());
    }
}

int main(int argc, char ** argv) {
    testing t(std::cout);
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("qwen35_combining_marks", test_qwen35_combining_marks);
    t.test("qwen35_long_combining_mark_run", test_qwen35_long_combining_mark_run);
    return t.summary();
}
