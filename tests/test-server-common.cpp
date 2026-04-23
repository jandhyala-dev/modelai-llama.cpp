#include "../tools/server/server-common.h"
#include "../tools/server/server-chat.h"
#include "testing.h"

#include <fstream>
#include <iostream>

static json make_tool(const std::string & name) {
    return json{
        {"type", "function"},
        {"function", {
            {"name", name},
            {"description", "test tool"},
            {"parameters", {
                {"type", "object"},
                {"properties", json::object()},
                {"required", json::array()},
            }},
        }},
    };
}

static std::string read_template(const std::string & path) {
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static server_chat_params make_chat_params() {
    return {
        /* use_jinja             */ true,
        /* prefill_assistant     */ true,
        /* reasoning_format      */ COMMON_REASONING_FORMAT_NONE,
        /* chat_template_kwargs  */ {},
        /* tmpls                 */ common_chat_templates_init(/* model = */ nullptr, read_template("models/templates/CohereForAI-c4ai-command-r7b-12-2024-tool_use.jinja")),
        /* allow_image           */ false,
        /* allow_audio           */ false,
        /* enable_thinking       */ false,
        /* reasoning_budget      */ -1,
        /* reasoning_budget_msg  */ "",
        /* media_path            */ "",
        /* parallel_tool_calls   */ false,
        /* force_pure_content    */ false,
    };
}

int main(int argc, char ** argv) {
    testing t(std::cout);
    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("convert_anthropic_to_oai normalizes billing header strings", [](testing & t) {
        const std::string system_header =
            "x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;You are Claude Code.";

        json body = {
            {"system", system_header},
            {"messages", json::array({
                {
                    {"role", "user"},
                    {"content", "hi"},
                },
            })},
        };

        json out = server_chat_convert_anthropic_to_oai(body);
        t.assert_equal("system header cch is normalized",
            "x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=fffff;You are Claude Code.",
            out.at("messages").at(0).at("content").get<std::string>());
    });

    t.test("oaicompat_chat_params_parse honors explicit parallel tool calls setting", [](testing & t) {
        json body = {
            {"messages", json::array({
                {
                    {"role", "user"},
                    {"content", "Call the tools."},
                },
            })},
            {"tools", json::array({
                make_tool("first_tool"),
                make_tool("second_tool"),
            })},
        };

        json serial_body = body;
        json parallel_body = body;
        serial_body["parallel_tool_calls"] = false;
        parallel_body["parallel_tool_calls"] = true;

        std::vector<raw_buffer> out_files_serial;
        std::vector<raw_buffer> out_files_parallel;
        auto serial = oaicompat_chat_params_parse(serial_body, make_chat_params(), out_files_serial);
        auto parallel = oaicompat_chat_params_parse(parallel_body, make_chat_params(), out_files_parallel);

        t.assert_true("serial grammar exists", serial.contains("grammar"));
        t.assert_true("parallel grammar exists", parallel.contains("grammar"));
        t.assert_true("explicit parallel tool-calls changes the generated grammar",
            serial.at("grammar").get<std::string>() != parallel.at("grammar").get<std::string>());
    });

    return t.summary();
}
