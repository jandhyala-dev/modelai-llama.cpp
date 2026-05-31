#include "src/llama-kv-compacted-prefix.h"
#include "src/llama-kv-compacted-prefix-exec.h"
#include "src/llama-io.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int fail(const std::string & message) {
    std::cerr << "test-kv-compact-imrope: " << message << std::endl;
    return 1;
}

bool check(bool cond, const std::string & message, int & rc) {
    if (!cond) {
        rc = fail(message);
        return false;
    }
    return true;
}

// Minimal IO helpers (same pattern as test-kv-compacted-prefix.cpp)
class test_io_write_buffer : public llama_io_write_i {
public:
    void write(const void * src, size_t size) override {
        const auto * bytes = reinterpret_cast<const uint8_t *>(src);
        buf.insert(buf.end(), bytes, bytes + size);
    }

    [[noreturn]] void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t /* size */) override {
        throw std::runtime_error("tensor writes are not used in IMROPE tests");
    }

    size_t n_bytes() override {
        return buf.size();
    }

    std::vector<uint8_t> buf;
};

class test_io_read_buffer : public llama_io_read_i {
public:
    explicit test_io_read_buffer(const std::vector<uint8_t> & src) : buf(src) {}

    void read(void * dst, size_t size) override {
        if (off + size > buf.size()) {
            throw std::runtime_error("unexpected end of IMROPE test buffer");
        }
        std::memcpy(dst, buf.data() + off, size);
        off += size;
    }

    [[noreturn]] void read_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t /* size */) override {
        throw std::runtime_error("tensor reads are not used in IMROPE tests");
    }

    size_t n_bytes() override {
        return off;
    }

private:
    const std::vector<uint8_t> & buf;
    size_t off = 0;
};

// Helper to create a simple ubatch for can_execute tests.
// When n_pos >= 3 (i.e. mrope dimensions), is_pos_2d() returns true.
struct test_ubatch_helper {
    llama_ubatch ubatch = {};
    std::vector<llama_pos> pos_data;
    std::vector<llama_seq_id> seq_id_unq_data;

    // n_pos: 1 = scalar positions, 3+ = MROPE (is_pos_2d() == true)
    test_ubatch_helper(llama_seq_id seq_id, llama_pos position, uint32_t n_pos_dims = 1) {
        const uint32_t n_tokens = 1;

        // pos array: n_tokens * n_pos_dims elements
        pos_data.resize(n_tokens * n_pos_dims, position);

        seq_id_unq_data = { seq_id };

        ubatch.n_tokens   = n_tokens;
        ubatch.n_seqs_unq = 1;
        ubatch.n_pos      = n_pos_dims;
        ubatch.pos        = pos_data.data();
        ubatch.seq_id_unq = seq_id_unq_data.data();
        // Other fields left as zero/null (not needed for can_execute guard tests)
        ubatch.b_equal_seqs = 1;
    }
};

std::vector<llama_compacted_prefix_layer_layout> make_layouts() {
    return {
        {
            /* layer_id      = */ 0,
            /* n_head_kv     = */ 2,
            /* n_embd_head_k = */ 4,
            /* n_embd_head_v = */ 8,
            /* type_k        = */ GGML_TYPE_F16,
            /* type_v        = */ GGML_TYPE_F16,
        },
    };
}

// ---- Test 1: is_imrope flag stored and retrieved correctly ----
int test_imrope_flag_storage() {
    int rc = 0;

    llama_compacted_prefix_store store(make_layouts());

    const std::vector<llama_pos> positions = { 0, 4, 8 };

    // Configure with is_imrope = false
    if (!check(store.configure_seq(0, 16, positions, 9, /* is_imrope = */ false),
               "configure_seq with is_imrope=false should succeed", rc)) return rc;
    {
        const auto * seq = store.get_seq(0);
        if (!check(seq != nullptr, "sequence 0 should exist", rc)) return rc;
        if (!check(!seq->is_imrope, "is_imrope should be false after configure with false", rc)) return rc;
    }

    // Reconfigure with is_imrope = true
    if (!check(store.configure_seq(0, 16, positions, 9, /* is_imrope = */ true),
               "configure_seq with is_imrope=true should succeed", rc)) return rc;
    {
        const auto * seq = store.get_seq(0);
        if (!check(seq != nullptr, "sequence 0 should exist after reconfigure", rc)) return rc;
        if (!check(seq->is_imrope, "is_imrope should be true after configure with true", rc)) return rc;
    }

    // clear resets the flag
    store.clear_seq(0, true);
    {
        const auto * seq = store.get_seq(0);
        // After clear, the sequence is disabled but the state struct still exists
        if (!check(seq != nullptr, "sequence 0 struct should exist after clear", rc)) return rc;
        if (!check(!seq->is_imrope, "is_imrope should be false after clear", rc)) return rc;
    }

    return rc;
}

// ---- Test 2: is_imrope flag preserved through serialization roundtrip ----
int test_imrope_serialization_roundtrip() {
    int rc = 0;

    auto layouts = make_layouts();
    llama_compacted_prefix_store store(layouts);

    const std::vector<llama_pos> positions = { 0, 4, 8 };

    // Configure with is_imrope = true
    if (!check(store.configure_seq(2, 16, positions, 9, /* is_imrope = */ true),
               "configure_seq for serialization test should succeed", rc)) return rc;
    if (!check(store.set_execution(2, true), "execution should enable", rc)) return rc;

    // Serialize
    test_io_write_buffer writer;
    store.state_write(writer, 2);

    // Deserialize into a fresh store, targeting a different seq_id
    llama_compacted_prefix_store restored(layouts);
    test_io_read_buffer reader(writer.buf);
    if (!check(restored.state_read(reader, 5), "state_read should succeed", rc)) return rc;

    const auto * seq5 = restored.get_seq(5);
    if (!check(seq5 != nullptr, "restored sequence 5 should exist", rc)) return rc;
    if (!check(seq5->is_imrope, "restored is_imrope should be true", rc)) return rc;
    if (!check(seq5->logical_positions == positions, "restored positions should match", rc)) return rc;
    if (!check(restored.execution_enabled(5), "restored execution should be enabled", rc)) return rc;

    // Now test roundtrip with is_imrope = false
    llama_compacted_prefix_store store2(layouts);
    if (!check(store2.configure_seq(1, 8, { 2, 6 }, 7, /* is_imrope = */ false),
               "configure_seq with is_imrope=false for roundtrip should succeed", rc)) return rc;

    test_io_write_buffer writer2;
    store2.state_write(writer2, 1);

    llama_compacted_prefix_store restored2(layouts);
    test_io_read_buffer reader2(writer2.buf);
    if (!check(restored2.state_read(reader2, 3), "state_read for false flag should succeed", rc)) return rc;

    const auto * seq3 = restored2.get_seq(3);
    if (!check(seq3 != nullptr, "restored sequence 3 should exist", rc)) return rc;
    if (!check(!seq3->is_imrope, "restored is_imrope should be false", rc)) return rc;

    return rc;
}

// ---- Test 3: is_imrope flag propagated through seq_cp ----
int test_imrope_seq_cp() {
    int rc = 0;

    llama_compacted_prefix_store store(make_layouts());

    const std::vector<llama_pos> positions = { 0, 4, 8 };

    // Configure source with is_imrope = true
    if (!check(store.configure_seq(0, 16, positions, 9, /* is_imrope = */ true),
               "configure_seq source with is_imrope=true should succeed", rc)) return rc;

    // Copy to another sequence
    store.seq_cp(0, 1, -1, -1);

    const auto * seq1 = store.get_seq(1);
    if (!check(seq1 != nullptr, "copied sequence 1 should exist", rc)) return rc;
    if (!check(seq1->is_imrope, "copied sequence should preserve is_imrope=true", rc)) return rc;

    // Verify partial copy also preserves the flag
    store.seq_cp(0, 2, 0, 5);

    const auto * seq2 = store.get_seq(2);
    if (!check(seq2 != nullptr, "partial-copied sequence 2 should exist", rc)) return rc;
    if (!check(seq2->is_imrope, "partial-copied sequence should preserve is_imrope=true", rc)) return rc;

    return rc;
}

// ---- Test 4: can_execute guard for is_pos_2d batches ----
int test_imrope_can_execute_guard() {
    int rc = 0;

    llama_compacted_prefix_store store(make_layouts());

    const std::vector<llama_pos> positions = { 0, 4, 8 };

    // ---- Non-IMROPE sequence: 2D positions must be rejected ----
    if (!check(store.configure_seq(0, 16, positions, 9, /* is_imrope = */ false),
               "configure non-IMROPE seq should succeed", rc)) return rc;
    {
        auto & state = *store.get_seq(0);
        state.set_execution_enabled(true);

        // Scalar batch (n_pos=1): should be allowed
        test_ubatch_helper scalar(0, 10, 1);
        if (!check(llama_compacted_prefix_can_execute(0, &state, scalar.ubatch, nullptr),
                   "scalar batch on non-IMROPE seq should be allowed", rc)) return rc;

        // 2D batch (n_pos=3, is_pos_2d()=true): should be rejected for non-IMROPE
        test_ubatch_helper mrope(0, 10, 3);
        if (!check(!llama_compacted_prefix_can_execute(0, &state, mrope.ubatch, nullptr),
                   "2D batch on non-IMROPE seq should be rejected", rc)) return rc;
    }

    // ---- IMROPE sequence: 2D positions must be accepted ----
    if (!check(store.configure_seq(1, 16, positions, 9, /* is_imrope = */ true),
               "configure IMROPE seq should succeed", rc)) return rc;
    {
        auto & state = *store.get_seq(1);
        state.set_execution_enabled(true);

        // Scalar batch: should be allowed
        test_ubatch_helper scalar(1, 10, 1);
        if (!check(llama_compacted_prefix_can_execute(1, &state, scalar.ubatch, nullptr),
                   "scalar batch on IMROPE seq should be allowed", rc)) return rc;

        // 2D batch: should be allowed for IMROPE
        test_ubatch_helper mrope(1, 10, 3);
        if (!check(llama_compacted_prefix_can_execute(1, &state, mrope.ubatch, nullptr),
                   "2D batch on IMROPE seq should be allowed", rc)) return rc;
    }

    return rc;
}

// ---- Test 5: can_execute candidate output populated correctly ----
int test_imrope_can_execute_candidate() {
    int rc = 0;

    llama_compacted_prefix_store store(make_layouts());

    const std::vector<llama_pos> positions = { 0, 4, 8 };

    if (!check(store.configure_seq(0, 16, positions, 9, /* is_imrope = */ true),
               "configure for candidate test should succeed", rc)) return rc;
    {
        auto & state = *store.get_seq(0);
        state.set_execution_enabled(true);

        test_ubatch_helper scalar(0, 10, 1);
        llama_compacted_prefix_exec_candidate candidate = {};
        if (!check(llama_compacted_prefix_can_execute(0, &state, scalar.ubatch, &candidate),
                   "can_execute should succeed for candidate test", rc)) return rc;

        if (!check(candidate.seq_id == 0, "candidate seq_id should be 0", rc)) return rc;
        if (!check(candidate.n_tokens == 3, "candidate n_tokens should match positions count (3)", rc)) return rc;
        if (!check(candidate.zero_beta, "candidate zero_beta should be true (default beta is zero)", rc)) return rc;
    }

    return rc;
}

} // namespace

int main() {
    if (const int rc = test_imrope_flag_storage()) {
        return rc;
    }
    if (const int rc = test_imrope_serialization_roundtrip()) {
        return rc;
    }
    if (const int rc = test_imrope_seq_cp()) {
        return rc;
    }
    if (const int rc = test_imrope_can_execute_guard()) {
        return rc;
    }
    if (const int rc = test_imrope_can_execute_candidate()) {
        return rc;
    }
    std::cout << "test-kv-compact-imrope: all tests passed" << std::endl;
    return 0;
}
