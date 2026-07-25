# Contributors

The project differentiates between 3 levels of contributors:

- Contributors: people who have contributed before (no special privileges)
- Collaborators (Triage): people with significant contributions, who may be responsible for some parts of the code, and are expected to maintain and review contributions for the code they own
- Maintainers: responsible for reviewing and merging PRs, after approval from the code owners

# AI Usage Policy

> [!IMPORTANT]
>
> AI-generated code is allowed. You are 100% responsible for every line, however it was produced.
>
> Undisclosed AI usage may result in your account being permanently banned from contributing to the project.
>
> Detailed information regarding permissible and restricted uses of AI can be found in the [AGENTS.md](AGENTS.md) file.

If AI is used to generate any portion of the code, contributors must adhere to the following requirements:

1. Explicitly disclose the manner in which AI was employed.
2. Check for an existing PR addressing the same change; if one exists, comment there to work with its author instead of opening a duplicate.
3. Perform a comprehensive manual review prior to submitting the pull request.
4. Be prepared to explain every line of code they submitted when asked about it by a maintainer.
5. It is strictly prohibited to use AI to write your posts for you (bug reports, feature requests, pull request descriptions, Github discussions, responding to humans, ...).

For more info, please refer to the [AGENTS.md](AGENTS.md) file.

# Pull requests (for contributors & collaborators)

### Before you start

- Search for existing discussions and PRs first - duplicates will likely be closed without questions.
- Features must begin with an issue, not a PR - let interest accumulate before writing code; niche features may only land as an example/tool, or on a private fork.
- Bug-fix PRs must include a reproducible issue and a regression test that fails before your change and passes after. Fixes without a test may be closed without review.
- New CLI or public API additions carry a **higher bar** than internal changes - justify why an existing mechanism doesn't suffice.
- Meeting all of the above still doesn't guarantee a merge - see [Pull requests (for maintainers)](#pull-requests-for-maintainers).
- If you are a new contributor
    - Limit your open PRs to 1
    - Do not submit trivial fixes (e.g. typos, formatting changes)

### Preparing your PR

- llama.cpp uses the ggml tensor library for model evaluation. If you are unfamiliar with ggml, consider taking a look at the [examples in the ggml repository](https://github.com/ggml-org/ggml/tree/master/examples/). [simple](https://github.com/ggml-org/ggml/tree/master/examples/simple) shows the bare minimum for using ggml. [gpt-2](https://github.com/ggml-org/ggml/tree/master/examples/gpt-2) has minimal implementations for language model inference using GPT-2. [mnist](https://github.com/ggml-org/ggml/tree/master/examples/mnist) demonstrates how to train and evaluate a simple image classifier
- Test your changes:
  - Execute [the full CI locally on your machine](ci/README.md) before publishing
  - Verify that the perplexity and the performance are not affected negatively by your changes (use `llama-perplexity` and `llama-bench`)
  - If you modified the `ggml` source, run the `test-backend-ops` tool to check whether different backend implementations of the `ggml` operators produce consistent results (this requires access to at least two different `ggml` backends)
  - If you modified a `ggml` operator or added a new one, add the corresponding test cases to `test-backend-ops`
- Create separate PRs for each feature or fix:
  - Avoid combining unrelated changes in a single PR
  - When adding support for a new model or feature, focus on **CPU support only** in the initial PR unless you have a good reason not to. Add support for other backends like CUDA in follow-up PRs
  - In particular, adding new data types (extension of the `ggml_type` enum) carries with it a disproportionate maintenance burden. As such, to add a new quantization type you will need to meet the following *additional* criteria *at minimum*:
    - convert a small model to GGUF using the new type and upload it to HuggingFace
    - provide [perplexity](https://github.com/ggml-org/llama.cpp/tree/master/tools/perplexity) comparisons to FP16/BF16 (whichever is the native precision) as well as to types of similar size
    - provide KL divergence data calculated vs. the FP16/BF16 (whichever is the native precision) version for both the new type as well as types of similar size
    - provide [performance data](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) for the new type in comparison to types of similar size on pure CPU
- Consider allowing write access to your branch for faster reviews, as reviewers can push commits directly

### After submitting your PR

- Expect requests for modifications to ensure the code meets llama.cpp's standards for quality and long-term maintainability
- Maintainers will rely on your insights and approval when making a final decision to approve and merge a PR
- If your PR becomes stale, rebase it on top of latest `master` to get maintainers attention
- Consider adding yourself to [CODEOWNERS](CODEOWNERS) to indicate your availability for fixing related issues and reviewing related PRs

# Pull requests (for maintainers)

- Squash-merge PRs
- Use the following format for the squashed commit title: `<module> : <commit title> (#<issue_number>)`. For example: `utils : fix typo in utils.py (#1234)`
- Optionally pick a `<module>` from here: https://github.com/ggml-org/llama.cpp/wiki/Modules
- Let other maintainers merge their own PRs
- When merging a PR, make sure you have a good understanding of the changes
- If a PR does not warrant a new release, add `[no release]` in the squashed commit to spare CI resources
- Be mindful of maintenance: most of the work going into a feature happens after the PR is merged. If the PR author is not committed to contribute long-term, someone else needs to take responsibility (you)

Maintainers reserve the right to decline review or close pull requests for any reason, without any questions, particularly under any of the following conditions:
- The proposed change is already mentioned in the roadmap or an existing issue, and it has been assigned to someone.
- The pull request duplicates an existing one.
- The contributor fails to adhere to this contributing guide or the AI policy.
- The change doesn't fit the existing architecture, or is too complex to justify its benefit.

# Coding guidelines

- Avoid adding third-party dependencies, extra files, extra headers, etc.
- Always consider cross-compatibility with other operating systems and architectures
- Avoid fancy-looking modern STL constructs, use basic `for` loops, avoid templates, keep it simple
- Vertical alignment makes things more readable and easier to batch edit
- Clean-up any trailing whitespaces, use 4 spaces for indentation, brackets on the same line, `void * ptr`, `int & a`
- Use sized integer types such as `int32_t` in the public API, e.g. `size_t` may also be appropriate for allocation sizes or byte offsets
- Declare structs with `struct foo {}` instead of `typedef struct foo {} foo`
    - In C++ code omit optional `struct` and `enum` keyword whenever they are not necessary
    ```cpp
    // OK
    llama_context * ctx;
    const llama_rope_type rope_type;

    // not OK
    struct llama_context * ctx;
    const enum llama_rope_type rope_type;
    ```

    _(NOTE: this guideline is yet to be applied to the `llama.cpp` codebase. New code should follow this guideline.)_

- Try to follow the existing patterns in the code (indentation, spaces, etc.). In case of doubt use `clang-format` (from clang-tools v15+) to format the added code
- For anything not covered in the current guidelines, refer to the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- Tensors store data in row-major order. We refer to dimension 0 as columns, 1 as rows, 2 as matrices
- Matrix multiplication is unconventional: [`C = ggml_mul_mat(ctx, A, B)`](https://github.com/ggml-org/llama.cpp/blob/880e352277fc017df4d5794f0c21c44e1eae2b84/ggml.h#L1058-L1064) means $C^T = A B^T \Leftrightarrow C = B A^T.$

![matmul](media/matmul.png)

# Naming guidelines

- Use `snake_case` for function, variable and type names
- Naming usually optimizes for longest common prefix (see https://github.com/ggml-org/ggml/pull/302#discussion_r1243240963)

    ```cpp
    // not OK
    int small_number;
    int big_number;

    // OK
    int number_small;
    int number_big;
    ```

- Enum values are always in upper case and prefixed with the enum name

    ```cpp
    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE = 0,
        LLAMA_VOCAB_TYPE_SPM  = 1,
        LLAMA_VOCAB_TYPE_BPE  = 2,
        LLAMA_VOCAB_TYPE_WPM  = 3,
        LLAMA_VOCAB_TYPE_UGM  = 4,
        LLAMA_VOCAB_TYPE_RWKV = 5,
    };
    ```

- The general naming pattern is `<class>_<method>`, with `<method>` being `<action>_<noun>`

    ```cpp
    llama_model_init();           // class: "llama_model",         method: "init"
    llama_sampler_chain_remove(); // class: "llama_sampler_chain", method: "remove"
    llama_sampler_get_seed();     // class: "llama_sampler",       method: "get_seed"
    llama_set_embeddings();       // class: "llama_context",       method: "set_embeddings"
    llama_n_threads();            // class: "llama_context",       method: "n_threads"
    llama_adapter_lora_free();    // class: "llama_adapter_lora",  method: "free"
    ```

    - The `get` `<action>` can be omitted
    - The `<noun>` can be omitted if not necessary
    - The `_context` suffix of the `<class>` is optional. Use it to disambiguate symbols when needed
    - Use `init`/`free` for constructor/destructor `<action>`

- Use the `_t` suffix when a type is supposed to be opaque to the user - it's not relevant to them if it is a struct or anything else

    ```cpp
    typedef struct llama_context * llama_context_t;

    enum llama_pooling_type llama_pooling_type(const llama_context_t ctx);
    ```

    _(NOTE: this guideline is yet to be applied to the `llama.cpp` codebase. New code should follow this guideline)_

- C/C++ filenames are all lowercase with dashes. Headers use the `.h` extension. Source files use the `.c` or `.cpp` extension
- Python filenames are all lowercase with underscores

- _(TODO: abbreviations usage)_

# Preprocessor directives

- _(TODO: add guidelines with examples and apply them to the codebase)_

    ```cpp
    #ifdef FOO
    #endif // FOO
    ```

# Code maintenance

- Existing code should have designated collaborators and/or maintainers specified in the [CODEOWNERS](CODEOWNERS) file responsible for:
  - Reviewing and merging related PRs
  - Fixing related bugs
  - Providing developer guidance/support

- When adding or modifying a large piece of code:
  - If you are a collaborator, make sure to add yourself to [CODEOWNERS](CODEOWNERS) to indicate your availability for reviewing related PRs
  - If you are a contributor, find an existing collaborator who is willing to review and maintain your code long-term
  - Provide the necessary CI workflow (and hardware) to test your changes (see [ci/README.md](https://github.com/ggml-org/llama.cpp/tree/master/ci))

- New code should follow the guidelines (coding, naming, etc.) outlined in this document. Exceptions are allowed in isolated, backend-specific parts of the code that do not interface directly with the `ggml` interfaces.
  _(NOTE: for legacy reasons, existing code is not required to follow this guideline)_

- For changes in server, please make sure to refer to the [server development documentation](./tools/server/README-dev.md)

# Documentation

- Documentation is a community effort
- When you need to look into the source code to figure out how to use an API consider adding a short summary to the header file for future reference
- When you notice incorrect or outdated documentation, please update it

# Resources

The Github issues, PRs and discussions contain a lot of information that can be useful to get familiar with the codebase. For convenience, some of the more important information is referenced from Github projects:

https://github.com/ggml-org/llama.cpp/projects

---

# modelai-llama.cpp Fork: Additional Guidelines

The sections above are inherited from upstream llama.cpp. The following apply specifically to the KV compaction code in this fork.

## Compaction Code Conventions

- Prefix compaction functions with `llama_kv_compact_`
- All solver math in fp32; cast to model dtype for KV storage only
- Commit messages for compaction work: `kv-compact: <description>`

## Contribution Workflow

Every change follows this process — no exceptions, no shortcuts:

1. **Branch** — Create a feature branch from `modelai-main`
2. **Implement** — Write the code, following the conventions above
3. **Document** — Update the relevant `docs/` file (algorithm, integration, bugs, changelog)
4. **Test** — `cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j$(sysctl -n hw.ncpu) && ctest --test-dir build -L main --output-on-failure` — all tests must pass
5. **Review** — Submit for adversarial review using the 13-section protocol in [AGENTS.md](AGENTS.md). The reviewer's job is to break the code, not confirm it looks reasonable
6. **Fix all findings** — Every Critical/Major finding must be resolved. No deferring fixable issues
7. **Re-test** — Run the full test suite again after fixes
8. **Commit** — Descriptive message with `kv-compact:` prefix for compaction work. Reference the review verdict
9. **Merge to `modelai-main`** — Only after review PASS with all findings addressed
10. **Push** — Verify the commit SHA matches what was reviewed

If a review finds bugs, the cycle repeats from step 6. Code that hasn't passed adversarial review does not enter `modelai-main`.

## Adversarial Review Protocol

See [AGENTS.md](AGENTS.md) for the full 13-section review protocol. Key points:

- Reviewers walk **real numbers** through your code (production, boundary, and adversarial traces)
- Every assumption must be either **enforced in code** or **documented** — undocumented assumptions are findings
- The **disprove-it pass** is mandatory: the reviewer assumes one bug exists and tries to find it before issuing PASS
- Critical/Major findings block merge; Minor findings are noted for follow-up

## Code Health and Governance

### Goal

Every change to this repository is evaluated across 8 dimensions before merge:

1. **Correctness** — Does the code produce the right results?
2. **Performance** — No regression to baseline decode speed
3. **Quality** — Compaction cosine similarity meets the 0.95 gate
4. **Safety** — Crash protection, NaN guards, memory bounds
5. **Security** — No PII, no leaked credentials, no injection vectors
6. **Documentation** — Changes reflected in the relevant `docs/` file
7. **Test coverage** — New behavior has tests; existing tests still pass
8. **OSS standards** — Compared against top open-source projects (vLLM, PyTorch, Transformers, llama.cpp upstream)

The goal is to fix ALL findings before OSS release. No finding is deferred if it is fixable in the current cycle.

### PR Review Standard

Every pull request — whether it adds KV cache compaction features, upstream sync tooling, governance workflows, or documentation — is held to the same standard:

- **Adversarial review** using the 13-section protocol in [AGENTS.md](AGENTS.md)
- **All findings assessed for fixability** with effort estimates. If fixable, fix it before merge.
- **Accepted risks documented explicitly** — if a finding cannot be fixed (e.g., upstream bug, platform limitation), it is logged with rationale in the PR description
- **Open PRs and issues assessed** — before merging, check for interactions with other open PRs or known issues

### Weekly Code Health Check (Saturdays)

Run these checks weekly to maintain repository hygiene:

1. **PII scan** — `grep -r "/Users/" docs/ src/ tools/ AGENTS.md CONTRIBUTING.md` and `grep -ri "model-ai\.app\|c-o-t\.ai\|COT Labs" docs/` — zero hits expected
2. **Dead test check** — `ctest --test-dir build -N -L main | tail -1` — verify test count matches expectations
3. **Duplication check** — review any repeated code patterns introduced in the past week
4. **PR triage** — review all open PRs: merge, request changes, or close stale ones
5. **Upstream watch** — review the `upstream-watch` GitHub Issue (auto-created by `modelai-upstream-watch.yml` if KV cache activity detected)

### Quarterly Deep Review

Every 3 months, conduct a full adversarial review of the entire compaction codebase:

1. Re-run all benchmarks (decode speed, compaction quality, 256K context extension)
2. Re-verify all 29 bug fixes are still valid (no regressions)
3. Audit the full `src/llama-kv-compact-*` surface against the latest upstream KV cache API
4. Compare documentation against current code — flag any drift
5. Review OSS governance against top projects — update practices if the ecosystem has moved

Results are logged as a GitHub Issue tagged `quarterly-review`.

## Fork Documentation

- `docs/kv-compaction-algorithm.md` — Algorithm overview
- `docs/kv-compaction-integration.md` — File map and architecture support matrix
- `docs/BUGS-AND-FIXES.md` — All bugs found, fixed, and tracked
- `docs/UPSTREAM-SYNC.md` — Upstream sync process and verification
- `docs/benchmark-fork-vs-upstream.md` — Performance comparison
