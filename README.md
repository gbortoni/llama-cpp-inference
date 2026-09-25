# llama-cpp-inference

A lightweight C++ command-line inference application built with the llama.cpp API for running local GGUF language models.

The application loads a GGUF model, tokenizes a prompt, creates a llama.cpp runtime context, and performs prompt prefill followed by autoregressive token generation. It supports configurable context capacity, batch capacity, CPU thread count, and stochastic sampling through top-k, top-p, temperature, and explicit random seeds.

The application provides two execution modes: an interactive mode for experimenting with sampling behavior and generated text, and a benchmark mode that fixes sampling parameters to create a controlled and repeatable inference workload. It reports separate prefill, autoregressive decode, and end-to-end throughput measurements.

The initial multi-token batch performs prompt prefill, while subsequent one-token batches perform autoregressive decoding. The application reconstructs the generated text and reports prompt tokens, generated tokens, inference timing, and throughput.

The project also uses RAII-based resource management and includes separate Debug and ASan/UBSan build configurations for debugging and memory-safety validation.

## Features

- Local GGUF model inference through the llama.cpp C API
- Command-line arguments for model path, prompt, and generation length
- Prompt tokenization and vocabulary handling
- Prompt prefill followed by one-token autoregressive decoding
- End-of-generation token handling
- Generated text reconstruction from token pieces
- Prompt and generated token counting
- RAII-based ownership of model, context, and sampler resources
- Separate normal, Debug, and ASan/UBSan build configurations
- LLDB-compatible Debug build for inspecting runtime failures
- Configurable context capacity, batch capacity, and CPU thread count
- Configurable top-k, top-p, temperature, and random seed
- Separate interactive and benchmark execution modes
- Fixed sampling configuration in benchmark mode
- Separate prefill, autoregressive decode, and end-to-end timing
- Warm-up before measured inference
- GPU synchronization for accurate Metal decode timing
- Requested vs. effective runtime configuration reporting

## Build

### Requirements

- C++17-compatible compiler
- CMake 3.16 or newer
- A local llama.cpp checkout
- A compatible GGUF model

This project links against llama.cpp through CMake using the `LLAMA_CPP_DIR` variable instead of hardcoding a machine-specific path.

### Configure and build

From the project root:

```bash
cmake -S . -B build \
  -DLLAMA_CPP_DIR=/path/to/llama.cpp
```

Then build:

```bash
cmake --build build
```

The executable will be created at:

```text
build/llama-cpp-inference
```

### Debug build

A separate Debug build can be created with:

```bash
cmake -S . -B build-debug \
  -DLLAMA_CPP_DIR=/path/to/llama.cpp \
  -DCMAKE_BUILD_TYPE=Debug
```

Then:

```bash
cmake --build build-debug
```

### ASan/UBSan build

Sanitizers can be enabled using:

```bash
cmake -S . -B build-asan \
  -DLLAMA_CPP_DIR=/path/to/llama.cpp \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_SANITIZERS=ON
```

Then:

```bash
cmake --build build-asan
```

The sanitizer build is intended for debugging and correctness checks rather than performance benchmarking.

## Usage

Run the application by providing a GGUF model path and prompt:

```bash
./build/llama-cpp-inference \
  --model /path/to/model.gguf \
  --prompt "Once upon a time" \
  --tokens 10
```

### Command-line arguments

- `--model <path>` — Path to the GGUF model file. Required.
- `--prompt <text>` — Prompt passed to the model. Required.
- `--mode <interactive|benchmark>` — Execution mode. Optional; defaults to `interactive`.
- `--tokens <n>` — Maximum number of tokens to generate. Optional; defaults to `64`.
- `--context <n>` — Requested context capacity. Optional; if omitted, the program requests the capacity required for the workload.
- `--batch <n>` — Requested batch capacity. Optional; if omitted, the prompt token count is used.
- `--threads <n>` — Requested CPU thread count for generation and batch processing. Optional; if omitted, llama.cpp uses its default.
- `--seed <n>` — Random seed used for stochastic sampling. Optional; defaults to `42` in interactive mode.
- `--top-k <n>` — Number of highest-scoring candidate tokens retained before sampling. Optional; defaults to `40`.
- `--top-p <p>` — Cumulative probability threshold used for nucleus sampling. Optional; defaults to `0.9`.
- `--temperature <t>` — Logit temperature applied before stochastic sampling. Optional; defaults to `0.8`.

Sampling options (`--seed`, `--top-k`, `--top-p`, and `--temperature`) may be customized in interactive mode. Benchmark mode uses a fixed sampling configuration and rejects user-provided sampling overrides.

For example:

```bash
./build/llama-cpp-inference \
  --model /path/to/stories15M-q4_0.gguf \
  --prompt "Once upon a time" \
  --tokens 10
```

The application validates required arguments and rejects invalid values such as non-integer or non-positive token, context, batch, or thread counts.

## Example Output

Example interactive generation:

```text
Generated text:
Once upon a time, there was a little girl named Lily.

--- Stats ---
Prompt tokens: 5
Generated tokens: 10
Elapsed time: 0.0236 s
Prefill time: 0.0030 s
Prefill tokens/sec: 1675
Autoregressive decode tokens: 9
Autoregressive decode time: 0.0198 s
Decode tokens/sec: 454
End-to-end generated tokens/sec: 423
```

The exact throughput values vary between runs and systems. The application separates prompt prefill throughput, autoregressive decode throughput, and end-to-end generation throughput so that model execution can be distinguished from broader generation-loop overhead.

## Inference Flow

The application follows this inference pipeline:

```text
CLI arguments
    ↓
Load compute backends
    ↓
Load GGUF model
    ↓
Get model vocabulary
    ↓
Tokenize prompt
    ↓
Create runtime context
    ↓
Create sampling chain
(top-k → top-p → temperature → distribution sampler)
    ↓
Create initial prompt batch
    ↓
llama_decode()
    ↓
Sample next token
    ↓
Check for end-of-generation token
    ↓
Convert token ID to text piece
    ↓
Append generated text
    ↓
Create one-token batch
    ↓
Repeat decode → sample → detokenize
```

The first call to `llama_decode()` processes the complete prompt batch and acts as the **prefill** stage.

After the first token is sampled, each subsequent batch contains only the newly generated token. The runtime context retains information from the previously evaluated sequence, allowing generation to continue autoregressively without resending the entire prompt.

The same generation loop therefore handles both stages:

```text
Initial iteration:

[prompt tokens] → decode → sample G1

Next iteration:

[G1] → decode → sample G2

Next iteration:

[G2] → decode → sample G3

...
```

Generation stops when either:

- The requested number of tokens has been generated, or
- The model produces an end-of-generation token.

## Resource Management

The application uses RAII to manage the lifetime of llama.cpp resources.

The model, context, and sampler are wrapped in `std::unique_ptr` objects with custom deleters:

```cpp
using ModelPtr =
    std::unique_ptr<llama_model, decltype(&llama_model_free)>;

using ContextPtr =
    std::unique_ptr<llama_context, decltype(&llama_free)>;

using SamplerPtr =
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
```

This allows C++ to automatically release the corresponding llama.cpp resources when their owners go out of scope.

For example:

```cpp
ModelPtr model(
    llama_model_load_from_file(model_path.c_str(), model_params),
    llama_model_free
);
```

When a llama.cpp function requires a raw pointer, `.get()` is used:

```cpp
const llama_vocab * vocab =
    llama_model_get_vocab(model.get());

int decode_result =
    llama_decode(ctx.get(), batch);
```

Calling `.get()` does not transfer ownership. It temporarily exposes the underlying raw pointer while the `std::unique_ptr` continues to manage the resource lifetime.

RAII simplifies error handling because early returns no longer require repetitive manual cleanup:

```cpp
if (decode_result != 0) {
    std::cerr << "Decode step failed.\n";
    return 1;
}
```

When execution leaves `main()`, resources are automatically destroyed in reverse construction order:

```text
sampler
   ↓
context
   ↓
model
```

The vocabulary is different: it is a borrowed pointer obtained from the model and is therefore not freed independently by the application.

## Debugging and Sanitizers

The project includes separate build configurations for debugging and sanitizer-based correctness checks.

### LLDB

The Debug build can be inspected using LLDB:

```bash
lldb -- ./build-debug/llama-cpp-inference \
  --model /path/to/model.gguf \
  --prompt "Hello" \
  --tokens 10
```

A real model-loading failure path was inspected by intentionally providing an invalid model path.

A breakpoint was placed inside:

```cpp
if (model == nullptr) {
    std::cerr << "Error: failed to load model.\n";
    return 1;
}
```

Example LLDB commands:

```text
breakpoint set --file main.cpp --line <line-number>
run
frame variable model
expr model.get()
bt
continue
```

During the failure inspection, LLDB confirmed that the RAII-managed model contained a null underlying pointer:

```text
expr model.get()
(llama_model *) $0 = nullptr
```

The program then continued through its normal error-handling path and exited with status `1`.

### AddressSanitizer and UndefinedBehaviorSanitizer

The project can also be compiled with:

- AddressSanitizer (ASan), for detecting memory-safety problems such as invalid accesses and use-after-free.
- UndefinedBehaviorSanitizer (UBSan), for detecting certain forms of undefined behavior.

Sanitizers are enabled with:

```text
-DENABLE_SANITIZERS=ON
```

The sanitizer configuration adds:

```text
-fsanitize=address,undefined
-fno-omit-frame-pointer
```

to the application compile options and enables the corresponding sanitizer options during linking.

On the development machine, the system AppleClang sanitizer runtime was incompatible with the installed macOS version, so the sanitizer build was created using a newer LLVM/Clang toolchain.

The pinned llama.cpp version also required several missing direct standard-library includes to compile with the newer toolchain.

For sanitizer testing, Metal was disabled and the CPU backend was used:

```bash
cmake -S . -B build-asan-llvm \
  -DLLAMA_CPP_DIR=/path/to/llama.cpp \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_SANITIZERS=ON \
  -DGGML_METAL=OFF \
  -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++"
```

Then:

```bash
cmake --build build-asan-llvm
```

A successful sanitizer run completed the same inference path without any ASan or UBSan diagnostics.

The sanitizer build is used for correctness testing rather than performance benchmarking because instrumentation can affect runtime performance.

## Implementation Notes and Design Decisions

### Sampling

The application uses a configurable stochastic sampling chain:

```text
model logits
    ↓
top-k filtering
    ↓
top-p filtering
    ↓
temperature scaling
    ↓
probabilistic token selection
```

The default interactive configuration is:

```text
top-k = 40
top-p = 0.9
temperature = 0.8
seed = 42
```

Top-k restricts sampling to the highest-scoring candidate tokens. Top-p further restricts the candidate set to the smallest group whose cumulative probability reaches the configured threshold. Temperature reshapes the probability distribution before the final stochastic selection.

The random seed initializes the sampler's pseudorandom number generator. Using the same seed and the same model, prompt, and sampling configuration reproduces the same sampling sequence under the same execution environment.

Benchmark mode uses a fixed sampling configuration so sampling changes do not unintentionally alter the workload during throughput comparisons.

### Prompt prefill and autoregressive decoding

The application uses the same `llama_decode()` function for both prompt processing and token-by-token generation.

The first batch contains the full tokenized prompt:

```text
[P0, P1, P2, ...]
```

After a token is sampled, the next batch contains only that newly generated token:

```text
[G1]
```

The runtime context retains the state from previously evaluated tokens, so the full prompt does not need to be resent on every generation step.

This produces the autoregressive sequence:

```text
prompt batch
    ↓
decode
    ↓
sample G1
    ↓
decode G1
    ↓
sample G2
    ↓
decode G2
    ↓
...
```

### Batch lifetime

`llama_batch_get_one()` references existing token memory rather than taking ownership of it.

For the initial prompt batch, the referenced memory is owned by:

```cpp
std::vector<llama_token> prompt_tokens;
```

For generated-token batches, `new_token_id` is declared outside the generation loop so that its memory remains valid while `llama_batch` points to it.

This avoids creating a dangling pointer between generation iterations.

### Vocabulary ownership

The vocabulary is retrieved from the loaded model:

```cpp
const llama_vocab * vocab =
    llama_model_get_vocab(model.get());
```

The application treats this as a borrowed pointer.

It is used for:

- Prompt tokenization
- End-of-generation checks
- Converting generated token IDs back into text pieces

The vocabulary is not independently freed because its lifetime is tied to the model.

### Context and batch sizing

The application computes the minimum context required for the requested workload as:

```text
required context = prompt tokens + requested generated tokens - 1
```

If `--context` is provided, the requested capacity must be large enough for the workload. If it is omitted, the program requests the minimum required context from llama.cpp.

The minimum batch capacity is the prompt token count because the initial prompt prefill is the largest multi-token batch used by the application. If `--batch` is provided, it must be large enough to hold the prompt. Otherwise, the prompt token count is used.

llama.cpp may internally adjust the requested capacities, so the application reports both requested and effective context and batch values.

The `-1` in the required-context calculation accounts for the fact that the final generated token is sampled from the logits of the preceding decode and does not need to be decoded again when generation stops.

### Timing

Inference timing uses:

```cpp
std::chrono::steady_clock
```

rather than a wall-clock timer because `steady_clock` is designed for measuring elapsed durations and is not affected by system-clock adjustments.

Before the measured run, the application performs a warm-up that exercises both the multi-token prefill path and the single-token autoregressive decode path. Runtime and KV-cache state are then cleared before measurement begins.

Because Metal execution can be asynchronous, `llama_synchronize()` is called before stopping each decode timer. This ensures that the measured duration includes completion of the GPU work rather than only the CPU-side submission time.

The application reports three performance measurements:

- **Prefill throughput** — prompt tokens divided by the measured initial decode time.
- **Autoregressive decode throughput** — autoregressive decode steps divided by their cumulative decode time.
- **End-to-end generation throughput** — generated tokens divided by total generation-loop time.

End-to-end time includes more than model decode alone, including sampling, token-to-piece conversion, and generation-loop overhead. The separate decode measurement is therefore useful for distinguishing model execution cost from broader application overhead.

## Current Scope

This version focuses on understanding and implementing the core llama.cpp inference lifecycle:

```text
load
→ tokenize
→ configure
→ prefill
→ sample
→ autoregressive decode
→ detokenize
→ measure
```

The project focuses on making the llama.cpp inference lifecycle observable and configurable while keeping the implementation small enough to inspect end-to-end. It now includes configurable stochastic sampling, runtime controls, warm-up, phase-specific timing, and a dedicated benchmark mode for controlled performance experiments.

## Inference Instrumentation & Benchmarking

The inference loop exposes separate measurements for:

- **Prefill**: processing the full prompt before generation begins.
- **Autoregressive decode**: processing one generated token at a time using the KV cache.
- **End-to-end generation**: total generation-loop throughput, including sampling and other overhead.

The CLI also supports runtime configuration of context capacity, batch capacity, and CPU thread count:

```bash
./build/llama-cpp-inference \
  --model <model.gguf> \
  --prompt "Your prompt" \
  --tokens 32 \
  --context 256 \
  --batch 32 \
  --threads 2
```

The program reports both requested and effective runtime settings because llama.cpp may internally adjust values such as context and batch capacity.

### Benchmark Setup

- Hardware: Apple M2 Pro
- Backend: Metal
- Model: Stories 15M
- Quantization: Q4_0
- Prompt tokens: 28
- Generated tokens: 32
- Actual sequence length: 59 positions
- Metal compute paths warmed before measurement
- GPU synchronization performed before stopping per-phase timers

The benchmark varies effective context capacity, batch capacity, and CPU thread count while keeping the token workload constant.

| Context | Batch | Threads | Prefill tok/s | Decode tok/s | End-to-End tok/s |
|--------:|------:|--------:|--------------:|-------------:|-----------------:|
| 256 | 32 | 2 | 7,407 | 527 | 499 |
| 256 | 32 | 6 | 7,544 | 701 | 647 |
| 256 | 64 | 2 | 8,041 | 783 | 719 |
| 256 | 64 | 6 | 7,693 | 787 | 718 |
| 512 | 32 | 2 | 8,189 | 687 | 639 |
| 512 | 32 | 6 | 7,894 | 764 | 701 |
| 512 | 64 | 2 | 7,729 | 833 | 756 |
| 512 | 64 | 6 | 8,114 | 756 | 695 |

### Observations

Prefill throughput remained relatively stable at approximately **7.4k–8.2k tokens/s**, while autoregressive decode ranged from approximately **527–833 tokens/s**.

For this workload, autoregressive decoding accounted for more than **90% of measured model-compute time**, making sequential token generation the dominant latency bottleneck.

The effects of context capacity, batch capacity, and CPU thread count were not consistent enough across these single-run measurements to claim a universally optimal configuration. These exploratory measurements motivated the repeated-run methodology documented below.

The configured batch size represents the maximum batch capacity available to llama.cpp. The benchmark prompt contains only 28 tokens, so increasing the configured batch from 32 to 64 does not increase the number of tokens processed during the actual prefill operation.

### Reproducible Benchmark Methodology

Benchmark mode is intended for controlled throughput comparisons rather than subjective generation experiments.

To keep benchmark runs comparable, the following are held constant unless they are the variable being intentionally tested:

- Model
- Prompt
- Requested generation length
- Context capacity
- Batch capacity
- Sampling configuration
- Random seed
- Backend and hardware
- Warm-up procedure

Benchmark mode fixes the sampling configuration to:

```text
seed = 42
top-k = 40
top-p = 0.9
temperature = 0.8
```

User-provided sampling overrides are rejected in benchmark mode so that sampling changes do not accidentally alter the workload being measured.

Before measurement, the application warms both the prefill and one-token decode paths. Runtime and KV-cache state are then cleared before the measured run.

For controlled experiments, each configuration is measured five times and the median throughput is reported. Repeated runs help expose runtime variability caused by effects such as OS scheduling, GPU behavior, and background system activity. Reporting the median reduces sensitivity to unusually fast or slow individual runs.

A fixed seed makes the sampling path repeatable under the same configuration, but it does not make runtime timings deterministic. Performance measurements can still vary between runs.

The requested token count is treated as a maximum because generation may terminate early if the model emits an end-of-generation token. The actual generated-token count is therefore reported and should be checked when comparing runs.

### Repeated-Run Thread Experiment

A controlled benchmark compared two requested CPU thread counts while holding the remaining workload and sampling configuration fixed.

Configuration:

```text
Model: Stories 15M Q4_0
Hardware: Apple M2 Pro
Backend: Metal
Context: 256
Batch: 64
Requested tokens: 30
Runs per configuration: 5
Benchmark sampling: fixed
```

All runs generated the full 30 requested tokens.

| Threads | Median Prefill tok/s | Median Decode tok/s | Median End-to-End tok/s |
|--------:|---------------------:|--------------------:|------------------------:|
| 2 | 7,791.22 | 509.37 | 481.56 |
| 6 | 7,742.11 | 506.21 | 478.35 |

The median results were very close. Moving from 2 to 6 requested CPU threads changed median prefill, decode, and end-to-end throughput by less than 1%.

For this model, hardware, Metal backend, prompt, and benchmark configuration, the measurements do not show a meaningful throughput advantage for 6 requested CPU threads over 2. Because runtime noise remains present even with a controlled workload, these results should not be interpreted as a universal conclusion about thread scaling.

### Reproducible CLI Examples

#### Interactive generation

Interactive mode is intended for experimenting with generated text and sampling behavior.

```bash
./build/llama-cpp-inference \
  --model /path/to/stories15M-q4_0.gguf \
  --prompt "Once upon a time" \
  --mode interactive \
  --tokens 30 \
  --context 256 \
  --batch 64 \
  --threads 4 \
  --seed 42 \
  --top-k 40 \
  --top-p 0.9 \
  --temperature 0.8
```

Sampling parameters may be changed in interactive mode. For example:

```bash
./build/llama-cpp-inference \
  --model /path/to/stories15M-q4_0.gguf \
  --prompt "Once upon a time" \
  --mode interactive \
  --tokens 30 \
  --seed 69 \
  --top-k 20 \
  --top-p 0.8 \
  --temperature 1.2
```

#### Benchmark mode

Benchmark mode uses a fixed sampling configuration and is intended for controlled performance measurements.

```bash
./build/llama-cpp-inference \
  --model /path/to/stories15M-q4_0.gguf \
  --prompt "Once upon a time" \
  --mode benchmark \
  --tokens 30 \
  --context 256 \
  --batch 64 \
  --threads 2
```

Benchmark mode internally fixes:

```text
seed = 42
top-k = 40
top-p = 0.9
temperature = 0.8
```

Sampling overrides are intentionally rejected in benchmark mode.

For example, the following command is invalid:

```bash
./build/llama-cpp-inference \
  --model /path/to/stories15M-q4_0.gguf \
  --prompt "Once upon a time" \
  --mode benchmark \
  --temperature 1.2
```

#### Controlled thread comparison

To compare thread counts, keep the rest of the workload unchanged and modify only `--threads`.

Two-thread configuration:

```bash
./build/llama-cpp-inference \
  --model /path/to/stories15M-q4_0.gguf \
  --prompt "Once upon a time" \
  --mode benchmark \
  --tokens 30 \
  --context 256 \
  --batch 64 \
  --threads 2
```

Six-thread configuration:

```bash
./build/llama-cpp-inference \
  --model /path/to/stories15M-q4_0.gguf \
  --prompt "Once upon a time" \
  --mode benchmark \
  --tokens 30 \
  --context 256 \
  --batch 64 \
  --threads 6
```

Each configuration should be run multiple times while keeping the model, prompt, generation length, context, batch capacity, hardware, backend, and benchmark sampling configuration constant. The median throughput across the repeated runs can then be used for comparison.

## KV Cache

During prompt processing, the transformer computes key and value vectors for each token at every layer. These values are stored in the **KV cache**.

During autoregressive generation, previous K/V vectors are reused rather than recomputed. Each newly processed token computes its own Q/K/V values, uses its query to attend over previously cached keys and values, and appends its new K/V vectors to the cache.

This avoids repeatedly recomputing the entire sequence during generation, but the KV cache grows with sequence length, increasing memory usage and the amount of cached data that each subsequent token must attend over.

## What I Learned

This project developed practical understanding of the llama.cpp inference lifecycle, including model loading, tokenization, prompt prefill, autoregressive decoding, KV-cache reuse, sampling, and detokenization.

The benchmarking work also showed why inference performance measurements require careful synchronization, warm-up, controlled workloads, repeated runs, and separation of prefill, decode, and end-to-end timing.

Implementing configurable sampling reinforced the distinction between model evaluation and token selection, as well as the role of top-k, top-p, temperature, and random seeds in reproducible generation.

## What I Would Improve Next

Future improvements could include:

- More comprehensive automated CLI validation and tests
- Stronger validation of floating-point sampling arguments
- Automated benchmark result collection instead of manual repeated runs
- Additional models and quantization formats
- Profiling to identify specific runtime bottlenecks rather than only measuring their timing