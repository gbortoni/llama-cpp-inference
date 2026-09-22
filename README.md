# llama-cpp-inference

A lightweight C++ command-line inference application built with the llama.cpp API for running local GGUF language models.

The application accepts a model path, prompt, and requested number of generated tokens through command-line arguments. It loads the model, tokenizes the prompt using the model vocabulary, creates the runtime context and greedy sampler, and runs an autoregressive inference loop.

The initial multi-token batch performs prompt prefill, while subsequent one-token batches perform autoregressive decoding. The application reconstructs the generated text and reports prompt tokens, generated tokens, total inference time, and throughput in tokens per second.

The project also uses RAII-based resource management and includes separate Debug and ASan/UBSan build configurations for debugging and memory-safety validation.

## Features

- Local GGUF model inference through the llama.cpp C API
- Command-line arguments for model path, prompt, and generation length
- Prompt tokenization and vocabulary handling
- Prompt prefill followed by one-token autoregressive decoding
- Greedy next-token sampling
- End-of-generation token handling
- Generated text reconstruction from token pieces
- Prompt and generated token counting
- Total inference timing and tokens/sec measurement
- RAII-based ownership of model, context, and sampler resources
- Separate normal, Debug, and ASan/UBSan build configurations
- LLDB-compatible Debug build for inspecting runtime failures

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
- `--tokens <n>` — Maximum number of tokens to generate. Optional; defaults to `64`.

For example:

```bash
./build/llama-cpp-inference \
  --model /path/to/stories15M-q4_0.gguf \
  --prompt "Once upon a time" \
  --tokens 10
```

The application validates the required arguments and rejects invalid token counts such as non-integer or non-positive values.

## Example Output

Example generation using greedy sampling:

```text
Generated text:
Once upon a time, there was a little girl named Lily.

--- Stats ---
Prompt tokens: 5
Generated tokens: 10
Elapsed time: 0.0512352 s
Tokens/sec: 195.178
```

The reported throughput is calculated as:

```text
generated tokens / total inference-loop time
```

The current timing measurement includes both the initial prompt prefill and subsequent autoregressive decode steps. It is therefore intended as a simple end-to-end inference measurement rather than an isolated decode-throughput benchmark.

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
Create greedy sampler
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

### Greedy sampling

The project currently uses greedy sampling:

```cpp
llama_sampler_chain_add(
    smpl.get(),
    llama_sampler_init_greedy()
);
```

At each generation step, the sampler selects the highest-scoring next token from the model output.

More advanced sampling strategies such as temperature, top-k, top-p, or repetition penalties are intentionally outside the scope of this version.

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

### Context sizing

The context is configured using:

```cpp
ctx_params.n_ctx = n_prompt + n_predict - 1;
ctx_params.n_batch = n_prompt;
```

`n_batch` is set to the prompt length because the initial prompt prefill is the largest batch used by this application. Subsequent decode calls process one generated token at a time.

The `-1` in the context-size calculation accounts for the fact that the final generated token is sampled from the logits of the previous decode and does not need to be decoded again when generation stops.

### Timing

Inference timing uses:

```cpp
std::chrono::steady_clock
```

rather than a wall-clock timer because `steady_clock` is designed for measuring elapsed durations and is not affected by system-clock adjustments.

The current timing measurement covers the complete inference loop:

```text
prompt prefill
+
autoregressive decode steps
```

Throughput is calculated as:

```text
generated tokens / elapsed inference time
```

This is an end-to-end inference measurement. Separate prefill and decode timing can be added in future benchmarking work.

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

The project intentionally keeps sampling and performance analysis simple so that the inference loop, resource ownership, debugging workflow, and runtime behavior remain easy to inspect.

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

The effects of context capacity, batch capacity, and CPU thread count were not consistent enough across these single-run measurements to claim a universally optimal configuration. More rigorous repeated-run statistical benchmarking is planned separately.

The configured batch size represents the maximum batch capacity available to llama.cpp. The benchmark prompt contains only 28 tokens, so increasing the configured batch from 32 to 64 does not increase the number of tokens processed during the actual prefill operation.

## KV Cache

During prompt processing, the transformer computes key and value vectors for each token at every layer. These values are stored in the **KV cache**.

During autoregressive generation, previous K/V vectors are reused rather than recomputed. Each newly processed token computes its own Q/K/V values, uses its query to attend over previously cached keys and values, and appends its new K/V vectors to the cache.

This avoids repeatedly recomputing the entire sequence during generation, but the KV cache grows with sequence length, increasing memory usage and the amount of cached data that each subsequent token must attend over.