#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include "llama.h"
#include "ggml-backend.h"


int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt;
    int n_predict = 64;
    std::optional<int> context_size;
    std::optional<int> batch_size;
    std::optional<int> thread_count;
    std::optional<uint32_t> input_seed;
    int top_k = 40;
    float top_p = 0.90f;
    float temperature = 0.8f;
    enum class RunMode {
        Interactive,
        Benchmark
    };
    RunMode mode = RunMode::Interactive;
    bool seed_provided = false;
    bool top_k_provided = false;
    bool top_p_provided = false;
    bool temperature_provided = false;
    constexpr uint32_t BENCHMARK_SEED = 42;
    constexpr int BENCHMARK_TOP_K = 40;
    constexpr float BENCHMARK_TOP_P = 0.9f;
    constexpr float BENCHMARK_TEMPERATURE = 0.8f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // --model
        if(arg == "--model"){
            if(i + 1 >= argc){
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--mode <interactive|benchmark>] [--tokens <n>] [--context <n>] [--batch <n>] [--threads <n>] [--seed <n>] [--top-k <n>] [--top-p <p>] [--temperature <t>]\n";
                return 1;
            }
            model_path = argv[i+1];
            ++i;
        }
        // --prompt
        else if(arg == "--prompt"){
            if(i + 1 >= argc){
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--mode <interactive|benchmark>] [--tokens <n>] [--context <n>] [--batch <n>] [--threads <n>] [--seed <n>] [--top-k <n>] [--top-p <p>] [--temperature <t>]\n";
                return 1;
            }
            prompt = argv[i+1];
            ++i;
        }
        else if (arg == "--mode") {
            if (i + 1 >= argc) {
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--mode <interactive|benchmark>] [--tokens <n>] [--context <n>] [--batch <n>] [--threads <n>] [--seed <n>] [--top-k <n>] [--top-p <p>] [--temperature <t>]\n";
                return 1;
            }

            std::string proposed_mode = argv[i + 1];
            if (proposed_mode == "benchmark") {
                mode = RunMode::Benchmark;
            }
            else if (proposed_mode == "interactive") {
                mode = RunMode::Interactive;
            }
            else {
                std::cerr << "Error: --mode must be either 'interactive' or 'benchmark'.\n";
                return 1;
            }
            ++i;
        }
        // --tokens
        else if (arg == "--tokens") {
            if (i + 1 >= argc) {
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--mode <interactive|benchmark>] [--tokens <n>] [--context <n>] [--batch <n>] [--threads <n>] [--seed <n>] [--top-k <n>] [--top-p <p>] [--temperature <t>]\n";
                return 1;
            }

            std::string value = argv[i + 1];
            std::size_t pos;

            try {
                n_predict = std::stoi(value, &pos);

                if (pos != value.size()) {
                    std::cerr << "Error: --tokens must be an integer.\n";
                    return 1;
                }
            }
            catch (const std::exception & e) {
                std::cerr << "Error: --tokens must be an integer.\n";
                return 1;
            }

            ++i;
        }
        else if (arg == "--context") {
            if (i + 1 >= argc) {
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--mode <interactive|benchmark>] [--tokens <n>] [--context <n>] [--batch <n>] [--threads <n>] [--seed <n>] [--top-k <n>] [--top-p <p>] [--temperature <t>]\n";
                return 1;
            }

            std::string value = argv[i + 1];
            std::size_t pos;

            try {
                context_size = std::stoi(value, &pos);

                if (pos != value.size()) {
                    std::cerr << "Error: --context must be an integer.\n";
                    return 1;
                }
                if (context_size <= 0) {
                    std::cerr << "Error: --context must be greater than 0.\n";
                    return 1;
                }
            }
            catch (const std::exception & e) {
                std::cerr << "Error: --context must be an integer.\n";
                return 1;
            }

            ++i;
        }
        else if (arg == "--batch") {
            if (i + 1 >= argc) {
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--mode <interactive|benchmark>] [--tokens <n>] [--context <n>] [--batch <n>] [--threads <n>] [--seed <n>] [--top-k <n>] [--top-p <p>] [--temperature <t>]\n";
                return 1;
            }

            std::string value = argv[i + 1];
            std::size_t pos;

            try {
                batch_size = std::stoi(value, &pos);

                if (pos != value.size()) {
                    std::cerr << "Error: --batch must be an integer.\n";
                    return 1;
                }
            }
            catch (const std::exception & e) {
                std::cerr << "Error: --batch must be an integer.\n";
                return 1;
            }

            ++i;
        }
        else if (arg == "--threads") {
            if (i + 1 >= argc) {
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--mode <interactive|benchmark>] [--tokens <n>] [--context <n>] [--batch <n>] [--threads <n>] [--seed <n>] [--top-k <n>] [--top-p <p>] [--temperature <t>]\n";
                return 1;
            }

            std::string value = argv[i + 1];
            std::size_t pos;

            try {
                thread_count = std::stoi(value, &pos);

                if (pos != value.size()) {
                    std::cerr << "Error: --threads must be an integer.\n";
                    return 1;
                }
            }
            catch (const std::exception & e) {
                std::cerr << "Error: --threads must be an integer.\n";
                return 1;
            }

            ++i;
        }
        else if (arg == "--seed") {
            if (i + 1 >= argc) {
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--mode <interactive|benchmark>] [--tokens <n>] [--context <n>] [--batch <n>] [--threads <n>] [--seed <n>] [--top-k <n>] [--top-p <p>] [--temperature <t>]\n";
                return 1;
            }

            std::string value = argv[i + 1];
            std::size_t pos;

            try {
                long long parsed_seed = std::stoll(value, &pos);

                if (pos != value.size()) {
                    std::cerr << "Error: --seed must be an integer.\n";
                    return 1;
                }
                if (parsed_seed < 0 || parsed_seed >= static_cast<long long>(UINT32_MAX)) {
                    std::cerr << "Error: --seed must be between 0 and " << UINT32_MAX - 1 << ".\n";
                    return 1;
                }

                input_seed = static_cast<uint32_t>(parsed_seed);
            }
            catch (const std::exception & e) {
                std::cerr << "Error: --seed must be an integer.\n";
                return 1;
            }
            seed_provided = true;
            ++i;
        }
        else if (arg == "--top-k") {
            if (i + 1 >= argc) {
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--mode <interactive|benchmark>] [--tokens <n>] [--context <n>] [--batch <n>] [--threads <n>] [--seed <n>] [--top-k <n>] [--top-p <p>] [--temperature <t>]\n";
                return 1;
            }

            std::string value = argv[i + 1];
            std::size_t pos;

            try {
                top_k = std::stoi(value, &pos);

                if (pos != value.size()) {
                    std::cerr << "Error: --top-k must be an integer.\n";
                    return 1;
                }
            }
            catch (const std::exception & e) {
                std::cerr << "Error: --top-k must be an integer.\n";
                return 1;
            }
            top_k_provided = true;
            ++i;
        }
        else if (arg == "--top-p") {
            if (i + 1 >= argc) {
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--mode <interactive|benchmark>] [--tokens <n>] [--context <n>] [--batch <n>] [--threads <n>] [--seed <n>] [--top-k <n>] [--top-p <p>] [--temperature <t>]\n";
                return 1;
            }

            std::string value = argv[i + 1];
            std::size_t pos;

            try {
                top_p = std::stof(value, &pos);

                if (pos != value.size()) {
                    std::cerr << "Error: --top-p must be a number.\n";
                    return 1;
                }
            }
            catch (const std::exception & e) {
                std::cerr << "Error: --top-p must be a number.\n";
                return 1;
            }
            top_p_provided = true;
            ++i;
        }
        else if (arg == "--temperature") {
            if (i + 1 >= argc) {
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--mode <interactive|benchmark>] [--tokens <n>] [--context <n>] [--batch <n>] [--threads <n>] [--seed <n>] [--top-k <n>] [--top-p <p>] [--temperature <t>]\n";
                return 1;
            }

            std::string value = argv[i + 1];
            std::size_t pos;

            try {
                temperature = std::stof(value, &pos);

                if (pos != value.size()) {
                    std::cerr << "Error: --temperature must be a number.\n";
                    return 1;
                }
            }
            catch (const std::exception & e) {
                std::cerr << "Error: --temperature must be a number.\n";
                return 1;
            }
            temperature_provided = true;
            ++i;
        }
        else {
            std::cerr << "Error: unknown argument '" << arg << "'.\n";
            return 1;
        }
    }

    // validate model_path
    if(model_path.empty()){
        std::cout << "model_path has to be something" << "\n";
        return 1;
    }
    // validate prompt
    if(prompt.empty()){
        std::cout << "Prompt has to be something" << "\n";
        return 1;
    }
    // validate n_predict
    if(n_predict <= 0){
        std::cout << "Tokens has to be a positive non-zero integer" << "\n";
        return 1;
    }
    // validate context_size
    if (context_size.has_value() && context_size.value() <= 0) {
        std::cerr << "Context has to be a positive non-zero integer\n";
        return 1;
    }
    // validate batch_size
    if (batch_size.has_value() && batch_size.value() <= 0) {
        std::cerr << "Batch has to be a positive non-zero integer\n";
        return 1;
    }
    // validate thread_count
    if (thread_count.has_value() && thread_count.value() <= 0) {
        std::cerr << "Threads has to be a positive non-zero integer\n";
        return 1;
    }
    // setting seed for stochastic sampling
    uint32_t seed = BENCHMARK_SEED;
    if(input_seed.has_value()){
        seed = input_seed.value();
    }
    // validate top-k
    if (top_k <= 0) {
        std::cerr << "Top-k has to be a positive non-zero integer\n";
        return 1;
    }
    // validate top-p
    if (top_p <= 0.0f || top_p > 1.0f) {
        std::cerr << "Top-p must be greater than 0 and less than or equal to 1\n";
        return 1;
    }
    // validate temperature
    if (temperature <= 0.0f){
        std::cerr << "Temperature must be greater than 0\n";
        return 1;
    }
    // validate if benchmark mode is valid
    if (mode == RunMode::Benchmark && (seed_provided || top_k_provided || top_p_provided || temperature_provided)) {
        std::cerr << "Error: custom sampling options are not allowed in benchmark mode.\n";
        return 1;
    }
    // set up fixed benchmark parameters
    if (mode == RunMode::Benchmark) {
        seed = BENCHMARK_SEED;
        top_k = BENCHMARK_TOP_K;
        top_p = BENCHMARK_TOP_P;
        temperature = BENCHMARK_TEMPERATURE;
    }


    std::cout << "Model:  " << model_path << "\n";
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Tokens: " << n_predict << "\n";
    if (mode == RunMode::Benchmark) {
                std::cout << "Mode: benchmark\n";
            }
    else {
                std::cout << "Mode: interactive\n";
    }
    std::cout << "Seed: " << seed << "\n";
    std::cout << "Top-k: " << top_k << "\n";
    std::cout << "Top-p: " << top_p << "\n";
    std::cout << "Temperature: " << temperature << "\n";


    // Load available compute backends
    ggml_backend_load_all();

    // Create model configuration
    llama_model_params model_params = llama_model_default_params();

    // Load the model (converts model_path into a C-style string and also inputs model_params)
    // model will become something that doesn't have to be explicitly freed, using RAII to our advantage
    using ModelPtr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;

    ModelPtr model(llama_model_load_from_file(model_path.c_str(), model_params),llama_model_free);  

    // Check that loading succeeded
    if (model == nullptr) {
        std::cerr << "Error: failed to load model.\n";
        return 1;
    }

    // create a vocab pointer (borrowed)
    const llama_vocab * vocab = llama_model_get_vocab(model.get());

    // call llama_tokenize() to get the amount of tokens we need to allocate in the vector
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);

    // create vector to store the prompt tokens
    std::vector<llama_token> prompt_tokens(n_prompt);

    // call llama_tokenize() to store the prompt in prompt_tokens vector
    int result = llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);

    // check if tokenization worked
    if (result < 0){
        std::cerr << "Error: failed to tokenize the prompt. \n";
        return 1;
    }

    int required_context = n_prompt + n_predict - 1;

    // print the number of tokens
    std::cout << "Prompt Tokens: " << n_prompt << "\n";
    // print the required context
    std::cout << "Required Context: " << required_context << "\n";
    // print the requested context
    if (context_size.has_value()) {
        std::cout << "Requested Context: " << context_size.value() << "\n";
    }
    else {
        std::cout << "Requested Context: default\n";
    }

    // create context parameters
    llama_context_params ctx_params = llama_context_default_params();

    // changing some context parameters
    if(context_size.has_value()){
        if(context_size.value() >= required_context){
            ctx_params.n_ctx = context_size.value();
        }
        else{
            std::cerr << "input context size is too small, required context size must be at least " << required_context << "\n";
            return 1;
        }
    }
    else{
        ctx_params.n_ctx = n_prompt + n_predict - 1;
    }
    if(batch_size.has_value()){
        if(batch_size.value() >= n_prompt){
            ctx_params.n_batch = batch_size.value();
        }
        else{
            std::cerr << "input batch size is too small, required batch size must be at least " << n_prompt << "\n";
            return 1;
        }
    }
    else{
        ctx_params.n_batch = n_prompt;
    }
    if (thread_count.has_value()) {
        ctx_params.n_threads = thread_count.value();
        ctx_params.n_threads_batch = thread_count.value();
    }

    // print the required context
    std::cout << "Required Batch: " << n_prompt << "\n";
    // print the requested context
    if (batch_size.has_value()) {
        std::cout << "Requested Batch: " << batch_size.value() << "\n";
    }
    else {
        std::cout << "Requested Batch: default\n";
    }

    ctx_params.no_perf = false;

    // create context using RAII to manage it
    using ContextPtr = std::unique_ptr<llama_context, decltype(&llama_free)>;
    ContextPtr ctx(llama_init_from_model(model.get(), ctx_params), llama_free);

    if(ctx == nullptr){
        std::cerr << "Error: failed to create context.\n";
        return 1;
    }

    // setting two thread parameters (first is for how many CPU threads are used during decode and second is for how many CPU threads are used during prefill)
    if (thread_count.has_value()) {
        llama_set_n_threads(ctx.get(), thread_count.value(), thread_count.value());
    }

    if (thread_count.has_value()) {
        std::cout << "Requested threads: " << thread_count.value() << "\n";
    } 
    else {
        std::cout << "Requested threads: default\n";
    }

    // check effective batch size that the llama API makes
    uint32_t effective_batch = llama_n_batch(ctx.get());

    // check effective context that the llama API makes
    uint32_t effective_context = llama_n_ctx(ctx.get());

    // check what threads were made
    int32_t effective_threads = llama_n_threads(ctx.get());
    int32_t effective_threads_batch = llama_n_threads_batch(ctx.get());

    std::cout << "Effective context: " << effective_context << "\n";
    std::cout << "Effective batch: " << effective_batch << "\n";
    std::cout << "Effective generation threads: " << effective_threads << "\n";
    std::cout << "Effective batch threads : " << effective_threads_batch << "\n";

    // set up sampler parameters and edit one of them
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    // create sampler pointer, initializing it with the new sampler parameters and using RAII to automatically free it once the program finishes executing it.
    using SamplerPtr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
    SamplerPtr smpl(llama_sampler_chain_init(sparams),llama_sampler_free);

    // initialize sampling chain
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(seed));

    // create batch before calling decode
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    llama_token new_token_id = 0;

    int generated_tokens = 0;

    std::string generated_text;

    // Warm up prefill path.
    llama_batch warmup_batch =
            llama_batch_get_one(prompt_tokens.data(), n_prompt);

    if (llama_decode(ctx.get(), warmup_batch) != 0) {
        std::cerr << "Error: warm-up prefill failed.\n";
        return 1;
    }
    llama_synchronize(ctx.get());

    // Warm up single-token decode path.
    llama_token warmup_token = prompt_tokens.back();

    warmup_batch = llama_batch_get_one(&warmup_token, 1);

    if (llama_decode(ctx.get(), warmup_batch) != 0) {
        std::cerr << "Error: warm-up decode failed.\n";
        return 1;
    }
    llama_synchronize(ctx.get());

    // Clear runtime/KV state before the measured run.
    llama_memory_clear(llama_get_memory(ctx.get()), false);

    // Reset sampler (seed sequence) before the measured run.
    llama_sampler_reset(smpl.get());

    bool is_prefill = true;
    
    double prefill_seconds = 0.0;

    double autoregressive_decode_seconds = 0.0;
    
    int autoregressive_decode_tokens = 0;

    // to measure how much time inference takes
    auto start_time = std::chrono::steady_clock::now();

    while(generated_tokens < n_predict){

        // start decode timer
        auto decode_start = std::chrono::steady_clock::now();

        // execute transformer architecture using decode
        int decode = llama_decode(ctx.get(), batch);

        llama_synchronize(ctx.get());

        // end decode timer
        auto decode_end = std::chrono::steady_clock::now();

        // calculate decode duration
        double decode_seconds = std::chrono::duration<double>(decode_end - decode_start).count();

        // check if decode worked
        if(decode != 0){
            std::cerr << "Decode step failed.\n";
            return 1;
        }

        if(is_prefill) {
            prefill_seconds = decode_seconds;
            is_prefill = false;
        }
        else{
            autoregressive_decode_seconds += decode_seconds;
            ++autoregressive_decode_tokens;            
        }

        // sample from the last token in the batch
        new_token_id = llama_sampler_sample(smpl.get(), ctx.get(), -1);

        if(llama_vocab_is_eog(vocab, new_token_id)) { 
            break;
        }

        // create a second buffer array made up of chars
        char buf[128];

        // n will tell us how many bytes were written
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);

        if(n < 0){
            std::cerr << "number of bytes is negative, not good! \n";
            return 1;
        }

        // construct a string from exactly those bytes
        std::string piece(buf, n);

        // add generated token to the output message string (generated message)
        generated_text+=piece;

        // replace batch with the new batch from the generated token
        batch = llama_batch_get_one(&new_token_id, 1);
        
        ++generated_tokens;

    }

    auto end_time = std::chrono::steady_clock::now();

    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();

    double prefill_tokens_per_second = n_prompt / prefill_seconds;

    double end_to_end_generated_tokens_per_second = generated_tokens / elapsed_seconds;

    // printing generated text only in interactive mode
    if (mode == RunMode::Interactive) {
        std::cout << "\nGenerated text:\n";
        std::cout << prompt << generated_text << "\n";
    }


    std::cout << "\n--- Stats ---\n";
    std::cout << "Prompt tokens: " << n_prompt << "\n";
    std::cout << "Generated tokens: " << generated_tokens << "\n";
    std::cout << "Elapsed time: " << elapsed_seconds << " s\n";
    std::cout << "Prefill time: " << prefill_seconds << " s\n";
    std::cout << "Prefill tokens/sec: " << prefill_tokens_per_second << "\n";
    std::cout << "Autoregressive decode tokens: " << autoregressive_decode_tokens << "\n";
    std::cout << "Autoregressive decode time: " << autoregressive_decode_seconds << " s\n";
    double decode_tokens_per_second = 0.0;
    if(!(autoregressive_decode_seconds == 0.0 || autoregressive_decode_tokens == 0)){
        decode_tokens_per_second = autoregressive_decode_tokens / autoregressive_decode_seconds;
        std::cout << "Decode tokens/sec: " << decode_tokens_per_second << "\n";
    }
    else{
        std::cout << "Decode tokens/sec: N/A" << "\n";
    }
    std::cout << "End-to-end generated tokens/sec: " << end_to_end_generated_tokens_per_second << "\n";

    return 0;
}