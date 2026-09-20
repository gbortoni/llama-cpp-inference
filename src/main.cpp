#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include "llama.h"
#include "ggml-backend.h"


int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt;
    int n_predict = 64;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // --model
        if(arg == "--model"){
            if(i + 1 >= argc){
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--tokens <n>]" << "\n";
                return 1;
            }
            model_path = argv[i+1];
            ++i;
        }
        // --prompt
        else if(arg == "--prompt"){
            if(i + 1 >= argc){
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--tokens <n>]" << "\n";
                return 1;
            }
            prompt = argv[i+1];
            ++i;
        }
        // --tokens
        else if (arg == "--tokens") {
            if (i + 1 >= argc) {
                std::cout << "Usage: ./llama-cpp-inference --model <path> --prompt <text> [--tokens <n>]\n";
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

    std::cout << "Model:  " << model_path << "\n";
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Tokens: " << n_predict << "\n";

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

    // print the number of tokens
    std::cout << "Prompt Tokens: " << n_prompt << "\n";

    // create context parameters
    llama_context_params ctx_params = llama_context_default_params();

    // changing some context parameters
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;

    // create context using RAII to manage it
    using ContextPtr = std::unique_ptr<llama_context, decltype(&llama_free)>;
    ContextPtr ctx(llama_init_from_model(model.get(), ctx_params), llama_free);

    if(ctx == nullptr){
        std::cerr << "Error: failed to create context.\n";
        return 1;
    }

    // set up sampler parameters and edit one of them
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;

    // create sampler pointer, initializing it with the new sampler parameters and using RAII to automatically free it once the program finishes executing it.
    using SamplerPtr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
    SamplerPtr smpl(llama_sampler_chain_init(sparams),llama_sampler_free);

    // initialize greedy sampling
    llama_sampler_chain_add(smpl.get(),llama_sampler_init_greedy());

    // create batch before calling decode
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    llama_token new_token_id = 0;

    int generated_tokens = 0;

    std::string generated_text;

    // to measure how much time inference takes
    auto start_time = std::chrono::steady_clock::now();

    while(generated_tokens < n_predict){

        // execute transformer architecture using decode
        int decode = llama_decode(ctx.get(), batch);

        // check if decode worked
        if(decode != 0){
            std::cerr << "Decode step failed.\n";
            return 1;
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

    double tokens_per_second = generated_tokens / elapsed_seconds;

    std::cout << "\nGenerated text:\n";
    std::cout << prompt << generated_text << "\n";

    std::cout << "\n--- Stats ---\n";
    std::cout << "Prompt tokens: " << n_prompt << "\n";
    std::cout << "Generated tokens: " << generated_tokens << "\n";
    std::cout << "Elapsed time: " << elapsed_seconds << " s\n";
    std::cout << "Tokens/sec: " << tokens_per_second << "\n";

    return 0;
}