cmake_minimum_required(VERSION 3.16)

project(llama-cpp-inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR
        "Please provide the path to llama.cpp with -DLLAMA_CPP_DIR=/path/to/llama.cpp"
    )
endif()

add_subdirectory(
    "${LLAMA_CPP_DIR}"
    "${CMAKE_BINARY_DIR}/llama.cpp-build"
    EXCLUDE_FROM_ALL
)

add_executable(
    llama-cpp-inference
    src/main.cpp
)

target_link_libraries(
    llama-cpp-inference
    PRIVATE llama
)

option(ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)

if(ENABLE_SANITIZERS)
    target_compile_options(
        llama-cpp-inference
        PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )

    target_link_options(
        llama-cpp-inference
        PRIVATE
        -fsanitize=address,undefined
    )
endif()