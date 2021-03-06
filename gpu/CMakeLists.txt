cmake_minimum_required(VERSION 3.11)
project(hw1 LANGUAGES C CXX)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

foreach (BENCHMARK IN ITEMS gpu gpu-blocked)
    # Create benchmarking executable
    add_executable(benchmark-${BENCHMARK} benchmark-gpu.cu dws-gpu.h dws-${BENCHMARK}.cu)
    target_compile_features(benchmark-gpu PRIVATE cxx_std_11 c_restrict)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

    # Generate job script
    configure_file(job.in job-gpu)
endforeach ()