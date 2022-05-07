cmake_minimum_required(VERSION 3.11)
project(hw1 LANGUAGES C CXX)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

# Common library target for benchmarking.
add_library(benchmark OBJECT benchmark-direct.cpp)
target_compile_features(benchmark PRIVATE cxx_std_11)

foreach (BENCHMARK IN ITEMS direct single blocked microkernel omp single-blocked)
    # Create benchmarking executable
    add_executable(benchmark-${BENCHMARK} dws-${BENCHMARK}.c)
    target_link_libraries(benchmark-${BENCHMARK} PRIVATE OpenMP::OpenMP_C)
    target_link_libraries(benchmark-${BENCHMARK} PRIVATE benchmark ${BLAS_LIBRARIES})
    target_compile_features(benchmark-${BENCHMARK} PRIVATE c_std_11 c_restrict)
    target_compile_options(benchmark-${BENCHMARK} PRIVATE -Wall -pedantic -march=knl)

    # Generate job script
    configure_file(job.in job-${BENCHMARK})
endforeach ()

add_executable(benchmark-gpu dws-gpu.cu)
target_compile_features(gpu PRIVATE cxx_std_11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")
configure_file(job-gpu job-gpu COPYONLY)
