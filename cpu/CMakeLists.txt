cmake_minimum_required(VERSION 3.14)
project(hw1 LANGUAGES C CXX)

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
add_library(benchmark OBJECT benchmark-cpu.cpp)
target_compile_features(benchmark PRIVATE cxx_std_11)

foreach (BENCHMARK IN ITEMS direct single blocked microkernel omp single-blocked simd unroll)
    # Create benchmarking executable
    add_executable(benchmark-${BENCHMARK} dws-${BENCHMARK}.cpp)
    target_link_libraries(benchmark-${BENCHMARK} PRIVATE benchmark OpenMP::OpenMP_CXX)
    target_compile_features(benchmark-${BENCHMARK} PRIVATE c_std_11 c_restrict)
    target_compile_options(benchmark-${BENCHMARK} PRIVATE -g -Wall -pedantic -march=knl -ffast-math -Ofast)

    # Generate job script
    configure_file(job.in job-${BENCHMARK})
endforeach ()


# Common library target for sweeping.
add_library(sweep OBJECT sweep-cpu.cpp)
target_compile_features(sweep PRIVATE cxx_std_11)

foreach (SWEEP IN ITEMS blocked)
    # Create sweeping executable
    add_executable(sweep-${SWEEP} dws-${SWEEP}.cpp)
    target_link_libraries(sweep-${SWEEP} PRIVATE sweep OpenMP::OpenMP_CXX)
    target_compile_features(sweep-${SWEEP} PRIVATE c_std_11 c_restrict)
    target_compile_options(sweep-${SWEEP} PRIVATE -Wall -pedantic -march=knl -ffast-math)

    # Generate job script
    configure_file(job.in sweep-job-${SWEEP})
endforeach ()