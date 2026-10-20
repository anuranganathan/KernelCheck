// Shared helpers for KernelCheck test programs.
//
// Output protocol (what the Python framework expects):
//   * stdout: the results only, as whitespace-separated numbers.
//   * stderr: diagnostics, plus one "kernel_ms=<time>" line from KernelTimer.
//   * exit code 0 on success, non-zero on any CUDA or argument error.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// Wrap every CUDA runtime call: on failure, print where it happened and exit.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                   \
                    cudaGetErrorName(err_), __FILE__, __LINE__,               \
                    cudaGetErrorString(err_));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel launches return nothing, so launch errors (e.g. an invalid block
// size) must be fetched explicitly right after the launch.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// Read argv[index] as a positive integer no larger than max_value.
inline int parse_size_arg(int argc, char** argv, int index, int max_value = 1 << 24) {
    if (index >= argc) {
        fprintf(stderr, "usage error: missing argument %d\n", index);
        exit(2);
    }
    char* end = nullptr;
    long value = strtol(argv[index], &end, 10);
    if (*end != '\0' || value <= 0 || value > max_value) {
        fprintf(stderr, "usage error: '%s' is not an integer in [1, %d]\n",
                argv[index], max_value);
        exit(2);
    }
    return static_cast<int>(value);
}

// Same formulas as int_inputs() in reference/references.py.
// a is in [-50, 50] and b in [-18, 18], so later arithmetic cannot overflow.
inline void make_int_inputs(std::vector<int>& a, std::vector<int>& b, int n) {
    a.resize(n);
    b.resize(n);
    for (int i = 0; i < n; ++i) {
        a[i] = (i * 7) % 101 - 50;
        b[i] = (i * 13) % 37 - 18;
    }
}

inline void print_ints(const std::vector<int>& values) {
    for (int v : values) printf("%d\n", v);
}

// %.9g prints enough digits to reproduce any float32 value exactly.
inline void print_floats(const std::vector<float>& values) {
    for (float v : values) printf("%.9g\n", v);
}

// Measures GPU time between begin() and end() using CUDA events, which time
// the work on the device itself (unlike a wall clock around the process,
// which also includes CUDA context creation).
struct KernelTimer {
    cudaEvent_t start, stop;
    KernelTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~KernelTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin() { CUDA_CHECK(cudaEventRecord(start)); }
    // Waits for the timed work to finish, so asynchronous kernel errors
    // are reported here as well.
    void end() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        fprintf(stderr, "kernel_ms=%.4f\n", ms);
    }
};
