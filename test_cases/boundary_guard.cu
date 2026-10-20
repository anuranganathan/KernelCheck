// boundary_guard: n is deliberately not a multiple of the block size, so the
// last block has threads with i >= n that must not write anything.
// The buffer is allocated with `guard` extra elements, all set to -1. If the
// bounds check were missing (or miscompiled), the extra threads would
// overwrite them. The writes stay inside our own allocation, so this test is
// safe even when it fails.
#include "kc_common.h"

__global__ void fill_kernel(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 2 * i + 1;
}

int main(int argc, char** argv) {
    int n = parse_size_arg(argc, argv, 1);
    int guard = parse_size_arg(argc, argv, 2, 4096);
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    int extra_threads = blocks * threads - n;
    if (guard < extra_threads)
        fprintf(stderr, "note: guard (%d) is smaller than the %d extra threads\n",
                guard, extra_threads);

    int total = n + guard;
    std::vector<int> out(total);
    int* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(int)));
    // Every byte 0xFF -> every int is -1 (two's complement).
    CUDA_CHECK(cudaMemset(d_out, 0xFF, total * sizeof(int)));

    KernelTimer timer;
    timer.begin();
    fill_kernel<<<blocks, threads>>>(d_out, n);
    CUDA_CHECK_LAUNCH();
    timer.end();

    CUDA_CHECK(cudaMemcpy(out.data(), d_out, total * sizeof(int), cudaMemcpyDeviceToHost));
    print_ints(out);
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
