// collatz: data-dependent loops and branches. Thread i counts the Collatz
// steps needed to reach 1 from i + 1. Neighbouring threads take different
// branches and loop different numbers of times (warp divergence), which
// exercises the compiler's control-flow handling.
#include "kc_common.h"

// Upper bound so the loop is guaranteed to terminate.
constexpr int MAX_STEPS = 10000;

__global__ void collatz_kernel(int* steps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long value = i + 1;   // 64-bit: intermediate values exceed the start value
    int count = 0;
    while (value != 1 && count < MAX_STEPS) {
        if (value % 2 == 0)
            value /= 2;
        else
            value = 3 * value + 1;
        ++count;
    }
    steps[i] = count;
}

int main(int argc, char** argv) {
    int n = parse_size_arg(argc, argv, 1, 1 << 20);
    std::vector<int> steps(n);
    int* d_steps;
    CUDA_CHECK(cudaMalloc(&d_steps, n * sizeof(int)));

    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    KernelTimer timer;
    timer.begin();
    collatz_kernel<<<blocks, threads>>>(d_steps, n);
    CUDA_CHECK_LAUNCH();
    timer.end();

    CUDA_CHECK(cudaMemcpy(steps.data(), d_steps, n * sizeof(int), cudaMemcpyDeviceToHost));
    print_ints(steps);
    CUDA_CHECK(cudaFree(d_steps));
    return 0;
}
