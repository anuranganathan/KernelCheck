// reduce_float: same tree reduction as reduce_int, but in float32.
// Floating-point addition is NOT associative, and the order in which blocks
// perform their atomicAdd is not fixed, so the result can differ slightly
// between runs and between compiler configurations. Neither is a compiler
// bug; this test shows why a tolerance (2e-5 relative here) is needed.
#include "kc_common.h"

constexpr int BLOCK = 256;

__global__ void reduce_kernel(const float* in, float* total, int n) {
    __shared__ float partial[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    partial[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(total, partial[0]);
}

int main(int argc, char** argv) {
    int n = parse_size_arg(argc, argv, 1);
    std::vector<float> values(n);
    for (int i = 0; i < n; ++i) values[i] = 1.0f / (1 + (i % 1000));

    float *d_in, *d_total;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, values.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_total, 0, sizeof(float)));   // all-zero bytes == 0.0f

    const int blocks = (n + BLOCK - 1) / BLOCK;
    KernelTimer timer;
    timer.begin();
    reduce_kernel<<<blocks, BLOCK>>>(d_in, d_total, n);
    CUDA_CHECK_LAUNCH();
    timer.end();

    float total = 0.0f;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));
    printf("%.9g\n", total);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_total));
    return 0;
}
