// reduce_int: sum of n integers using a shared-memory tree reduction.
// Each block reduces BLOCK values in shared memory (halving the active threads
// each step, with __syncthreads() between steps), then thread 0 adds the
// block's partial sum into the global total with atomicAdd. Integer addition
// is associative, so the result is exact whatever order the blocks finish in.
#include "kc_common.h"

constexpr int BLOCK = 256;   // must be a power of two for this reduction

__global__ void reduce_kernel(const int* in, int* total, int n) {
    __shared__ int partial[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    partial[tid] = (i < n) ? in[i] : 0;   // threads past the end contribute 0
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();   // reached by every thread in the block
    }
    if (tid == 0) atomicAdd(total, partial[0]);
}

int main(int argc, char** argv) {
    int n = parse_size_arg(argc, argv, 1);
    std::vector<int> values(n);
    for (int i = 0; i < n; ++i) values[i] = i % 10;   // sum stays far below INT_MAX

    int *d_in, *d_total;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, values.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_total, 0, sizeof(int)));

    const int blocks = (n + BLOCK - 1) / BLOCK;
    KernelTimer timer;
    timer.begin();
    reduce_kernel<<<blocks, BLOCK>>>(d_in, d_total, n);
    CUDA_CHECK_LAUNCH();
    timer.end();

    int total = 0;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost));
    printf("%d\n", total);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_total));
    return 0;
}
