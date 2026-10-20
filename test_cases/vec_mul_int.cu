// int_mul: element-wise integer multiplication, c[i] = a[i] * b[i].
// Inputs are small (|a| <= 50, |b| <= 18), so the product never overflows.
#include "kc_common.h"

__global__ void mul_kernel(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

int main(int argc, char** argv) {
    int n = parse_size_arg(argc, argv, 1);
    std::vector<int> a, b, c(n);
    make_int_inputs(a, b, n);
    size_t bytes = n * sizeof(int);

    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    KernelTimer timer;
    timer.begin();
    mul_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_LAUNCH();
    timer.end();

    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    print_ints(c);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
