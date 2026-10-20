// int_divmod: integer division and remainder with negative operands.
// C++ truncates toward zero (-7 / 2 == -3, -7 % 2 == -1); the reference must
// do the same. Divisors are in [-4, 4] with 0 replaced by 5, so there is
// never a division by zero (which would be undefined behaviour).
#include "kc_common.h"

__global__ void divmod_kernel(const int* a, const int* d, int* q, int* r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        q[i] = a[i] / d[i];
        r[i] = a[i] % d[i];
    }
}

int main(int argc, char** argv) {
    int n = parse_size_arg(argc, argv, 1);
    std::vector<int> a, unused, d(n), q(n), r(n);
    make_int_inputs(a, unused, n);
    for (int i = 0; i < n; ++i) {
        d[i] = (i % 9) - 4;
        if (d[i] == 0) d[i] = 5;
    }
    size_t bytes = n * sizeof(int);

    int *d_a, *d_d, *d_q, *d_r;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_d, bytes));
    CUDA_CHECK(cudaMalloc(&d_q, bytes));
    CUDA_CHECK(cudaMalloc(&d_r, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d, d.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    KernelTimer timer;
    timer.begin();
    divmod_kernel<<<blocks, threads>>>(d_a, d_d, d_q, d_r, n);
    CUDA_CHECK_LAUNCH();
    timer.end();

    CUDA_CHECK(cudaMemcpy(q.data(), d_q, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.data(), d_r, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) printf("%d %d\n", q[i], r[i]);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_d));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_r));
    return 0;
}
