// saxpy: the classic BLAS operation y = alpha * x + y, updating y in place.
#include "kc_common.h"

__global__ void saxpy_kernel(float alpha, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = alpha * x[i] + y[i];
}

int main(int argc, char** argv) {
    int n = parse_size_arg(argc, argv, 1);
    const float alpha = 2.5f;
    std::vector<float> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = (i % 64) * 0.125f;
        y[i] = 1.0f - (i % 32) * 0.0625f;
    }
    size_t bytes = n * sizeof(float);

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    KernelTimer timer;
    timer.begin();
    saxpy_kernel<<<blocks, threads>>>(alpha, d_x, d_y, n);
    CUDA_CHECK_LAUNCH();
    timer.end();

    CUDA_CHECK(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    print_floats(y);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
