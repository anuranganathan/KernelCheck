// float_arith: single-precision arithmetic, out[i] = x*y + x/y - 0.5*y.
// The compiler may fuse x*y + ... into one FMA instruction (nvcc does this by
// default), which rounds once instead of twice. That is allowed by the CUDA
// floating-point rules, so results are checked with a tolerance.
#include "kc_common.h"

__global__ void arith_kernel(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * y[i] + x[i] / y[i] - 0.5f * y[i];
}

int main(int argc, char** argv) {
    int n = parse_size_arg(argc, argv, 1);
    std::vector<float> x(n), y(n), out(n);
    // Same formulas as float_inputs() in reference/references.py; every value
    // is exactly representable in float32. y >= 1, so there is no division by 0.
    for (int i = 0; i < n; ++i) {
        x[i] = (i % 100) * 0.25f - 12.5f;
        y[i] = ((i * 3) % 50) * 0.5f + 1.0f;
    }
    size_t bytes = n * sizeof(float);

    float *d_x, *d_y, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    KernelTimer timer;
    timer.begin();
    arith_kernel<<<blocks, threads>>>(d_x, d_y, d_out, n);
    CUDA_CHECK_LAUNCH();
    timer.end();

    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    print_floats(out);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
