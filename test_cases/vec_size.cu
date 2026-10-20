// vec_size: c[i] = 3*a[i] - b[i] for a size given on the command line.
// Uses a grid-stride loop with a fixed, small grid, so sizes larger than the
// number of launched threads (e.g. 65537) make each thread handle several
// elements, and tiny sizes (e.g. 1) leave almost every thread idle.
#include "kc_common.h"

__global__ void scale_sub_kernel(const int* a, const int* b, int* c, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        c[i] = 3 * a[i] - b[i];
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

    const int threads = 128;
    const int blocks = 32;   // 4096 threads in total, regardless of n
    KernelTimer timer;
    timer.begin();
    scale_sub_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_LAUNCH();
    timer.end();

    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    print_ints(c);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
