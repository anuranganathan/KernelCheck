// squares: the smallest possible test. One block of 8 threads; thread t
// writes t * t. Checked against a literal expected output in manifest.json
// instead of a reference function.
#include "kc_common.h"

constexpr int N = 8;

__global__ void squares_kernel(int* out) {
    int t = threadIdx.x;
    out[t] = t * t;
}

int main() {
    std::vector<int> out(N);
    int* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));

    KernelTimer timer;
    timer.begin();
    squares_kernel<<<1, N>>>(d_out);
    CUDA_CHECK_LAUNCH();
    timer.end();

    CUDA_CHECK(cudaMemcpy(out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost));
    print_ints(out);
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
