#include <CL/sycl.hpp>

using namespace sycl;

int main() {
    queue q;

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    float* A = malloc_shared<float>(M * K, q);
    float* B = malloc_shared<float>(K * N, q);
    float* C = malloc_shared<float>(M * N, q);

    // Initialize matrices A and B
    for (int i = 0; i < M * K; i++) A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = rand() / (float)RAND_MAX;

    q.parallel_for(range<2>{M, N}, [=](id<2> idx) {
        int i = idx[0];
        int j = idx[1];
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }).wait();

    // Check results
    // ...

    free(A, q);
    free(B, q);
    free(C, q);

    return 0;
}
