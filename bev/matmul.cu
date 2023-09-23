
#define NUM_THREADS 32
#define TILE_SIZE 1024 // NUM_THREADS * NUM_THREADS

typedef unsigned char uchar;

template <typename T>
__global__ void cudaMatrixMul(T *a, T *b, T *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col > k || row > m)
        return;

    __shared__ int tileA[TILE_SIZE];
    __shared__ int tileB[TILE_SIZE];

    T sum = 0;

    for (int i = 0; i < n; i += blockDim.x)
    {
        tileA[threadIdx.y * blockDim.x + threadIdx.x] = a[row * n + i + threadIdx.x];
        tileB[threadIdx.y * blockDim.x + threadIdx.x] = b[(i + threadIdx.y) * k + col];

        __syncthreads();

        for (int j = 0; j < blockDim.x; j++)
        {
            sum += tileA[threadIdx.y * blockDim.x + j] * tileB[j * blockDim.x + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < k)
        c[row * k + col] = sum;
}

int padDimensionSize(int dim)
{
    return dim + NUM_THREADS - dim % NUM_THREADS;
}

template <typename T>
void matmul(T *m1, T *m2, T *result, int m, int n, int k)
{
    T *cudaM1;
    T *cudaM2;
    T *cudaResult;

    size_t sizeM1 = m * n * sizeof(T);
    size_t sizeM2 = n * k * sizeof(T);
    size_t sizeResult = m * k * sizeof(T);

    cudaMalloc(&cudaM1, sizeM1);
    cudaMalloc(&cudaM2, sizeM2);
    cudaMalloc(&cudaResult, sizeResult);

    cudaMemcpy(cudaM1, m1, sizeM1, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaM2, m2, sizeM2, cudaMemcpyHostToDevice);

    int BLOCKS_X = (k / NUM_THREADS);
    int BLOCKS_Y = (m / NUM_THREADS);

    if (BLOCKS_X == 0)
        BLOCKS_X = 1;

    if (BLOCKS_Y == 0)
        BLOCKS_Y = 1;

    dim3 threads(NUM_THREADS, NUM_THREADS);
    dim3 blocks(BLOCKS_X, BLOCKS_Y);

    cudaMatrixMul<<<blocks, threads>>>(cudaM1, cudaM2, cudaResult, m, n, k);

    cudaMemcpy(result, cudaResult, sizeResult, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(cudaM1);
    cudaFree(cudaM2);
    cudaFree(cudaResult);
}

template void matmul(float *m1, float *m2, float *result, int m, int n, int k);
template void matmul(int *m1, int *m2, int *result, int m, int n, int k);
template void matmul(uchar *m1, uchar *m2, uchar *result, int m, int n, int k);
