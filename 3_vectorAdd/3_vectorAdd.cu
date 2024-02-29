#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef float FLOAT;

/* CUDA kernel function */
__global__ void vec_add(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    /* 2D grid   实际上是2d的grid和1d的block，
gridDim.x表示的block在2dgrid上的x方向上维度，所以blockIdx.y * gridDim.x可以看作是直到你所在的block前所有行的block，之后加上blockIdx.x
就相当于是block in grid,之后再乘blockDim.x相当于是除了当前thread所在的block之前的所有block中的线程数量，最后加上当前的block中的线程索引就是全局线程索引了*/    
    int idx = (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x);
    /* 1D grid */
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) z[idx] = y[idx] + x[idx];
}

void vec_add_cpu(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    for (int i = 0; i < N; i++) z[i] = y[i] + x[i];
}

int main()
{
    int N = 10000;
    int nbytes = N * sizeof(FLOAT);

    /* 1D block */
    int bs = 256;

    /* 2D grid */
    int s = ceil(sqrt((N + bs - 1.) / bs));
    dim3 grid(s, s);
    /* 1D grid */
    // int s = ceil((N + bs - 1.) / bs);
    // dim3 grid(s);

    FLOAT *dx, *hx;
    FLOAT *dy, *hy;
    FLOAT *dz, *hz;

    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);
    
    /* init time */
    float milliseconds = 0;

    /* alllocate CPU mem */
    hx = (FLOAT *) malloc(nbytes);
    hy = (FLOAT *) malloc(nbytes);
    hz = (FLOAT *) malloc(nbytes);

    /* init */
    for (int i = 0; i < N; i++) {
        hx[i] = 1;
        hy[i] = 1;
    }

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    /* launch GPU kernel */
    vec_add<<<grid, bs>>>(dx, dy, dz, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);  


	/* copy GPU result to CPU */
    cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost);

    /* CPU compute */
    FLOAT* hz_cpu_res = (FLOAT *) malloc(nbytes);
    vec_add_cpu(hx, hy, hz_cpu_res, N);

    /* check GPU result with CPU*/
    for (int i = 0; i < N; ++i) {
        if (fabs(hz_cpu_res[i] - hz[i]) > 1e-6) {
            printf("Result verification failed at element index %d!\n", i);
        }
    }
    printf("Result right\n");
    printf("Mem BW= %f (GB/sec)\n", (float)N*4/milliseconds/1e6);///1.78gb/s
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu_res);
//需要销毁cuda事件 没写上去

// 销毁 CUDA 事件
cudaEventDestroy(start);
cudaEventDestroy(stop);
    return 0;
}
