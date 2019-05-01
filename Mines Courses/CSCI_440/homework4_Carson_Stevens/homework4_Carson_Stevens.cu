//
// Created by steve on 4/19/2019.
//

#include <iostream>
#include <cstdlib>
#include <random>
#include <ctime>
#include <chrono>


using namespace std;


#define thread_num 4
#define block_num 2


__global__ void scan_with_addition(int* sum_array, int* A_gpu, const int N) {
    int tid = threadIdx.x;
    extern __shared__ int temp[];

    int out = 0;
    int in = 1;

    temp[tid] = (tid > 0) ? sum_array[tid-1] : 0;
    __syncthreads();

    for(int offset=1; offset < N; offset *= 2){
        out = 1 - out;
        in = 1 - out;

        if(tid >= offset){
            temp[out*N+tid] = temp[in*N+tid-offset] + temp[in*N+tid];
        }
        else{
            temp[out*N+tid] = temp[in*N+tid];
        }

        __syncthreads();
    }

    A_gpu[tid] = temp[out*N+tid];
}

__global__ void prescan(int *g_odata, int *g_idata, const int N){
    extern  __shared__  int temp[];
    int thid = threadIdx.x;
    int offset = 1;
    temp[2*thid] = g_idata[2*thid];
    temp[2*thid+1] = g_idata[2*thid+1];

    for(int d = N>>1; d > 0; d >>= 1){
        __syncthreads();
        if(thid < d){
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
        }
        offset *= 2;
    }
    if(thid == 0){
        temp[N-1] = 0;
    }
    for(int d = 1; d < N; d *= 2){
        offset >>= 1;
        __syncthreads();
        if(thid < d){
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    g_odata[2*thid] = temp[2*thid];
    g_odata[2*thid+1] = temp[2*thid+1];
}

__global__ void reduce(int *g_idata, int *g_odata, const int n){
    extern __shared__ smem[];
    int *sdata = smem.getPointer();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    if (i + blockSize < n){
        sdata[tid] += g_idata[i+blockSize];
    }

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32){
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; EMUSYNC; }
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

string speedup(double baseline_duration, double duration) {
    double speedup = baseline_duration / duration;
    return to_string(speedup) + " times ";
}

int main(int argc, char *argv[]) {
    ///////////////////////////////////////////
    //SETUP
    ///////////////////////////////////////////
    srand(time(0));
    const int N = atoi(argv[1]);
    int sum_array[N];
    int A_cpu[N];
    int A_gpu[N];
    int *dev_sum_array;
    int *dev_A_gpu;


    ///////////////////////////////////////////
    // Array Initialization
    ///////////////////////////////////////////

    // Initialize array to be summed
    for (int i = 0; i < N; i++) {
        sum_array[i] = rand() % 1000 + 1;
    }

    // Compute A_cpu
    auto start = chrono::high_resolution_clock::now();
    A_cpu[0] = 0;
    for (int i = 1; i < N; i++) {
        A_cpu[i] = sum_array[i - 1] + A_cpu[i - 1];
        //cout << A_cpu[i] << endl;
    }
    auto stop = chrono::high_resolution_clock::now();
    auto baseline = stop - start;


    ///////////////////////////////////////////
    // CUDA
    ///////////////////////////////////////////

    // copy data to device
    cudaMalloc((void **) &dev_sum_array, sizeof(int) * N);
    cudaMalloc((void **) &dev_A_gpu, sizeof(int) * N);
    cudaMemcpy(dev_sum_array, sum_array, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A_gpu, A_gpu, sizeof(int) * N, cudaMemcpyHostToDevice);

//    dim3  blocksize(N);
//    dim3 gridsize(1);

    // Call function
    start = chrono::high_resolution_clock::now();
    //reduce<<< gridsize, blocksize >>>(dev_sum_array,dev_A_gpu);
    //prescan<<< 1, N, 2*N*sizeof(int) >>>(dev_sum_array, dev_A_gpu, N);
    int threads = 512;
    int blocks = 1;
    if (N > 512) {
        threads = 512;
        blocks = (N-1 / threads) + 1;
    }

    if(N > 512) {
        threads = 512;
    } else if (N > 256) {
        threads = 256;
    } else if (N > 128) {
        threads = 128;
    } else if (N > 64) {
        threads = 64;
    } else if (N > 32) {
        threads = 32;
    } else if (N > 16) {
        threads = 16;
    } else if (N > 8) {
        threads = 8;
    } else if (N > 4) {
        threads = 4;
    } else if (N > 2){
        threads = 2;
    } else{
        threads = 1;
    }

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = threads * sizeof(int);

    switch (threads) {

        case 512:
            reduce < 512 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
            break;
        case 256:
            reduce < 256 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
            break;
        case 128:
            reduce < 128 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
            break;
        case 64:
            reduce < 64 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
            break;
        case 32:
            reduce < 32 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
            break;
        case 16:
            reduce < 16 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
            break;
        case 8:
            reduce < 8 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
            break;
        case 4:
            reduce < 4 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
            break;
        case 2:
            reduce < 2 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
            break;
        case 1:
            reduce < 1 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
            break;
    }
    cudaDeviceSynchronize();
    //}


    stop = chrono::high_resolution_clock::now();
    auto real = stop - start;

    // copy result back
    cudaMemcpy(A_gpu, dev_A_gpu, sizeof(int)*N, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(dev_sum_array);
    cudaFree(dev_A_gpu);




    /////////////////////////////////////////////////
    // TESTING/VALIDITY
    /////////////////////////////////////////////////
    cout << ">>>\tTESTING RESULTS BY COMPARISION\t<<<" << endl << endl;
    bool check = true;
    int break_index = 0;
    for(int i = 0; i < N; i++){
        if(A_gpu[i] != A_cpu[i]){
            cout << "GPU:\t" << A_gpu[i] << endl << "CPU:\t" << A_cpu[i] << endl << endl;
            check = false;
            break_index = i;
            break;
        }
    }
    if(check){
        cout << "Tested arrays are equivalent." << endl;
        cout << "\tSpeed up measured at " << speedup(chrono::duration <double, milli> (baseline).count(), chrono::duration <double, milli>
                (real).count()) << "the baseline." << endl;
    }
    else{
        cout << "FAILED @ INDEX: " << break_index << endl;
    }

    return 0;
}
