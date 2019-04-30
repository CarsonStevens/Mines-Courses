//
// Created by steve on 4/19/2019.
//

#include <iostream>
#include <cstdlib>
#include <random>
#include <ctime>

using namespace std;

__global__ void scan_with_addition(unsigned int* g_idata, unsigned int* g_odata, int n){
    extern __shared__ unsigned int smem[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    smem[tid] = (i < n) ? g_idata[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];

//    // perform first level of reduction,
//    // reading from global memory, writing to shared memory
//    unsigned int tid = threadIdx.x;
//    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
//
//    smem[tid] = (i < n) ? g_idata[i] : 0;
//    if (i + blockDim.x < n)
//        smem[tid] += g_idata[i+blockDim.x];
//
//    __syncthreads();
//
//    // do reduction in shared mem
//    for(unsigned int s=blockDim.x/2; s>0; s>>=1)
//    {
//        if (tid < s)
//        {
//            smem[tid] += smem[tid + s];
//        }
//        __syncthreads();
//    }
//
//    // write result for this block to global mem
//    if (tid == 0) g_odata[blockIdx.x] = smem[0];

}




int main(int argc, char* argv[]) {
    ///////////////////////////////////////////
    //SETUP
    ///////////////////////////////////////////
    srand(time(0));
    int N = atoi(argv[1]);
    unsigned int sum_array[N];
    unsigned int A_cpu[N];
    unsigned int A_gpu[N];
    unsigned int *dev_sum_array;
    unsigned int *dev_A_gpu;


    ///////////////////////////////////////////
    // Array Initialization
    ///////////////////////////////////////////

    // Initialize array to be summed
    for(int i = 0; i < N; i++){
        sum_array[i] = rand()%1000 + 1;
    }

    // Compute A_cpu
    A_cpu[0] = 0;
    for(int i = 1; i < N; i++){
        A_cpu[i] = sum_array[i-1] + A_cpu[i-1];
        //cout << A_cpu[i] << endl;
    }


    ///////////////////////////////////////////
    // CUDA
    ///////////////////////////////////////////

    // copy data to device
    cudaMalloc((void **)&dev_sum_array, sizeof(unsigned int)*N);
    cudaMalloc((void **)&dev_A_gpu, sizeof(unsigned int)*N);
    cudaMemcpy(dev_sum_array, sum_array, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A_gpu, A_gpu, sizeof(unsigned int)*N, cudaMemcpyHostToDevice);


    // Round up according to array size
    //gridSize = (N + blockSize - 1) / blockSize;
    int threads = 16;
    int blocks = 2;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = threads * sizeof(unsigned int);

    // Call function
    // second blockSize for shared memory
    scan_with_addition<<< dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
    cudaDeviceSynchronize();

    // copy result back
    cudaMemcpy(A_gpu, dev_A_gpu, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(dev_sum_array);
    cudaFree(dev_A_gpu);


    /////////////////////////////////////////////////
    // TESING/VALIDITY
    /////////////////////////////////////////////////
    cout << ">>>\tTESTING RESULTS BY COMPARISION\t<<<" << endl << endl;
    bool check = true;
    int break_index = 0;
    for(int i = 0; i < N; i++){
        cout << "GPU:\t" << A_gpu[i] << endl << "CPU:\t" << A_cpu[i] << endl << endl;
        if(A_gpu[i] != A_cpu[i]){
            check = false;
            break_index = i;
            //break;
        }
    }
    if(check){
        cout << "Tested arrays are equivalent." << endl;
    }
    else{
        cout << "FAILED @ INDEX: " << break_index << endl;
    }

    return 0;
}
