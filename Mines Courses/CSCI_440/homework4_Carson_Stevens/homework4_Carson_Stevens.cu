/*
 * Author: Carson Stevens
 *
 * Date: April 30, 2019
 *
 * CSCI 440 HW4
 *
 * Description: Implement prefix sum scan for an array of size N
 *
 * Comments: Version only works for arrays up to size 1024, BUT
 *           Commented out version attempts to solve this issue.
 *           Also noted that speed up would be better shown with
 *           the larger arrays.
 */

#include <iostream>
#include <cstdlib>
#include <random>
#include <ctime>
#include <chrono>


using namespace std;

/*
 * Working kernel that has no manual parallelization
 */
__global__ void scan_with_addition_naive(int* sum_array, int* A_gpu, const int N){
    A_gpu[0] = 0;
    for(int i = 1; i < N; i++){
        A_gpu[i] = A_gpu[i-1] + sum_array[i-1];
    }
}

/*
 * Kernel that parallelizes the scan/sum, but only works for N <= 1024
 */
__global__ void scan_with_addition(int* sum_array, int* A_gpu, const int N) {

    extern __shared__ int cache[];

    int tid = threadIdx.x;
    int out = 0;
    int in = 1;

    cache[tid] = (tid > 0) ? sum_array[tid - 1] : 0;
    __syncthreads();

    for(int offset=1; offset < N; offset *= 2){
        out = 1 - out;
        in = 1 - out;

        if(tid >= offset){
            cache[out * N + tid] = cache[in * N + tid - offset] + cache[in * N + tid];
        }
        else{
            cache[out * N + tid] = cache[in * N + tid];
        }

        __syncthreads();
    }

    A_gpu[tid] = cache[out * N + tid];
}

/*
 *  Arbitrary sized array implementation (N > 1024). Discussed below in main
 */

//__global__ void reduce(int *sum_array, int *A_gpu, const int N){
//    extern __shared__ smem[];
//    int *block_cache = smem.getPointer();
//
//    // perform first level of reduction,
//    // reading from global memory, writing to shared memory
//    unsigned int tid = threadIdx.x;
//    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
//
//    block_cache[tid] = (i < N) ? sum_array[i] : 0;
//    if (i + blockSize < n){
//        block_cache[tid] += sum_array[i+blockSize];
//    }
//
//    __syncthreads();
//
//    // do reduction in shared mem
//    for(unsigned int s = blockDim.x/2; s > 32; s>>=1){
//        if (tid < s){
//            block_cache[tid] += block_cache[tid + s];
//        }
//        __syncthreads();
//    }
//
//    if (tid < 32){
//        if (blockSize >=  64) { block_cache[tid] += block_cache[tid + 32]; __syncthreads();}
//        if (blockSize >=  32) { block_cache[tid] += block_cache[tid + 16]; __syncthreads();}
//        if (blockSize >=  16) { block_cache[tid] += block_cache[tid +  8]; __syncthreads();}
//        if (blockSize >=   8) { block_cache[tid] += block_cache[tid +  4]; __syncthreads();}
//        if (blockSize >=   4) { block_cache[tid] += block_cache[tid +  2]; __syncthreads();}
//        if (blockSize >=   2) { block_cache[tid] += block_cache[tid +  1]; __syncthreads();}
//    }
//
//    // write result for this block to global mem
//    if (tid == 0){
//        A_gpu[blockIdx.x] = block_cache[0];
//    }
//}

/*
 * Helper function to help print results
 */
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


    // Call function: start clock
    start = chrono::high_resolution_clock::now();

    // Most basic implementation (only works for N up to 1024; blocksize)
    //scan_with_addition_naive<<< 1, N>>>(dev_sum_array, dev_A_gpu, N);

    // More parallelized version, but still only works for N up to 1024; blocksize
    scan_with_addition<<< 1, N, 2*N*sizeof(int) >>>(dev_sum_array, dev_A_gpu, N);
    cudaDeviceSynchronize();

    //Stop the clock
    stop = chrono::high_resolution_clock::now();
    auto real = stop - start;


    /*
     * I understand that the above ones do not work with N > 1024 because each block can only
     * hold twice the number of threads (E.G 512 maximum block size = 1024 threads). Below is
     * how I attempted to fix this problem.
     *
     * Attempt to allow for arbitrary sized array (N > 1024): The problem with the examples
     * above is that they only operate with the length of one block. To make this work for
     * any size array, the original array needs to be parsed up into blocks and each individual
     * block then goes through the reduction. Then after the reduction. The blocks then must
     * undergo their own reduction that adds all of the blocks back into the outputted array.
     *
     * The implementation I tried didn't end up working, but was supposed to take in the size N
     * and then figure out how many blocks would need to be processed. To do this, the equation
     * (N-1)/512 gives the amount of total block iterations there will be. (Since a block's maximum
     * size is 512, it can have 1024 threads). Arrays larger than that should be broken
     * into their own chunks of 1024 or less elements.
     *
     * The application of these different blocks is seen in the dimGrid feature that should be the
     * dimensions of the amount of chunks described in the paragraph above. As far as the function
     * goes, the getPointer allows the different blocks to share the same shared memory resolving
     * memory issues.
     */


//    int threads = 512;
//    int blocks = 1;
//    if (N > 512) {
//        threads = 512;
//        blocks = (N-1 / threads);
//    }
//
//    if(N > 512) {
//        threads = 512;
//    } else if (N > 256) {
//        threads = 256;
//    } else if (N > 128) {
//        threads = 128;
//    } else if (N > 64) {
//        threads = 64;
//    } else if (N > 32) {
//        threads = 32;
//    } else if (N > 16) {
//        threads = 16;
//    } else if (N > 8) {
//        threads = 8;
//    } else if (N > 4) {
//        threads = 4;
//    } else if (N > 2){
//        threads = 2;
//    } else{
//        threads = 1;
//    }
//
//    dim3 dimBlock(threads, 1, 1);
//    dim3 dimGrid(blocks, 1, 1);
//    int smemSize = threads * sizeof(int);
//
//    switch (threads) {
//
//        case 512:
//            reduce < 512 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
//            break;
//        case 256:
//            reduce < 256 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
//            break;
//        case 128:
//            reduce < 128 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
//            break;
//        case 64:
//            reduce < 64 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
//            break;
//        case 32:
//            reduce < 32 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
//            break;
//        case 16:
//            reduce < 16 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
//            break;
//        case 8:
//            reduce < 8 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
//            break;
//        case 4:
//            reduce < 4 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
//            break;
//        case 2:
//            reduce < 2 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
//            break;
//        case 1:
//            reduce < 1 ><<<dimGrid, dimBlock, smemSize >>>(dev_sum_array, dev_A_gpu, N);
//            break;
//    }


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
        cout << "\t> Tested arrays are equivalent." << endl;
        cout << "\t\t> Speed up measured at " << speedup(chrono::duration <double, milli> (baseline).count(), chrono::duration <double, milli>
                (real).count()) << "the baseline." << endl << endl;
    }
    else{
        cout << "FAILED @ INDEX: " << break_index << endl << endl;
    }

    return 0;
}
