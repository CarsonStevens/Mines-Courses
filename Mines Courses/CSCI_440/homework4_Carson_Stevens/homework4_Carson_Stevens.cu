//
// Created by steve on 4/19/2019.
//

#include <iostream>
#include <cstdlib>
#include <random>
#include <ctime>

using namespace std;

template<unsigned int blockSize>
__global__ void scan_with_addition(unsigned long long int *g_idata, unsigned long long int *g_odata, int n){
    extern __shared__ unsigned long long int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n){
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    } if (blockSize >= 128) {
        if (tid <  64) {
            sdata[tid] += sdata[tid +  64];
        }
        __syncthreads();
    }
    if (tid < 32){
        if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
        if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
        if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
        if (blockSize >=   8) sdata[tid] += sdata[tid +  4];
        if (blockSize >=   4) sdata[tid] += sdata[tid +  2];
        if (blockSize >=   2) sdata[tid] += sdata[tid +  1]; }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}
/*
 * Uses the cache array as a buffer to store the partial results of the last
 * 2 in shared memory. This way threads have access to the last two elements
 * and the array can be built in a hand over hand style, but with layers of depth.
 * each time a element is added, a new thread can be created to process the element
 * behind it. In this manner, it works almost like a pipeline of addition with hand
 * over hand locking on a list, but but multiple lists at an increased depth of one
 * for each new element traversed.
 */

//__global__ void scan_with_addition(unsigned long long int *sum_array, unsigned long long int *A_gpu, int n){
//
//    extern __shared__ unsigned long long int cache[]; // allocated on invocation
//
//    int thid = threadIdx.x;
//    int pout = 0, pin = 1;
//    // load input into shared memory.
//    // This is exclusive scan, so shift right by one and set first elt to 0
//    cache[pout*n + thid] = (thid > 0) ? sum_array[thid-1] : 0;
//    __syncthreads();
//    for (int offset = 1; offset < n; offset *= 2)
//    {
//        pout = 1 - pout; // swap double buffer indices
//        pin = 1 - pout;
//        if (thid >= offset) {
//            cache[pout * n + thid] += cache[pin * n + thid - offset];
//        }
//        else {
//            cache[pout * n + thid] = cache[pin * n + thid];
//        }
//        __syncthreads();
//    }
//    A_gpu[thid] = cache[pout*n+thid]; // write output
//}

/*
 * The idea is to build a balanced binary tree on the input data and
 * sweep it to and from the root to compute the prefix sum. The algorithm
 * consists of two phases: the reduce phase (also known as the up-sweep
 * phase) and the down-sweep phase. In the reduce phase we traverse the
 * tree from leaves to root computing partial sums at internal nodes of
 * the tree, as shown in Figure 2. This is also known as a parallel reduction,
 * because after this phase, the root node (the last node in the array)
 * holds the sum of all nodes in the array. In the down-sweep phase,
 * we traverse back up the tree from the root, using the partial sums to
 * build the scan in place on the array using the partial sums computed
 * by the reduce phase. Only works with arrays up to size 1024 on GX80;
 */

//__global__ void scan_with_addition(int *sum_array, int *A_gpu, int n){
//
//    extern __shared__ int temp[];
//    int thIdx = threadIdx.x;
//    int offset = 1;
//
//    //load input into shared memory
//    temp[2*thIdx] = sum_array[2*thIdx];
//    temp[2*thIdx+1] = sum_array[2*thIdx+1];
//
//    //build sum inplace up the tree
//    for(int d = n>>1; d > 0; d >>= 1){
//        __syncthreads();
//
//        if(thIdx < d){
//            int ai = offset * (2*thIdx+1)-1;
//            int bi = offset * (2*thIdx+2)-1;
//
//            temp[bi] += temp[ai];
//        }
//        offset *= 2;
//    }
//    // clear the last element
//    if(thIdx == 0){ temp[n-1] = 0; }
//    for(int d = 1; d < n; d *= 2){
//        offset >>= 1;
//        __syncthreads();
//        if(thIdx < d){
//            int ai = offset *(2*thIdx+1)-1;
//            int bi = offset *(2*thIdx+2)-1;
//
//            int t = temp[ai];
//            temp[ai] = temp[bi];
//            temp[bi] = t;
//        }
//    }
//    __syncthreads();
//
//    A_gpu[2*thIdx] = temp[2*thIdx];
//    A_gpu[2*thIdx+1] = temp[2*thIdx+1];
//}



int main(int argc, char* argv[]) {
    ///////////////////////////////////////////
    //SETUP
    ///////////////////////////////////////////
    srand(time(0));
    int N = atoi(argv[1]);
    unsigned long long int sum_array[N];
    unsigned long long int A_cpu[N];
    unsigned long long int A_gpu[N];
    unsigned long long int *dev_sum_array;
    unsigned long long int *dev_A_gpu;


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
    cudaMalloc((void **)&dev_sum_array, sizeof(unsigned long long int)*N);
    cudaMalloc((void **)&dev_A_gpu, sizeof(unsigned long long int)*N);
    cudaMemcpy(dev_sum_array, sum_array, sizeof(unsigned long long int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A_gpu, A_gpu, sizeof(unsigned long long int)*N, cudaMemcpyHostToDevice);

    int blockDim = 512;
    int gridSize = (N+blockSize -1)/blockDim;

    // Establish thread and block size
//    int blockSize;
    //int minGridSize;
//    //int gridSize;
//
//    //Optimization function
    //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockDim, scan_with_addition, 0, N);
//
//    // Round up according to array size
//    //gridSize = (N + blockSize - 1) / blockSize;
//
//    // Call function
//    // second blockSize for shared memory
    scan_with_addition<<<gridSize, blockDim, blockDim>>>(dev_sum_array, dev_A_gpu, N);
//    cudaDeviceSynchronize();
//    int threads = 1;
//    if(N>=2) threads = 2;
//    if(N>=4) threads = 4;
//    if(N>=8) threads = 8;
//    if(N>=16) threads = 16;
//    if(N>=32) threads = 32;
//    if(N>=64) threads = 64;
//    if(N>=128) threads = 128;
//    if(N>=256) threads = 256;
//    if(N>=516) threads = 516;
//    switch (threads) {
//        case 512:
//            reduce5<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
//            break;
//        case 256:
//            reduce5<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
//            break;
//        case 128:
//            reduce5<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
//            break;
//        case 64:
//            reduce5< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
//            break;
//        case 32:
//            reduce5< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
//            break;
//        case 16: reduce5< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
//            break;
//        case  8:
//            reduce5<  8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
//            break;
//        case  4:
//            reduce5<  4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
//            break;
//        case  2:
//            reduce5< 2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
//            break;
//        case  1:
//            reduce5<  1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
//            break;
//    }

    // copy result back
    cudaMemcpy(A_gpu, dev_A_gpu, sizeof(unsigned long long int)*N, cudaMemcpyDeviceToHost);

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
