//
// Created by steve on 4/19/2019.
//

#include <iostream>
#include <cstdlib>
#include <random>
#include <ctime>

using namespace std;

//__device__ bool lastBlock(int* counter) {
//    __threadfence(); //ensure that partial result is visible by all blocks
//    int last = 0;
//    if (threadIdx.x == 0){
//        last = atomicAdd(counter, 1);
//    }
//    return __syncthreads_or(last == gridDim.x-1);
//}
//
//__global__ void scan_with_addition(const int N, const int* sum_array, const int* A_gpu, int* lastBlockCounter) {
//
//    int thIdx = threadIdx.x;
//    int gthIdx = thIdx + blockIdx.x*blockDim.x;
//    const int gridSize = blockDim.x*gridDim.x;
//    int sum = 0;
//    for (int i = gthIdx; i < N; i += gridSize){
//        sum += sum_array[i];
//    }
//
//    __shared__ int shArr[N];
//    shArr[thIdx] = sum;
//    __syncthreads();
//    for (int size = blockDim.x/2; size>0; size/=2) { //uniform
//        if (thIdx<size){
//            shArr[thIdx] += shArr[thIdx+size];
//        }
//        __syncthreads();
//    }
//    if(thIdx == 0){
//        A_gpu[blockIdx.x] = shArr[0];
//    }
//
//    if (lastBlock(lastBlockCounter)) {
//        shArr[thIdx] = thIdx<gridSize ? A_gpu[thIdx] : 0;
//        __syncthreads();
//        for (int size = blockDim.x/2; size>0; size/=2) { //uniform
//            if (thIdx<size){
//                shArr[thIdx] += shArr[thIdx+size];
//            }
//            __syncthreads();
//        }
//        if (thIdx == 0){
//            A_gpu[0] = shArr[0];
//        }
//
//    }
//}
//__device__ int sumCommSingleWarp(volatile int* shArr) {
//    int idx = threadIdx.x % warpSize; //the lane index in the warp
//    if (idx<16) {
//        shArr[idx] += shArr[idx+16];
//        shArr[idx] += shArr[idx+8];
//        shArr[idx] += shArr[idx+4];
//        shArr[idx] += shArr[idx+2];
//        shArr[idx] += shArr[idx+1];
//    }
//    return shArr[0];
//}
/*
 * The argument &r[idx & ~(warpSize-1)] is basically r + warpIdx*32.
 * This effectively splits the r array into chunks of 32 elements,
 * and each chunk is assigned to separate warp.
 */
//__global__ void sumCommSingleBlockWithWarps(const int *a, int *out) {
//    int idx = threadIdx.x;
//    int sum = 0;
//    for (int i = idx; i < arraySize; i += blockSize)
//        sum += a[i];
//    __shared__ int r[blockSize];
//    r[idx] = sum;
//    sumCommSingleWarp(&r[idx & ~(warpSize-1)]);
//    __syncthreads();
//    if (idx<warpSize) { //first warp only
//        r[idx] = idx*warpSize<blockSize ? r[idx*warpSize] : 0;
//        //sumCommSingleWarp(r);
//        int idx = threadIdx.x % warpSize; //the lane index in the warp
//        if (idx<16) {
//            shArr[idx] += shArr[idx+16];
//            shArr[idx] += shArr[idx+8];
//            shArr[idx] += shArr[idx+4];
//            shArr[idx] += shArr[idx+2];
//            shArr[idx] += shArr[idx+1];
//        }
//        if (idx == 0)
//            *out = r[0];
//    }
//}

__global__ void scan_with_addition(int *sum_array, int *A_gpu, int n){

    extern __shared__ int temp[];
    int thIdx = threadIdx.x;
    int offset = 1;

    //load input into shared memory
    temp[2*thIdx] = sum_array[2*thIdx];
    temp[2*thIdx+1] = sum_array[2*thIdx+1];

    //build sum inplace up the tree
    for(int d = n>>1; d > 0; d >>= 1){
        __syncthreads();

        if(thIdx < d){
            int ai = offset * (2*thIdx+1)-1;
            int bi = offset * (2*thIdx+2)-1;

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    // clear the last element
    if(thIdx == 0){ temp[n-1] = 0; }
    for(int d = 1; d < n; d *= 2){
        offset >>= 1;
        __syncthreads();
        if(thIdx < d){
            int ai = offset *(2*thIdx+1)-1;
            int bi = offset *(2*thIdx+2)-1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = t;
        }
    }
    __syncthreads();

    A_gpu[2*thIdx] = temp[2*thIdx];
    A_gpu[2*thIdx+1] = temp[2*thIdx+1];
}

//static const int arraySize = 10000;
//static const int blockSize = 1024;
//
//__global__ void scan_with_addition(int N, const int *a, int *out) {
//    int idx = threadIdx.x;
//    int sum = 0;
//    for (int i = idx; i < N; i += blockDim.x)
//        sum += a[i];
//    __shared__ int r[blockDim.x];
//    r[idx] = sum;
//    __syncthreads();
//    for (int size = blockDim.x/2; size>0; size/=2) { //uniform
//        if (idx<size)
//            r[idx] += r[idx+size];
//        __syncthreads();
//    }
//    if (idx == 0)
//        *out = r[0];
//}
//
//...
//
//sumCommSingleBlock<<<1, blockSize>>>(dev_a, dev_out);


int main(int argc, char* argv[]) {

    srand(time(0));
    int N = (int)argv[1];
    int sum_array[N];
    int A_cpu[N];
    int A_gpu[N];
    int *dev_sum_array[N];
    int *dev_A_gpu[N];

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


    cudaMalloc((void **)&dev_sum_array, sizeof(int)*N);
    cudaMalloc((void **)&dev_A_gpu, sizeof(int)*N);

    // copy data to device
    cudaMemcpy(dev_sum_array, sum_array, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A_gpu, A_gpu, sizeof(int)*N, cudaMemcpyHostToDevice);


    // Establish thread and block size
//    int minGridSize;
//    int blockSize;
//    int gridSize;

    int blockSize;
    int minGridSize = N;
    //int gridSize = 24;
    //int* dev_lastBlockCounter;
    //cudaMalloc((void**)&dev_lastBlockCounter, sizeof(int));
    //cudaMemset(dev_lastBlockCounter, 0, sizeof(int));

    //Optimization function
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scan_with_addition, 0, N);

    // Round up according to array size
    //gridSize = (N + blockSize - 1) / blockSize;

    // Call function
    // second blockSize for shared memory
    scan_with_addition<<<1, blockSize, blockSize>>>(dev_sum_array, dev_A_gpu, N);
    cudaDeviceSynchronize();

    // copy result back
    cudaMemcpy(A_gpu, dev_A_gpu, sizeof(int)*N, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(dev_sum_array);
    cudaFree(dev_A_gpu);

    cout << ">>>\tTESTING RESULTS BY COMPARISION\t<<<" << endl << endl;
    bool check = true;
    int break_index = 0;
    for(int i = 0; i < N; i++){
        if(A_gpu[i] != A_cpu[i]){
            check = false;
            break_index = i;
            break;
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
