//
// Created by steve on 4/19/2019.
//

#include <iostream>
#include <cstdlib>
#include <random>
#include <ctime>

using namespace std;


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



int main(int argc, char* argv[]) {

    srand(time(0));
    int N = atoi(argv[1]);
    int sum_array[N];
    int A_cpu[N];
    int A_gpu[N];
    int *dev_sum_array;
    int *dev_A_gpu;

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
