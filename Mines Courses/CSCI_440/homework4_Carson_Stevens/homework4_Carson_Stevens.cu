//
// Created by steve on 4/19/2019.
//

#include <iostream>
#include <cstdlib>
#include <random>
#include <ctime>

using namespace std;

__global__ void scan_with_addition(const int N, const int* sum_array, const int* A_gpu) {

    int index = threadIdx.x;
    int gthIdx = index + blockIdx.x * blockSize;
    const int gridSize = blockSize * gridDim.x;
    int sum = 0;
    for (int i = gthIdx; i < N; i += gridSize){
        sum += sum_array[i];
    }
    __shared__ int cache[blockSize];
    cache[index] = sum;
    __syncthreads();
    for (int size = blockSize / 2; size > 0; size /= 2) { //uniform
        if (index < size)
            cache[index] += cache[index + size];
        __syncthreads();
    }
    if (index == 0){
        A_gpu[blockIdx.x] = cache[0];
    }
}

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
    int minGridSize;
    int blockSize;
    int gridSize;
    //Optimization function
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, scan_with_addition, 0, N);

    // Round up according to array size
    gridSize = (N + blockSize - 1) / blockSize;
    // Call function
    // second blockSize for shared memory
    scan_with_addition<<<gridSize, blockSize, blockSize>>>(N, dev_sum_array, dev_A_gpu);


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
