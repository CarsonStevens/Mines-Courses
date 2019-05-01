//
// Created by steve on 4/19/2019.
//

#include <iostream>
#include <cstdlib>
#include <random>
#include <ctime>
#include <chrono>

using namespace std;

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

__global__ void prescan(int *g_odata, int *g_idata, int n){
    extern __shared__ int temp[];// allocated on invocation


    int thid = threadIdx.x;
    int offset = 1;

    temp[2*thid] = g_idata[2*thid]; // load input into shared memory
    temp[2*thid+1] = g_idata[2*thid+1];
    // build sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1){


        __syncthreads();

        if (thid < d){
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0) { temp[n - 1] = 0; } // clear the last element

    // traverse down tree & build scan
    for (int d = 1; d < n; d *= 2){
        offset >>= 1;
        __syncthreads();
        if (thid < d){
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    g_odata[2*thid] = temp[2*thid]; // write results to device memory
    g_odata[2*thid+1] = temp[2*thid+1];
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
    cudaMemcpy(dev_sum_array, sum_array, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A_gpu, A_gpu, sizeof(int)*N, cudaMemcpyHostToDevice);

    dim3  blocksize(N);
    dim3 gridsize(1);
    // Call function
    start = chrono::high_resolution_clock::now();
    //reduce<<< gridsize, blocksize >>>(dev_sum_array,dev_A_gpu);
    prescan<<< gridsize, blocksize, 2*N*sizeof(int) >>>(dev_sum_array, dev_A_gpu, N);
    cudaDeviceSynchronize();
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
