/*
 * Author: Carson Stevens
 * Date: April 2, 2019
 * Description: 	1)	Read in input from file
 *                  2)	Perform Sparse Matrix Vector
 *                      Multiplication on read in matrix
 */


#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>


using namespace std;

__global__ void spmv(const int num_rows, const int* ptr, const int* indices,
                     const float* data, const float* mult_data, float* result){

    // WORKING
    // Cache the rows of mult_data[] corresponding to this block.
    extern __shared__ float cache[];

    int block_begin = blockIdx.x * blockDim.x;
    int block_end = block_begin + blockDim.x;
    int row = block_begin + threadIdx.x;

    // Fetch and cache window of mult_data[].
    if( row < num_rows){
        cache[threadIdx.x] = mult_data[row];
    }
    __syncthreads();

    if(row < num_rows){

        int row_begin = ptr[row];
        int row_end = ptr[row+1];
        float mult_temp = 0;
        float sum = 0 ;

        for(int col = row_begin; col < row_end; ++col){
            int j = indices[col];

            // Fetch updated mult_temp from our cache when possible
            if(j >= block_begin && j < block_end) {
                mult_temp = cache[j-block_begin];
            }
            else {
                mult_temp = mult_data[j];
            }

            //Sum the dot product
            sum += data[col] * mult_temp;
        }

        //Send updated sum to the result matrix when all threads done.
        result[row] = sum;
    }

    /* WORKING: No optimization
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0;
        int row_start = ptr[row];
        int row_end = ptr[row + 1];

        // Compute sum per thread
        for (int i = row_start; i < row_end; i++) {
            dot += data[i] * mult_data[indices[i]];
        }

        result[row] = dot;
    }
    */

    /* NOT WORKING
    extern __shared__ float vals[];

    // global thread indexes
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    // global warp index
    int warp_id = thread_id/32;

    // thread index inside warp
    int lane = thread_id % 32;

    // one warp per row
    int row = warp_id;

    if (row < num_rows){

        int row_start = ptr[row];
        int row_end = ptr[row+1];

        // Compute sum per thread
        for(int i = row_start + lane; i < row_end; i++){
            vals[threadIdx.x] += data[i] * mult_data[indices[i]];
        }

        //Synchronization for shared memory
        if(lane < 16){
            vals[threadIdx.x] += vals[threadIdx.x + 16];
        }
        if(lane < 8){
            vals[threadIdx.x] += vals[threadIdx.x + 8];
        }
        if(lane < 4){
            vals[threadIdx.x] += vals[threadIdx.x + 4];
        }
        if(lane < 2){
            vals[threadIdx.x] += vals[threadIdx.x + 2];
        }
        if(lane < 1){
            vals[threadIdx.x] += vals[threadIdx.x + 1];
        }

        // first thread writes the result
        if(lane == 0){
            result[row] += vals[threadIdx.x];
        }
    }
    */

}

int main(int argc, char* argv[]){

    srand(0);
    //Load file
    //Declare ifstream object for .txt file parsing.
    //open the file from which to read the data
    ifstream file(argv[1]);
    if (!file) {
        cerr << "Error opening input:\t" << argv[1] << endl;
        return (1);
    }

    int num_cols;
    int num_rows;
    int number_of_entries;
    file >> num_cols >> num_rows >> number_of_entries;

    // Define matrices for computation
    int column[number_of_entries];
    int row_ptr[num_rows+1];
    float data[number_of_entries];
    float mult_data[num_cols];
    float result[num_rows];

    int* dev_columns;
    int* dev_row_ptr;
    float* dev_data;
    float* dev_mult_data;
    float* dev_result;

    //Initialize the result array to 0
    for(int i = 0; i < num_rows; i++){
        result[i] = 0.0;
    }
    //Initialize the multiply vector with data
    for(int i = 0; i < num_cols; i++){
        mult_data[i] = (rand() % 100000000) / 111111111.0;
    }

    //Markers for keeping track of data
    int counter = 0;
    int ptr_counter = 0;
    int last_row = 0;
    int current_row = 0;

    while (counter < number_of_entries){

        //Read in values
        file >> current_row >> column[counter] >> data[counter];

        //-1 to change read in form to zero indexing
        column[counter]--;

        //Check to see if new entry for row_ptr
        if(current_row != last_row){
            row_ptr[current_row-1] = counter;
            ptr_counter++;
            last_row = current_row;
        }
        counter++;
    }
    row_ptr[num_rows] = number_of_entries;
    file.close();

    int size_int = sizeof(int);
    int size_float = sizeof(float);

    // Allocate memory on GPU
    cudaMalloc((void **)&dev_columns, size_int*number_of_entries);
    cudaMalloc((void **)&dev_row_ptr, size_int*(num_rows+1));
    cudaMalloc((void **)&dev_data, size_float*number_of_entries);
    cudaMalloc((void **)&dev_mult_data, size_float*num_cols);
    cudaMalloc((void **)&dev_result, size_float*num_rows);

    // copy data to device
    cudaMemcpy(dev_columns, column, size_int*number_of_entries, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_row_ptr, row_ptr, size_int*(num_rows+1), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data, data, size_float*number_of_entries, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mult_data, mult_data, size_float*num_cols, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_result, result, size_float*num_rows, cudaMemcpyHostToDevice);

    // Establish thread and block size
    int minGridSize;
    int blockSize;
    int gridSize;
    //Optimization function
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, spmv, 0, number_of_entries);

    // Round up according to array size
    gridSize = (number_of_entries + blockSize - 1) / blockSize;
    // Call function
    // second blockSize for shared memory
    spmv<<<gridSize, blockSize, blockSize>>>(num_rows, dev_row_ptr, dev_columns, dev_data, dev_mult_data, dev_result);

    // copy result back
    cudaMemcpy(result, dev_result, size_float*num_rows, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(dev_columns);
    cudaFree(dev_row_ptr);
    cudaFree(dev_data);
    cudaFree(dev_mult_data);
    cudaFree(dev_result);

    //To Print result
    cout << "[ ";
    for(int i = 0; i < num_rows; i++){
        cout << result[i] << " ";
    }
    cout << "]" << endl;

}