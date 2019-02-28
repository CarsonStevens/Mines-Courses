/*
 * Author: Carson Stevens
 * Date: February 27, 2019
 * Description: 	1)	Read in input from m1.txt into a matrix
 *                  2)	Use CUDA to count the number of 1s in the matrix
 */

#include <iostream>
#include <sstream>
#include <fstream>

__global__ void find_ones(int *matrix, int *result, int width, int height);

int main( int argc, char* argv[] ) {
    //grab file name from input
    string file = argv[1];

    //Load file
    //Declare ifstream object for .txt file parsing.
    //open the file from which to read the data
    ifstream data(file);
    if (!data) {
        cerr << "Error opening input." << endl;
        return (1);
    }
    int width = 0;
    int height = 0;
    int entry = 0;
    int result = 0;
    int *answer;

    data >> width >> height;

    int matrix[width][height];
    int *gpu_matrix;

    //Read values into the matrix
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            data >> entry;
            matrix[width][height] = entry;
        }
    }

    int size = width * height * sizeof(int);

    //Allocate CUDA space
    cudaMalloc((void **) &gpu_matrix, size);
    cudaMalloc((void **) &answer, size);
    cudaMalloc((void **) width, size);
    cudaMalloc((void **) height, size);

    //Move to GPU
    cudaMemcpy(gpu_matrix, matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(answer, result, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(1, 1);
    dim3 dimBlock(width, height);

    find_ones << < dimGrid, dimBlock >>> (gpu_matrix, answer, width, height);

    //return to memory
    cudaMemcpy(result, cudaMemcpyDeviceToHost);
    cudaFree(gpu_matrix);
    cudaFree(answer);


    //print answer
    cout << result << endl;
}

__global__ void find_ones(int *matrix, int *result, int width, int height){
    result = 0;

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    for(int k = 0; k < width; k++){
        for(int l = 0; l < height; l++){
            if(matrix[k][l] == 1){
                result++;
            }
        }
    }
}




