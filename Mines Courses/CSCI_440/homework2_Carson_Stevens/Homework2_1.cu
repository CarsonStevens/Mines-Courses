/*
 * Author: Carson Stevens
 * Date: February 27, 2019
 * Description: 	1)	Read in input from m1.txt into a matrix
 *                  2)	Use CUDA to count the number of 1s in the matrix
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

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

    int size = sizeof(int);

    //Allocate CUDA space
    cudaMalloc((void **) &gpu_matrix, width*height*size);
    cudaMalloc((void **) &answer, 1*size);


    //Move to GPU
    cudaMemcpy(gpu_matrix, matrix, width*height*size, cudaMemcpyHostToDevice);
    cudaMemcpy(answer, &result, size, cudaMemcpyHostToDevice);

    dim3 dimThreadsPerBlock(width, height, 1);
    dim3 numBlock(((width+dimThreadsPerBlock.x-1)/threadsPerBlock.x), ((height+dimThreadsPerBlock.y-1)/threadsPerBlock.y), 1);

    find_ones << < numBlock, dimThreadPerBlock>>> (gpu_matrix, answer, width, height);

    //return to memory
    cudaMemcpy(result, cudaMemcpyDeviceToHost);
    cudaFree(gpu_matrix);
    cudaFree(answer);


    //print answer
    cout << result << endl;
}

__global__ void find_ones(int *matrix, int *result, int width, int height){

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    for(int k = 0; k < width; k++){
        for(int l = 0; l < height; l++){
            if(matrix[k][l] == 1){
                atomicAdd(result,1);
            }
        }
    }
}




