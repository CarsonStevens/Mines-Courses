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

using namespace std;

__global__ void find_ones(int *matrix, int *answer, int width, int height){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(matrix[row*width+col] == 1){
        atomicAdd(answer, 1);
    }
}

int main( int argc, char* argv[] ) {

    //Load file
    //Declare ifstream object for .txt file parsing.
    //open the file from which to read the data
    ifstream file(argv[1]);
    if (!file) {
        cerr << "Error opening input." << endl;
        return (1);
    }

    // for reading in values from .txt
    int width = 0;
    int height = 0;
    int result = 0;
    int *answer;

    file >> width >> height;
    int matrix[width][height];
    int *gpu_matrix;


    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            file >> matrix[i][j];
            //cout << "(" << i << "," << j << ")\t" << dev_matrix[i][j] << endl;
        }
    }
    file.close();

    int size = sizeof(int);

    //Allocate CUDA space
    cudaMalloc((void **) &gpu_matrix, width*height*size);
    cudaMalloc((void **) &answer, 1*size);


    //Move to GPU
    cudaMemcpy(gpu_matrix, matrix, width*height*size, cudaMemcpyHostToDevice);
    cudaMemcpy(answer, &result, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(width, height, 1);
    dim3 numBlock(((width+dimBlock.x-1)/dimBlock.x), ((height+dimBlock.y-1)/dimBlock.y), 1);

    find_ones <<<numBlock, dimBlock>>> (gpu_matrix, answer, width, height);

    //return to memory
    cudaMemcpy(&result, answer, 1*size, cudaMemcpyDeviceToHost);

    //print answer
    cout << result << endl;
    cudaFree(gpu_matrix);
    cudaFree(answer);
}






