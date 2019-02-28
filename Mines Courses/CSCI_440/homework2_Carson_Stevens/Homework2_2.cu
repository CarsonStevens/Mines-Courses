/*
 * Author: Carson Stevens
 * Date: February 27, 2019
 * Description: 	1)	Read in input from m1.txt into a matrix
 *                  2)	Use CUDA to return the same matrix transposed
 */


#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;

__global__ void transpose_matrix(int *transpose, int *matrix, int width, int height){

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    //int width_block = dimThreadsPerBlock.x * blockDim.x;

    transpose[x*height + y] = matrix[y*width + x];

    // Mapping for transpose
//    for (int j = 0; j < blockDim.x; j+= width) {
//        transpose[x * width_block + (y + j)] = matrix[(y + j) * width_block + x];
//    }
}


int main( int argc, char* argv[] ) {

    //Load file
    //Declare ifstream object for .txt file parsing.
    //open the file from which to read the data
    ifstream data(argv[1]);
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

    // Define matrices for original and transpose
    int dev_matrix[width][height];
    int dev_transpose[height][width];
    int *transpose;
    int *matrix;

    //Read values into the matrix
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            data >> entry;
            matrix[width][height] = entry;
        }
    }

    int size = sizeof(int);

    //Allocate CUDA space
    cudaMalloc((void **) &matrix, width * height * size);
    cudaMalloc((void **) &transpose, width * height * size);

    cudaMemcpy(dev_matrix, matrix, width * height * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_transpose, transpose, width * height * size, cudaMemcpyHostToDevice);

    dim3 dimThreadsPerBlock(width, height, 1);
    dim3 numBlock(((width+dimThreadsPerBlock.x-1)/dimThreadsPerBlock.x), ((height+dimThreadsPerBlock.y-1)/dimThreadsPerBlock.y), 1);

    transpose_matrix<<<numBlock, dimThreadsPerBlock>>>(transpose, matrix, width, height);
    cudaMemcpy(dev_transpose, transpose, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_matrix);
    cudaFree(dev_transpose);

    //Print results to output
    cout << width << " " << height << endl;
    for(int i = 0; i < height; i ++){
        for(int j = 0; j < width; j++){
            cout << dev_transpose[i][j];
            if(i != width-1){
                cout << " ";
            }
        }
        cout << endl;
    }
}


