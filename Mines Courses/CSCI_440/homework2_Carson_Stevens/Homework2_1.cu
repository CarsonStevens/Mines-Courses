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

__global__ void find_ones(int *matrix, int *answer, int width){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    cout << "iteration" << endl;
    if(matrix[row*width + col] == 1){
        atomicAdd(answer, 1);
    }

//    for(int k = 0; k < width; k++){
//        for(int l = 0; l < height; l++){
//            if(matrix[row][col] == 1){
//                atomicAdd(result,1);
//            }
//        }
//    }
}

int main( int argc, char* argv[] ) {
    //grab file name from input
//    string file = argv[1];

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

    int matrix[width][height];
    int *gpu_matrix;

    //Read values into the matrix
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            data >> entry;
            //cout << entry << endl;
            matrix[width][height] = entry;
        }
    }
    //cout << "done with loop" << endl;
    int size = sizeof(int);

    //Allocate CUDA space
    cudaMalloc((void **) &gpu_matrix, width*height*size);
    cudaMalloc((void **) &answer, 1*size);


    //Move to GPU
    cudaMemcpy(gpu_matrix, matrix, width*height*size, cudaMemcpyHostToDevice);
    cudaMemcpy(answer, &result, size, cudaMemcpyHostToDevice);

    dim3 dimThreadsPerBlock(width, height, 1);
    dim3 numBlock(((width+dimThreadsPerBlock.x-1)/dimThreadsPerBlock.x), ((height+dimThreadsPerBlock.y-1)/dimThreadsPerBlock.y), 1);

    find_ones <<<numBlock, dimThreadsPerBlock>>> (gpu_matrix, answer, width);

    //return to memory
    cudaMemcpy(&result, answer, 1*size, cudaMemcpyDeviceToHost);
    cudaFree(gpu_matrix);
    cudaFree(answer);


    //print answer
    cout << result << endl;
}






