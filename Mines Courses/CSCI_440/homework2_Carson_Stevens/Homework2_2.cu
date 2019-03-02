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

//    int x = threadIdx.x + blockDim.x * blockIdx.x;
//    int y = threadIdx.y + blockDim.y * blockIdx.y;
//    int width_block = blockDim.x * blockDim.y;

    //transpose[x*height + y] = matrix[y*width + x];

    int xIndex = blockIdx.x*blockDim.x+ threadIdx.x;
    int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
    int index_in = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;

    for (int i=0; i<blockDim.x; i+=blockDim.y){
        transpose[index_out+i] = matrix[index_in+i*width];
    }

    
    //Mapping for transpose
//    for (int j = 0; j < blockDim.x; j+= width){
//        transpose[x * width_block + (y + j)] = matrix[(y + j) * width_block + x];
//    }
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

    file >> width >> height;
    // Define matrices for original and transpose
    int dev_matrix[width][height];
    int dev_transpose[height][width];
    int *transpose;
    int *matrix;

    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            file >> dev_matrix[i][j];
            //cout << "(" << i << "," << j << ")\t" << dev_matrix[i][j] << endl;
        }
    }

    int size = sizeof(int);

    //Allocate CUDA space
    cudaMalloc((void **) &matrix, width * height * size);
    cudaMalloc((void **) &transpose, width * height * size);

    cudaMemcpy(matrix, dev_matrix, width * height * size, cudaMemcpyHostToDevice);
    cudaMemcpy(transpose, dev_transpose, width * height * size, cudaMemcpyHostToDevice);

    dim3 dimThreadsPerBlock(width, height, 1);
    dim3 numBlock(((width+dimThreadsPerBlock.x-1)/dimThreadsPerBlock.x), ((height+dimThreadsPerBlock.y-1)/dimThreadsPerBlock.y), 1);

    transpose_matrix<<<numBlock, dimThreadsPerBlock>>>(transpose, matrix, width, height);
    cudaMemcpy(transpose, dev_transpose, size, cudaMemcpyDeviceToHost);


    //Print results to output
    cout << "original" << endl << width << " " << height << endl;

    for (int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            cout << dev_matrix[i][j] << " ";
        }
        cout << endl;
    }

    cout << "transpose" << endl;
    for (int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            cout << dev_transpose[i][j] << " ";
        }
        cout << endl;
    }

    cudaFree(matrix);
    cudaFree(transpose);

}


