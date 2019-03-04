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

__global__ void transpose_matrix(int *dev_transpose, int *dev_matrix, int width, int height){

    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
//    int width_block = blockDim.x * blockDim.y;

    dev_transpose[row*height + col] = dev_matrix[col*width + row];

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
    int matrix[width][height];
    int transpose[height][width];
    int *dev_transpose;
    int *dev_matrix;

    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            file >> matrix[i][j];
            //cout << "(" << i << "," << j << ")\t" << matrix[i][j] << endl;
        }
    }
    file.close();

    int size = sizeof(int);

    //Allocate CUDA space
    cudaMalloc((void **) &dev_matrix, width * height * size);
    cudaMalloc((void **) &dev_transpose, width * height * size);

    cudaMemcpy(dev_matrix, matrix, width * height * size, cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_transpose, transpose, width * height * size, cudaMemcpyHostToDevice);

    dim3 dimBlock(width, height, 1);
    dim3 numBlock(((width+dimBlock.x-1)/dimBlock.x), ((height+dimBlock.y-1)/dimBlock.y), 1);

    transpose_matrix<<<numBlock, dimBlock>>>(dev_transpose, dev_matrix, width, height);
    cudaMemcpy(transpose, &dev_transpose, size*width*height, cudaMemcpyDeviceToHost);


    //Print results to output
    cout << "original" << endl << width << " " << height << endl;

    for (int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }

    cout << "transpose" << endl;
    for (int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            cout << transpose[i][j] << " ";
        }
        cout << endl;
    }

    cudaFree(dev_matrix);
    cudaFree(dev_transpose);

}


