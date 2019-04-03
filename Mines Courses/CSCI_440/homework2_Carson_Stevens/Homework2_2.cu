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

__global__ void matrix_transpose(int* dev_transpose, const int* dev_matrix, int width, int height){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    //Mapping
    dev_transpose[x*height + y] = dev_matrix[y*width + x];
}

int main(int argc, char* argv[]){

    //Load file
    //Declare ifstream object for .txt file parsing.
    //open the file from which to read the data
    ifstream file(argv[1]);
    if (!file) {
        cerr << "Error opening input." << endl;
        return (1);
    }

    int width;
    int height;
    file >> width >> height;

    // Define matrices for original and transpose
    int matrix[height][width];
    int transpose[width][height];
    int *dev_transpose;
    int *dev_matrix;

    //Read values into matrix
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            file >> matrix[i][j];
            //cout << "(" << i << "," << j << ")\t" << matrix[i][j] << endl;
        }
    }
    file.close();

    int size = sizeof(int)*width*height;




    // Allocate memory on GPU
    cudaMalloc((void **)&dev_matrix, size);
    cudaMalloc((void **)&dev_transpose, size);

    // copy data to device
    cudaMemcpy(dev_matrix, matrix, size, cudaMemcpyHostToDevice);

    // Establish thread and block size
    dim3 threadsPerBlock(width, height, 1);
    dim3 numBlocks((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y, 1);

    //Call function
    matrix_transpose<<<numBlocks, threadsPerBlock>>>(dev_transpose,dev_matrix,width,height);

    // copy result back
    cudaMemcpy(transpose, dev_transpose, size, cudaMemcpyDeviceToHost);

    // print result
    cout <<  height << " " << width << endl;
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            cout << transpose[i][j];
            if(j != height){
                cout << " ";
            }
        }
        cout << endl;
    }

    // free memory
    cudaFree(dev_matrix);
    cudaFree(dev_transpose);

    return 0;
}
