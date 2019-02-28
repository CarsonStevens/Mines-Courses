/*
 * Author: Carson Stevens
 * Date: February 27, 2019
 * Description: 	1)	Read in input from m1.txt into a matrix
 *                  2)	Use CUDA to return the same matrix transposed
 */


#include <iostream>
#include <sstream>
#include <fstream>

__global__ void transpose_matrix(float *transpose, const float *matrix){

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int width = gridDim.x * blockDim.x;

    for (int j = 0; j < blockDim.x; j+= BLOCK_ROWS)
        transpose[x*width + (y+j)] = matrix[(y+j)*width + x];
}


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

    int dev_matrix[width][height];
    int dev_transpose[height][width];
    int *transpose;
    int *matrix

    //Read values into the matrix
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            data >> entry;
            matrix[width][height] = entry;
        }
    }

    int size = width * height * sizeof(int);

    //Allocate CUDA space
    cudaMalloc((void **) &matrix, size);
    cudaMalloc((void **) &transpose, size);

    cudaMemcpy(dev_matrix, matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_transpose, transpose, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(1, 1);
    dim3 dimBlock(width, height);

    transpose_matrix<<<dimGrid, dimBlock>>>(transpose, matrix);
    cudaMemcpy(dev_transpose, transpose, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_matrix);
    cudaFree(dev_transpose);

    //Print results to output
    cout << width << " " << height << endl;
    for(int i = 0; i < width; i ++){
        for(int j = 0; j < height; j++){
            cout << dev_transpose[i][j];
            if(i != width-1){
                cout << " ";
            }
        }
        cout << endl;
    }
}


