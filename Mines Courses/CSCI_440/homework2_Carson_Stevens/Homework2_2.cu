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

    // allocate memory on device
    cudaMalloc((void **)&dev_matrix,height*width*sizeof(int));
    cudaMalloc((void **)&dev_transpose,height*width*sizeof(int));

    // copy host data to device using cudaMemcpy
    cudaMemcpy(dev_matrix,matrix,width*height*sizeof(int),cudaMemcpyHostToDevice);

    // kernel call
    dim3 threadsPerBlock(width,height,1);
    dim3 numBlocks((width+threadsPerBlock.x-1)/threadsPerBlock.x,
                   (height+threadsPerBlock.y-1)/threadsPerBlock.y,1);

    transpose_matrix<<<numBlocks, threadsPerBlock>>>(dev_transpose,dev_matrix,width,height);

    // copy result from device to host
    cudaMemcpy(transpose,dev_transpose,width*height*sizeof(int),cudaMemcpyDeviceToHost);
//    //Allocate CUDA space
//    cudaMalloc((void **) &dev_matrix, width * height * size);
//    cudaMalloc((void **) &dev_transpose, width * height * size);
//
//    cudaMemcpy(dev_matrix, matrix, width * height * size, cudaMemcpyHostToDevice);
//    //cudaMemcpy(dev_transpose, transpose, width * height * size, cudaMemcpyHostToDevice);
//
//    dim3 dimBlock(width, height, 1);
//    dim3 numBlock(((width+dimBlock.x-1)/dimBlock.x), ((height+dimBlock.y-1)/dimBlock.y), 1);
//
//    transpose_matrix<<<numBlock, dimBlock>>>(dev_transpose, dev_matrix, width, height);
//    cudaMemcpy(transpose, &dev_transpose, size*width*height, cudaMemcpyDeviceToHost);


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
//
//
//
//#include <stdio.h>
//#include <stdlib.h>
//#include <iostream>
//#include <fstream>
//#include <string.h>
//using namespace std;
//
//__global__ void kernel_transpose(int* transposed_d, const int* matrix_d, int width, int height){
//    int col = threadIdx.x + blockIdx.x * blockDim.x;
//    int row = threadIdx.y + blockIdx.y * blockDim.y;
//
//    // accessing 2d array as a 1d array
//    // e.g : matrix_d[2] = transposed_d[4] where matrix_d is 4col3row and transposed_d is 3col4row
//    transposed_d[col*height + row] = matrix_d[row*width + col];
//}
//
//int main(int argc, char* argv[]){
//    ifstream file;
//    string str;
//    file.open(argv[1]); // create stream to read txt file
//
//    int col,row;
//    file >> str;
//    col = atoi(str.c_str());
//    file >> str;
//    row = atoi(str.c_str());
//
//    // host variables
//    int matrix[row][col];
//    int transposed[col][row];
//
//    // device variables
//    int *matrix_d;
//    int *transposed_d;
//
//    int temp;
//    // populate host matrix from text file
//    for(int i=0;i<row;i++){
//        for(int j=0;j<col;j++){
//            file >> str;
//            temp = atoi(str.c_str());
//            matrix[i][j] = temp;
//        }
//    }
//    file.close();
//
//    // allocate memory on device
//    cudaMalloc((void **)&matrix_d,row*col*sizeof(int));
//    cudaMalloc((void **)&transposed_d,row*col*sizeof(int));
//
//    // copy host data to device using cudaMemcpy
//    cudaMemcpy(matrix_d,matrix,row*col*sizeof(int),cudaMemcpyHostToDevice);
//
//    // kernel call
//    dim3 threadsPerBlock(col,row,1);
//    dim3 numBlocks((col+threadsPerBlock.x-1)/threadsPerBlock.x,
//                   (row+threadsPerBlock.y-1)/threadsPerBlock.y,1);
//
//    kernel_transpose<<<numBlocks, threadsPerBlock>>>(transposed_d,matrix_d,col,row);
//
//    // copy result from device to host
//    cudaMemcpy(transposed,transposed_d,row*col*sizeof(int),cudaMemcpyDeviceToHost);
//
//    cout << "\n original"<<endl;
//    for(int i=0;i<row;i++){
//        cout << "\n";
//        for(int j=0;j<col;j++){
//            cout << matrix[i][j] << " ";
//        }
//    }
//
//    // print result
//    cout << "\n transposed" << endl;
//    for(int i=0;i<col;i++){
//        cout << "\n";
//        for(int j=0;j<row;j++){
//            cout << transposed[i][j] << " ";
//        }
//    }
//    cout << endl;
//
//    // free memory on device
//    cudaFree(matrix_d);
//    cudaFree(transposed_d);
//
//    return 0;
//}
