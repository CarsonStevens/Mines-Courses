/*Assignment 3: Guess the Number
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 *Set a looping program that loops until the user guesses the correct number.
 */

#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
using namespace std;


 
int main() {
    
    int M;
    int N;


    cout << "Welcome to the Matrix Calculator!" << endl << endl;
    
    //Prompts user for the number of rows and columns in the matrix
    cout << "Please enter the number of rows in the matrix you want: ";
    cin >> M;
    cout << "Please enter the number of columns in the matrix you want: ";
    cin >> N;
    cout << endl;
    
    //Initialized the matrix sizes
    int matrix[M][N];
    int transMatrix[N][M];
    int productMatrix[M][M];
   
   //Prompts the user for the numbers in the array.
    for ( int i = 1; i <= M; ++i) {
        int userNum;
        for (int j = 1; j <= N; ++j){
            cout << "Enter Row " << i << " Column " << j << " : ";
            cin >> userNum;
            matrix[i-1][j-1] = userNum;
        }
    }
    
    //Prints out the matrix
    cout << endl << endl << "The matrix is: " << endl;
    
    for ( int i = 0; i < M; ++i) {
        cout << "[ ";
        for (int j = 0; j < N; ++j) {
            cout << setw(5) << setfill(' ') << matrix[i][j] << " ";
        }
        cout << "]" << endl;
    }
    
    //Prints out the transpose
    cout << endl << endl << "The transpose of this matrix is: " << endl;
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            transMatrix[i][j] = matrix[j][i];
        }
    }
    
    
    for ( int i = 0; i < N; ++i) {
        cout << "[ ";
        for (int j = 0; j < M; ++j) {
            cout << setw(5) << setfill(' ') << transMatrix[i][j] << " ";
        }
        cout << "]" << endl;
    }
    
    cout << endl << endl << "The product of the matrix and its transpose is: " << endl;
    
    // Initializing elements of matrix mult to 0.
    for(int i = 0; i < M; ++i)
        for(int j = 0; j < M; ++j) {
            productMatrix[i][j] = 0;
        }
    
    // MxM, N numCol
    for (int i = 0; i < M; ++i){
        for (int j = 0; j < M; ++j){
            for (int k = 0; k < N; ++k){
                productMatrix[i][j] +=  (matrix[i][k] * transMatrix[k][j]);
            }
        }
    }

    
    //Prints out the product of the two
    
    for ( int i = 0; i < M; ++i) {
        cout << "[ ";
        for (int j = 0; j < M; ++j) {
            cout << setw(5) << setfill(' ') << productMatrix[i][j] << " ";
        }
        cout << "]" << endl;
    }
    
    return 0;
}