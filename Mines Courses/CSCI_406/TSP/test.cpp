#include <iostream>
#include <ctime>
#include <ios>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;
#include "point.h"

using namespace std;

//Prints the array
void printArr(int a[],int n)
{
    for (int i=0; i<n; i++)
        cout << a[i] << " ";
    printf("\n");
}

void printVec(vector<Point> points){
    for(auto point : points){
        point.print();
        cout <<  "->";
    }
    cout << endl;
}

void Permute(vector<Point> points, int size, int n, vector<vector<Point>> &permutations){
    if(size == 1){
        permutations.push_back(points);
        // printVec(points);
        return;
    }
    for(int i = 0; i < size; i++){
        Permute(points, size-1, n, permutations);
        if(size%2 == 1){
            swap(points.at(0),points.at(points.size() - 1));
        }
        else{
            swap(points.at(i), points.at(points.size() - 1));
        }
    }
}

// Generating permutation using Heap Algorithm
void heapPermutation(int a[], int size, int n)
{
    // if size becomes 1 then prints the obtained
    // permutation
    if (size == 1)
    {
        printArr(a, n);
        return;
    }
 
    for (int i=0; i<size; i++)
    {
        heapPermutation(a,size-1,n);
 
        // if size is odd, swap first and last
        // element
        if (size%2==1)
            swap(a[0], a[size-1]);
 
        // If size is even, swap ith and last
        // element
        else
            swap(a[i], a[size-1]);
    }
}

// Driver code
int main()
{
    int a[] = {1, 2, 3};
    vector<Point> points;
    vector<vector<Point>> permutations;
    for(int i = 0; i < 2; i++){
        Point point(i,i);
        points.push_back(point);
    }
    int n = sizeof a/sizeof a[0];
    Permute(points, points.size(), points.size(), permutations);
    heapPermutation(a, n, n);
    for(auto points : permutations){
        printVec(points);
    }
    return 0;
}