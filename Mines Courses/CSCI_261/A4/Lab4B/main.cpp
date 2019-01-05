/*Lab3B, Pair Programming / Arrays
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 *Prompt user for 15 values, store them in an array and print 
 * the array in increasing and decreasing order
 */
 
 
#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
using namespace std;


int FindMax(int orderList[]){
    const int TOTAL_NUM = 15;
    int maxNum;
    
    for (int i = 0; i < TOTAL_NUM; ++i) {
        if (orderList[i] > maxNum) {
         maxNum = orderList[i];
        }
    }
    return maxNum;
}

int FindMin(int orderList[]){
    const int TOTAL_NUM = 15;
    int minNum = orderList[0];
    
    for (int i = 0; i < TOTAL_NUM; ++i) {
        if (orderList[i] < minNum) {
         minNum = orderList[i];
        }
    }
    return minNum;
}


int main() {

    const int FIXED = 15;
    int userNum;
    int numList[FIXED];
    
    
    cout << "Hey! Witness my first array mojo! " << endl << "Enter 15 numbers and I will tell you what they are." << endl;
    
    for (int i = 0; i < FIXED; ++i) {
        cout << "Number " << i + 1 << ": ";
        cin >> numList[i];
        cout << endl;
    }
    
    cout << "So awesome!" << endl << "The numbers are: ";
    
    for (int i = 0; i < FIXED; ++i) {
       cout << numList[i] << " ";
    }
    
    cout << "The largest number is: " << FindMax(numList) << endl;
    
    cout << "The smallest number is: " << FindMin(numList) << endl;
    
    cout << endl << "Have a nice day!" << endl;
    
    return 0;
}    