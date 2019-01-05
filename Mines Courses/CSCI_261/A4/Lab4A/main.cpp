/*Lab3A, Pair Programming / Arrays
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 *Prompt user for 15 values, store them in an array and print the array
 */
 
 
#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
using namespace std;

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
    
    cout << endl << "Have a nice day!";
    
    return 0;
}    