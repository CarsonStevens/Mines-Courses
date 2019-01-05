/*Lab5B, Lab4B with vectors
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 *Prompt user for 15 values, store them in an vector and print the vector including:
 *sorted vector, number of numbers entered, smallest number, largest number, first and last numbers.
 */
 
 
#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
#include <vector>
using namespace std;

int FindMax(vector<int> numbers, int total){
    
    int maxNum = 0;
    
    for (int i = 0; i < total; ++i) {
        if (numbers.at(i) > maxNum) {
         maxNum = numbers.at(i);
        }
    }
    return maxNum;
}

int FindMin(vector<int> numbers, int total, int larger ){

    int minNum = larger;
    
    for (int i = 0; i < total; ++i) {
        if (numbers.at(i) < minNum) {
         minNum = numbers.at(i);
        }
    }
    return minNum;
}



int main() {
    cout << "Hey! Witness my first vector mojo!" << endl;
    cout << "Enter as many non-negative numbers as you'd like and I will tell you what they are." << endl;
    cout << "When you wish to be done, enter -1 to stop entering numbers." << endl << endl;
    
    
    vector <int> userNumbers(0);
    int numList;
    int i = 0;
    
    while (numList != (-1)) {
        
        cout << "Number " << i + 1 << ":\t";
        cin >> numList;
        if (numList >= 0){
            userNumbers.push_back(numList);
            cout << endl;
            ++i;
        }
        else if (numList == (-1)){
            cout << endl;
            break;
        }
        else {
            cout << endl << "Invalid number. Please try again." << endl << endl;
            continue;
        }
    }
    
    
    // //userNumbers.pop_back();
    //  cout << "i = " << i << endl << endl << "Test print of numbers is: ";
    // // for (int k = 0; k < 2; k++){
    // //     cout <<userNumbers.at(k) << " ";
    // // }
    // // cout << endl;
    
    
    cout << "The numbers are: ";
    for (int j = 0; j < (i); ++j){
        cout << userNumbers.at(j);
        cout << " ";
    }
    
    //i = i -1;
    
    cout << endl;
    int largest = FindMax(userNumbers, i) + 1;
    cout << "The smallest number entered is: " << FindMin(userNumbers, i, largest) << endl;
    cout << "The largest number entered is: " << FindMax(userNumbers, i) << endl;
    
    cout << "The first number is: " << userNumbers.at(0) << endl;
    cout << "The last number is: " << userNumbers.back() << endl;
    cout << "Have a nice day!" << endl;
    cout << "Gaze at my awesome." << endl;
    
    return 0;
} 