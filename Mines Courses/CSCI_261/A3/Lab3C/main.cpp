 /*Lab3C Multiplication Table 
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 *Print a multiplication table
 */
 
 
#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
#include <cmath>
using namespace std;

int main() {
    
    int userNumber;
    
    //Prompt user for number between 2 and 50 not inclusive
    cout << "Please enter a number greater than 2 and less than 50 : " << endl;
    cin >> userNumber;
    
    //Makes sure that number is within the limits
    
    while ((userNumber < 2) || (userNumber > 50)) {
        cout << "Please try again: " << endl;
        cin >> userNumber;
    }
    
    // Sets the first row with initial spacing
    
    cout << setw(5) << setfill(' ') << " ";
    
    for (int range = 1; range <= userNumber; range++) {
        cout << setw(5) << setfill(' ') << right << range;
    }
    
    //Executes the mulitplication and puts the rows in with a nested for loop
    
    cout << endl;
    for (int range = 1; range <= userNumber; range++) {
        cout << setw(5) << setfill(' ') << range;
        for (int rangeTwo = 1; rangeTwo <= userNumber; rangeTwo++) { 
            cout << setw(5) << setfill(' ') << range * rangeTwo;
        }
        cout << endl;
    }
    
    return 0;
}