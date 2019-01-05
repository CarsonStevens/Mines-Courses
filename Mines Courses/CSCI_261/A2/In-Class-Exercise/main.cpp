/*A2 In-class Exercise
 *
 *Author: Carson Stevens & Stephanie Holzschuh
 *
 *Practicing Switch
 */



#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
#include <ctime>
#include <string>
#include <cmath>
using namespace std;

int main() {
    
    int input = 0;
    
    cout << "Enter a number between 1 and 5: " << endl;
    cin >> input;
    
    
    switch (input) {
        case 1 :
            cout << "The first number. " << endl;
            break;
        case 2 :
        case 3 :
            cout << "The second or third number." << endl;
            break;
        default :
            cout << "Input error." << endl;
            break;
    }
}