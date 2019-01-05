/*Lab2B, Pair Programming / Human VS Computer
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 *Create a rock paper siccors game using a random generator
 */
 
 
#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
#include <ctime>
using namespace std;

int main() {
    srand( time(0) );
    
    //Defining variables for user
    char P;
    char R;
    char S;
    char playerChoice;
    
    //Prompt user for input
    cout << "Welcome to Rock, Paper, Scissors! Please enter P, R, or S:" << endl;
    cin >> playerChoice;
    
    //Print user choice
    cout << '\n' << "Player choose " << playerChoice << endl;
    
    //Randomly generate computer choice number
    
    for (int i = 0; i <= 0; ++i) {
        cout << "Computer choose " << rand () % 3 << endl;
    }   
    
    return 0;
}
