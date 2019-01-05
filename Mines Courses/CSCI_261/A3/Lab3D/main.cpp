/*Lab3D Yahtzee
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 *Create a part of a game of Yahtzee
 */
 
#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
#include <cmath>
using namespace std;

// is Twos checks if any of the die are 2s

bool isTwos( int die1, int die2, int die3, int die4, int die5 ) {
    //Checks if each die is a 2
    if ((die1== 2) || (die2== 2) || (die3== 2) || (die4== 2) || (die5 == 2)) {
        //returns true if any of the die are a 2
        return true;
    }
    else {
        return false;
    }
}


// score Fours adds the total number of die that are 4.

int scoreFours( int die1, int die2, int die3, int die4, int die5 ) {
    
    //Initialize and defines the variables of the function
    int i = 0;
    int total = 0;
    
    //Checks if each die is a 4 and it is, adds 1 to the amount of die that are 4.
    if(die1 == 4) {
        ++i;
    }
    if(die2 == 4) {
        ++i;
    }
    if(die3 == 4) {
        ++i;
    }
    if(die4 == 4) {
        ++i;
    }
    if(die5 == 4) {
        ++i;
    }
    
    // Multiplies the amount of die that are the number 4 and totals them.
    total = i * 4;
    return total;
}