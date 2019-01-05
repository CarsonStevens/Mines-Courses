/* CSCI261 Section E Assignment 7
 *
 * Description: If a user solves the puzzle, award a prize
 *
 * Author: Carson Stevens
 *
 */
 
#include <iostream>
 
using namespace std;
#include "MagicItem.h"
 
int main(){
    
    MagicItem answer;           //Defines the users input for the class MagicItem
    
    getTheAnswer(answer);       //Prints out the riddle, and gets the user's input
    
    printResults(answer);       //Prints out the prize or the solution depending on
                                //correctness

    return 0;
}