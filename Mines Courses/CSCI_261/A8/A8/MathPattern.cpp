/* CSCI261 Section E Assignment 8
 *
 * Description: If a user solves the puzzle, award a prize
 *
 * Author: Carson Stevens
 *
 */
#include <iostream>
#include <string>
using namespace std;

#include "MathPattern.h"

MathPattern::MathPattern(){                                                         //Default constructor
    prize = "a delicious grilled cheese sandwhich!";                            //Default prize
    pattern = "\t\t    ?\n\t\t60\t20\n\t   12\t    4\t     4\n\t4\t2\t1\t3";    //Default pattern to solve
    introduction();                                                             //Prints intro
    _result = 1260;                                                             //Defines the answer to the solution.
}

int MathPattern:: getAnswer(){                                                    //Gets the answer when called upon.
    return _answer;
}


string MathPattern::getPattern() {                                                //Prints the pattern to be solved.
    cout << pattern;
    return pattern;
}

string MathPattern::getPrize(){                                                   //Gets the prize when called.
    return prize;
}

void MathPattern::introduction(){                                                 //Prints the introduction
    cout << "Guess what the next number in the pattern is to win a prize!";
    return;
}

int MathPattern::setAnswer(int numToCheck){                                       //Sets the answer if it is a number
    
    if (cin.fail()) {    //Not an int.
        cout << endl << "Error. That isn't an integer.";                        //Prints error if a not int is entered.
        return -1;
    }
    else{
        _answer = numToCheck;                                                   //Sets _answer here.
    }
    return 0;
}


bool MathPattern::correct(){                                                      //Checks if the user input is equals the answer to the pattern
    if(_answer == _result){
        return true;
    }
    else{
        return false;
    }
    
}

void MathPattern::getSolution(){                                                     //prints the solution. called when user is wrong.
    
    cout << "The correct answer is 1260. Number 12 is derived as 4*2+4, first number 4 is derived 2*1+2," << endl;
    cout << "second number 4 is derived as 1*3+1, number 60 is derived as 12*4+12, number 20 is derived as 4*4+4," << endl;
    cout << "finally number 1260 is derived as 60*20+60 = 1260." << endl; 
    
    return;
}

void getTheAnswer(MathPattern& thing){                                            // gets the users input and calls to set it.
    int temp;
    
    cout << endl << endl;
    thing.getPattern();
    
    cout << endl << endl << "What number should be where the ? is:\t";
    cin >> temp;
    thing.setAnswer(temp);
    cout << endl;
}

void printResults(MathPattern& thing){                                            //Prints the results of the user's input.
    if (thing.correct() == true){
         cout << "You won " << thing.getPrize() << endl;
     }
     else{
         cout << "Better luck next time! Here is the solution so you can rest your mind: " << endl;
         thing.getSolution();
         cout << endl;
     }
}