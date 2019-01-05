/* CSCI261 Section E Assignment 7
 *
 * Description: If a user solves the puzzle, award a prize
 *
 * Author: Carson Stevens
 *
 */

#include <string>
#include <iostream>
using namespace std;
#pragma once

class MagicItem{
    
    public:
        MagicItem();                //Default constructor
        void introduction();        //Prints intro
        void solution();            //Prints solution
        bool correct();             //Checks if user input is correct
        int setAnswer(int);         //Sets answer if it is resonable
        int getAnswer();            //Gets said resonable answer
        string getPattern();        //Prints the pattern when called
        string getPrize();          //Prints the prize when called
        
        
    private:
        string pattern;             //is the pattern of numbers
        string prize;               //is the prize
        int _answer;                //is the answer to the pattern
        int _result;                //is the user's answer
        
};

void getTheAnswer(MagicItem& thing);        //Prints the initial pattern and gets the user's input

void printResults(MagicItem& thing);        //Prints the prize or solution depending on the result