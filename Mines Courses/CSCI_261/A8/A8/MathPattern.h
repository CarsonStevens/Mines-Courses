/* CSCI261 Section E Assignment 8
 *
 * Description: create a pannel of games and prizes for a user to solve/win
 *
 * Author: Carson Stevens
 *
 */

#include <string>
#include <iostream>
using namespace std;
#pragma once

class MathPattern{
    
    public:
        MathPattern();              //Default constructor
        void getSolution();         //Prints solution
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
        void introduction();        //Prints intro
};

void getTheAnswer(MathPattern& thing);        //Prints the initial pattern and gets the user's input

void printResults(MathPattern& thing);        //Prints the prize or solution depending on the result