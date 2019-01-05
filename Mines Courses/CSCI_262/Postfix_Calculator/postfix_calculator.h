/*
	postfix_calculator.h

	Header file for the postfix calculator.  

    CSCI 262, Spring 2018, Project 2

	author: 
*/

#ifndef _POSTFIX_CALCULATOR_H
#define _POSTFIX_CALCULATOR_H

#include <string>
#include <stack>
#include <iostream>

/*
    TODO:  You will need to
	add some member variables to store the state of your
	calculator; you may also add any methods you need.  Keep
	in mind that whatever changes you make, your code should
	still work with calculator_main.cpp.
*/

class postfix_calculator {
	public:
	
		// 1. If a value appears next in the expression, push this value on to the stack.
	
		// 2. If an operator appears next, pop two items from the top of the stack and push the result of the operation on to the stack.
	
		// Take in and process a string query such as "7 14 * 3 /".
		// The string is read left to right.  Numbers are pushed onto
		// the stack.  All other strings are assumed to be operators.
		// If the operator is a binary operator such as "+", then two
		// numbers are popped off the stack, the binary operator is
		// applied, and the answer is pushed back on the stack.  If
		// the operator is a unary operator such as "sin" (if 
		// implemented), then only one number is popped off and the
		// answer pushed.
	    // 
	    // Note that some expressions might leave extra numbers on
	    // the stack - e.g., "1 2 3 +" should result in the stack
	    // containing the values "1 5".  A subsequent user entry of
	    // "*" would then be valid, for instance. 
		//
		// IF the query causes a stack underflow situation (i.e., 
		// an operator requires an operand but the stack is empty),
		// then the method should return false, and the stack should
	    // be emptied.
		bool evaluate(std::string expr);
	
		// Remove all items from the current stack.
		void clear();
	
		// Return the number at the top of the stack (this will be the
		// answer if a valid query was passed to evaluate).  If the
		// stack is empty, returns 0.0.  The number  should *not* be 
		// popped from the stack (this allows further  calculations
		// using this number).
		double top();
		
		
		// Return a string representation of the stack showing all the
		// elements in it.  Hint: make a copy of the stack to display
	    // the contents of.
		std::string to_string();
	
		// Constructor
		postfix_calculator();
	
	
	private:
		// TODO: declare member variables here for your stack, etc.
		std::string _expression; //Used for debuging. Prints out the stack
		std::stack <double> _numbers; // holds the calculators stack
		double _answer;	// is the top members of the stack
};

#endif
