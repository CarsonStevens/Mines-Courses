/*
	postfix_calculator.cpp

	Implementation of the postfix calculator. 

    CSCI 262, Spring 2018, Project 2

	author: 
*/

#include "postfix_calculator.h"
#include <string>
#include <stack>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

postfix_calculator::postfix_calculator(){
	_answer = 0.0;
	_expression = "";
}

bool postfix_calculator::evaluate(string expr) {
	
	
	
	string top;
	istringstream ss(expr);
	while (ss >> top){
		
		//operations that only need 1 operand
		if((top == "sqrt") || (top == "ln") || (top == "sin") || (top == "cos") || (top == "tan")){
			if(_numbers.size() <= 0){
				return false;
			}
			
			double operand1 = _numbers.top();
			_expression = _expression.substr(_expression.find_first_of(" \t")+1);
			//cout << "Stack now is: \t" << _expression << endl;
			_numbers.pop();
			
			if(top == "sqrt"){
				_answer = sqrt(operand1);
			}
			
			if(top == "ln"){
				_answer = log(operand1);
			}
			
			if(top == "sin"){
				_answer = sin(operand1);
			}
			
			if(top == "cos"){
				_answer = cos(operand1);
			}
			
			if(top == "tan"){
				_answer = tan(operand1);
			}
			
			_numbers.push(_answer);
			_expression.append(std::to_string(_numbers.top()) + " ");
			//cout << "Stack now is: \t" << _expression << endl;
		}
		
		else if ((top == "+") || (top == "-") || (top == "*") || (top == "/") || (top == "log") || (top == "^") || (top == "nthRoot")){
			
			if(_numbers.size() <= 1){
				return false;
			}
			
			double operand1 = _numbers.top();
			_expression = _expression.substr(_expression.find_first_of(" \t")+1);
			//cout << "Stack now is: \t" << _expression << endl;
			_numbers.pop();
			
			
			double operand2 = _numbers.top();
			_expression = _expression.substr(_expression.find_first_of(" \t")+1);
			//cout << "Stack now is: \t" << _expression << endl;
			_numbers.pop();
			
			if(top == "nthRoot"){
				//operand2 is the one inside the radical
				//operand1 is the nthroot of the radical
				_answer = pow(operand1,1.0/operand2);
			}
			
			if(top == "^"){
				_answer = pow(operand2, operand1);
			}
			
			if(top == "log"){
				//operand1 is the base of the the log
				//operand2 is what is being taken the log of
				_answer = log(operand1)/log(operand2);
			}
			
			if(top == "+"){
				_answer = operand2 + operand1;
			}
			if(top == "-"){
				_answer = operand2 - operand1;
			}
			if(top == "*"){
				_answer = operand2 * operand1;
			}
			if(top == "/"){
				_answer = operand2 / operand1;
			}
			
			_numbers.push(_answer);
			_expression.insert(0, (std::to_string(_numbers.top()) + " "));
			//cout << "Stack now is: \t" << _expression << endl;
			
			
		}	
		else{
			
			double value;
			stringstream sss(top);
			sss >> value;
			_numbers.push(value);
			_expression.insert(0, (std::to_string(_numbers.top()) + " "));
			//cout << "Stack now is: \t" << _expression << endl;
		}
	}
}

void postfix_calculator::clear(){
	while(_numbers.empty() != true){
		_numbers.pop();
	}
	_expression = "";
}


double postfix_calculator::top(){
	//returns 0 when empty.
	if(_numbers.empty() == true){
		return 0.0;
	}
	//returns answer which is on the top of the stack
	else{
		_answer = _numbers.top();
		return _answer;
	}

}


string postfix_calculator::to_string(){
	// numbers added/subtracted to the string in the evaluate member
	return _expression;
}

	

	// TODO: Implement as per postfix_calculator.h
	//
	// Read the comments in postfix_calculator.h for this
	// method first!  That is your guide to the required
	// functioning of this method.
	//
	// There are various ways to parse expr.  I suggest
	// you create an istringstream object, constructed on
	// expr:
	// 	istringstream string_in(expr);
	// Then you can get each substring from expr using >>.
    //
    // Check each substring first to see if you have one of 
    // the four operators; if not, you can assume the value
    // is a number that you can convert to a double.  (This
    // may not be a good assumption - but we won't test you
    // on robustly handling anything other than numbers and
    // operators.)  You can use the stod() function in the
    // string library to convert strings to doubles.

// TODO: Implement the remaining functions specified
// in postfix_calculator.h.
//
// You should start by creating "stubs" for each of
// the methods - these are methods that do nothing
// except return a value if needed.  For example, the 
// evaluate() method above does nothing but return true.
//
// Once you've got stubs for everything, then you should
// be able to compile and test the user interface.  Then
// start implementing functions, *testing constantly*!