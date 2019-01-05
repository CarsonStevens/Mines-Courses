#pragma once
// CSCI-261-Assignment-A7- Magic Item
// Name: Devvrat Singh
// CWID: 10804592
// Section: B
// Prof: Julie Krause
// Filename: MagicItem.h

#include <iostream>
#include <string>
using namespace std;

class JackpotTrivia { // Definition of JackpotTrivia class.

public: // The public members of the class defined below
	JackpotTrivia(); // The constructor titled JackpotTrivia declaration.
	void introduction(); // Declaration of the function titled introduction
	bool triviaCheck(string userInput, string correctAnswer)const; // Function declaration of bool titled triviacheck which compares the userInput to the correct answer.
	void trivia1(); // Declaring the void function titled trivia1.
	void trivia2(); // Declaring the void function titled trivia2
	void trivia3(); // Declaring the void function titled trivia3
	void trivia4(); // Declaring the void function titled trivia4
	



private: // The private members of the class defined below.
	string _myTrivia1; // Defining the variable to hold question 1
	string _myTrivia2; // Defining the variable to hold question 2
	string _myTrivia3; // Defining the variable to hold question 3
	string _myTrivia4; // Defining the variable to hold question 4
	string _correct1; // Defining the variable to hold answer 1
	string _correct2; // Defining the variable to hold answer 2
	string _correct3; // Defining the variable to hold answer 3
	string _correct4; // Defining the variable to hold answer 4

};
