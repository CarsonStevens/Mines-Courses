// CSCI-261-Assignment-A7- Magic Item
// Name: Devvrat Singh
// CWID: 10804592
// Section: B
// Prof: Julie Krause
// Filename: MagicItem.cpp

#include "MagicItem.h"

JackpotTrivia::JackpotTrivia() { // Defining the function JackpotTrivia which holds all the trivia questions with their correct answers.

	_myTrivia1 = "Question (1) Name the seventh planet from the sun?";
	_correct1 = "Uranus";
	

	_myTrivia2 = "Question (2) Name the largest freshwater lake in the world?";
	_correct2 = "Lake-Superior";
	

	_myTrivia3 = "Question (3) What is another word for lexicon?";
	_correct3 = "Dictionary";
	

	_myTrivia4 = "Question (4) Who invented the rabies vaccination?";
	_correct4 = "Louis-Pasteur";
	


}

void JackpotTrivia::introduction() { // Defining the function introduction which provides the user with the instructions and rules of the trivia game.

	cout << "Welcome to the game of Jackpot Trivia!!!!!!" << endl;
	cout << "The instructions and the rules for the game are as follows:" << endl;
	cout << "(1) You need to answer a total of four trivia questions which will get progressively difficult." << endl;
	cout << "(2) You will win one diamond for the first correct question." << endl;
	cout << "(3) You will win two diamonds for the next correct question." << endl;
	cout << "(4) The next correct question will earn you three diamonds." << endl;
	cout << "(5) The final question which will be the hardest of them all earns you a pot full of diamonds." << endl;
	cout << "(6) Note that you can only move to the next question after answering the previous question correctly." << endl;
	cout << "(7) However you don't have to answer all of them correctly to win something." << endl;
	cout << "(8) Sit tight and enjoy the trivia ride" << endl << endl;



}

bool JackpotTrivia::triviaCheck(string userInput, string correctAnswer)const { // Defining the function triviaCheck that compares the user entered answer to the correct answer.

	if (userInput == correctAnswer) {

		return 1;
	}

	else {

		return 0;
	}


}


void JackpotTrivia::trivia1() { // Defining the function trivia1 which asks user the first trivia questions and provides the user with the appropriate response
	string userInput;

	cout << _myTrivia1 << endl;
	cin >> userInput;

	if (triviaCheck(userInput, _correct1)) {

		cout << "You have just won a diamond. Three more and you have the pot!!!!!" << endl << endl;
		trivia2();
		
	}

	else {

		cout << "Better luck next time!!!!!" << endl << endl;
	}


}

void JackpotTrivia::trivia2() { // Defining the function trivia2 which asks user the second question and provides the user with the appropriate response
	string userInput;

	cout << _myTrivia2 << endl;
	cin >> userInput;

	if (triviaCheck(userInput, _correct2)) {

		cout << "You have just won two diamonds. Two more and the pot is yours." << endl << endl;
		trivia3();
		
	}

	else {

		cout << "You were just two away from the pot.Better luck next time!!!!!" << endl << endl;
	}


}

void JackpotTrivia::trivia3() { // Defining the function trivia3 which asks user the third trivia question and provides the appropriate response
	string userInput;

	cout << _myTrivia3 << endl;
	cin >> userInput;

	if (triviaCheck(userInput, _correct3)) {

		cout << "You have just won three diamonds. One more to go and the pot belongs to you!!!!!!" << endl << endl;
		trivia4();
		
	}

	else {

		cout << "Ohhh you were just one away from the glowing pot. Better luck next time!!!!!" << endl << endl;
	}


}

void JackpotTrivia::trivia4() { // Defining the function trivia4 which asks user the fourth trivia question and provides the appropriate response
	string userInput;

	cout << _myTrivia4 << endl;
	cin >> userInput;

	if (triviaCheck(userInput, _correct4)) {

		cout << "Guess what?????? You just won a pot full of 30 diamonds. You sir are one rich person now. Enjoy your prize and I will see you soon!!!!!!" << endl << endl;
		
	}

	else {

		cout << "Ohhhhh. So close. I feel so bad for you. The pot was almost your's. Better luck next time!!!!!" << endl << endl;
	}


}