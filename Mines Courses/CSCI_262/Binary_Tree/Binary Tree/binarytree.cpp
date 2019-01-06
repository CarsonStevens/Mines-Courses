/*

  created by: Borys Niewiadomski

  contact info: theaudition08@sbcglobal.net

  Description: This is an animal guessing program were the program has some built
  in questions and some built in answer. If the program answers wrong and it does
  not have any more answers than it asks the user to input the correct animal
  followed by a question to distunguish the animal it guessed and the users
  correct animal


  */

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include "bintree.h"
#include "useful.h"

using namespace std;

void instruct(); //the instructions

BinaryTreeNode<string>* beginning_tree();//the starting questions and answers

void learn(BinaryTreeNode<string>*& leaf_ptr);//fills in users input and saves it

void ask_and_move(BinaryTreeNode<string>*& current_ptr);//moves pointer to next
                                   //question if the question is answered with a no
void play(BinaryTreeNode<string>* current_ptr);//asks user to think of a question


int main()
{

	BinaryTreeNode<string> *taxonomy_root_ptr;

	instruct();//show insctructions
	taxonomy_root_ptr = beginning_tree();//set the beginning tree
	do
	{
		play(taxonomy_root_ptr);//play the game
	//	taxonomy_root_ptr = beginning_tree();//set the beginning tree
	}
	while (inquire("Shall we play again?"));//as long as user does not enter no
	                                                          //play agian
	cout << "Thank you for teaching me a thing or two." << endl;

	return EXIT_SUCCESS;
}

void instruct() //the game instructions
{
	cout << "Instructions: " << endl;
	cout << "1.You will think of an animal." << endl;
	cout << "2.Then the program will try to guess your animal." << endl;
	cout << "    -If the program cannot guess your animal it will ask you" << endl;
	cout << "     to enter the correct answer followed by a question to"<< endl;
	cout << "     tell apart the last guessed animal and your animal." << endl;
	cout << "3.You can play again or exit." << endl << endl;
}

BinaryTreeNode<string>* beginning_tree()
{
	BinaryTreeNode<string> *root_ptr;
	BinaryTreeNode<string> *child_ptr;

	const string root_question("Are you a mammal?");
	const string left_question("Are you bigger than a cat?");
	const string right_question("Do you live underwater?");
	const string animal1("Kangaroo");
	const string animal2("Mouse");
	const string animal3("Trout");
	const string animal4("Robin");

	root_ptr = create_node(root_question);//position root_ptr at the root question
	
	child_ptr = create_node(left_question);//make the left question
	child_ptr->left = create_node(animal1);//make left answer
	child_ptr->right = create_node(animal2);//make right answer
	root_ptr->left = child_ptr;

	child_ptr = create_node(right_question);//mkae the right question
	child_ptr->left = create_node(animal3);//make left answer
	child_ptr->right = create_node(animal4);//make right answer
	root_ptr->right = child_ptr;
	
	return root_ptr;
}

void learn(BinaryTreeNode<string>*& leaf_ptr)//inserts answer and question into
{                                                 //the tree
	string guess_animal;
	string correct_animal;
	string new_question;

	guess_animal = leaf_ptr->data;
	cout << "I give up. What are you? " << endl;
	getline(cin, correct_animal);//enter the correct animal

	//save the correct animal to a file
	ofstream animal_file;
	animal_file.open("Animals.txt", ios::app);
	if(!animal_file.is_open())
		cout << "Error opening animal file!";
	else if (animal_file.is_open())
		animal_file << correct_animal << endl;
	animal_file.close();

	cout << "Please type a yes/no question that will distinguish a" << endl;
	cout << correct_animal << " from a " << guess_animal << "." << endl;
	getline(cin, new_question);//enter the correct question

	//save the new question to a file
	ofstream question_file;
	question_file.open("Questions.txt", ios::app);
	if(!question_file.is_open())
		cout << "Error opening question file!";
	else if (question_file.is_open())
		question_file << new_question << endl;
	question_file.close();

	leaf_ptr->data = new_question;
	cout << "As a " << correct_animal << ", " << new_question << endl;
	if (inquire("Please answer"))//if yes then set the correct animal to be the
	{                          //right side answer
		leaf_ptr->left = create_node(correct_animal);
		leaf_ptr->right = create_node(guess_animal);
	}
	else //other wise make it the left side answer
	{
		leaf_ptr->left = create_node(guess_animal);
		leaf_ptr->right = create_node(correct_animal);
	}

}

void ask_and_move(BinaryTreeNode<string>*& current_ptr)//reposition the pointer
{
	cout << current_ptr->data;
	if (inquire(" Please answer"))//if answer is yes then position pointer left
		current_ptr = current_ptr->left;
	else //other wise move it to the right
		current_ptr = current_ptr->right;
}

void play(BinaryTreeNode<string>* current_ptr)//play the game
{
	cout << "Think of an animal, then press enter.";
	eat_line();//hit enter key
	
	while (!is_leaf(*current_ptr))//if the current pointer is not empty
		ask_and_move(current_ptr);//then move the current pointer

	cout << ("My gues is " + current_ptr->data);//programs guess
	if (!inquire(". Am i right?"))//if answers no
		learn(current_ptr);//then put the answer and question into the tree
	else
		cout << "I knew it all along!" << endl;//other wise tell user it knew it
}        

