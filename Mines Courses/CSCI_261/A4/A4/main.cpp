/*Assignment 4 / Wheel of Fortune
 *
 *Author: Carson Stevens
 *
 *Create a game of hangman
 */
 
 
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <ios>
#include <iomanip>
#include "hangman.h"
using namespace std;


int main () {
    
    //Variables and initialization
    srand(time(0));
    char userGuess; // the letter that the user guesses
    int numGuesses = 0;
    int counter = 0;
    char used[numGuesses];
    char guess; // going to be used to convert the user input to capital letter
    const int MAX_WRONG = 12; // the amount of characters that the user can guess and be wrong
    int wrongGuess = 0; // the amount of wrong guesses
    string word; // the word that needs to be guessed
    string words[] = {"PROGRAMMING"}; // if words are added, change the % rand function to the number of words
    char programming[] = {'P', 'R', 'O', 'G', 'R', 'A', 'M', 'M', 'I', 'N', 'G'};
    bool boolAnswer =  false;
    bool checked = true;
    
    //chose the  word randomly

    int n = rand() % 1;
    word = words[n];
    char solution[word.length()]; //Array that stores the solution word.
    char blank[word.length()]; //Stores the unsolved letters
    char intialBlank[word.length()];
    int length = word.length(); // the length of the randomized word.
    
    //Initializing solution array to the proper word... If new words are added, use else if/else statement to add
    if (word == "PROGRAMMING") {
        for (int i = 0; i < length; ++i) {
            solution[i] == programming[i];
        }
    }
    
    //Intro to the game
    cout << "Wheel! Of!! Fortune!!!" << endl << endl;
    
    //Initialize the Blank array
    for (int i = 0; i < length; ++i) {
        blank[i] = '_';
    }
    
    
    // Executes when the user's solution doesn't equal the answer
    while (( boolAnswer == false) && (MAX_WRONG > wrongGuess)) {
        
        cout << "Take a guess: ";
        answerKey(blank, word.length());
        cout << endl << "Your guess: ";
        cin >> userGuess;
        cout << endl;
        ++numGuesses;
        guess = toupper(userGuess);
        
        //checks to see if the user has guessed the letter yet. Executes after the first guess
        if (numGuesses > 0) {
            checked = check (guess, used, numGuesses);
        }
        
        
        // When the letter has not been used twice, this adds the new letter to the answer if the guess is in the answer
        if (checked == true) {
            
            int timeInWord = 0;
            for (int i = 0; i < length; ++i) {
                if (guess == programming[i]) {
                    blank[i] = guess;
                    timeInWord++;
                }
                //cout << "i: " << i << "\tcount: " << timeInWord << std::endl;
            }
                
                
            if (timeInWord == 0) {
                ++wrongGuess;
                cout << "That letter is not in the word." << endl;
                
                if (wrongGuess != MAX_WRONG) {
                    cout << "You have " << MAX_WRONG - wrongGuess << " wrong guesses left before you lose!" << endl << endl;
                }
            }
                
            used[counter] = guess;
            ++counter;
            cout << endl;
        }
        
        if (checked == false) {
            cout << "You've already guessed that! Try again!" << endl;
        }
        
        boolAnswer = answer(blank, programming, word.length());
    }
    
    
    //Prints that the user wins when the puzzle is solved
    if (boolAnswer == true) {
        cout << "CONGRATS! You solved the puzzle: ";
        for (int i = 0; i < length; ++i) {
                cout << blank[i] << " ";
            }
    }
    
    // Prints that the user loses if the number of wrong guesses equals the max number of wrong guesses.
    if (MAX_WRONG == wrongGuess) {
        cout << endl << "You lose!";
    }
    
    return 0;
}