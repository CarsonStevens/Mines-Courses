/*
    main.cpp
        
    Implements the user interface for hangman.
    
    assignment: CSCI 262 Project - Evil Hangman        

    author: Carson Stevens

    last modified: 9/24/2017
*/

#include <iostream>
#include <string>
#include <cstdlib>
#include <cctype>

#include "hangman.h"

using namespace std;

// helper function prototypes
int get_integer(string prompt); // return a positive integer
char get_letter(string prompt); // return a valid lowercase letter
bool get_yesno(string prompt);  // return true == yes / false == no
string trim(string s);          // remove leading/trailing whitespace

int main() {
    cout << "Welcome to Hangman!" << endl;
    cout << "===================" << endl << endl;

    // get the hangman object
    hangman game;
    
    // Keep playing the game until the user decides otherwise
    while (true) {
        
        //  GETTING THE USERS INPUT FOR WORD LENGTH AND GUESSES //
        //////////////////////////////////////////////////////////
        int length;
        int check = false;
        cout << "How long of a word do you want?\t";
        while(check != true){
            cin >> length;
            // cin >> length;
            if(game.valid(length) == true){
                check = true;
                break;
            }
            else{
                cout << endl << "No words of that length. Please enter another length!\t";
            }
        }
        cout << endl;
        int num_guesses = 0;
        while (num_guesses  <= 0 || num_guesses > 26){
            num_guesses= get_integer("How many guesses would you like?");
            if( num_guesses == 26){
                cout << endl;
                cout << "Really... is it even a game anymore..." << endl;
                cout << "There's that many letters in the alphabet..." << endl;
                break;
            }
            if( num_guesses >= 26){
                cout << endl;
                cout << "Really... is it even a game anymore..." << endl;
                cout << "There's not even that many letters in the alphabet..." << endl;
            }
        }    
        num_guesses++;
        cout << endl;
        
        bool answers = get_yesno("Would you like to see the amount of possible remaining words? Y(es) or N(o)? \t");
        
        
        ///////////////////////////////////////////////////////////
        
        
        game.start_new_game(num_guesses, length);

        while (!game.is_won() && !game.is_lost()) {
            
            if(answers == true){
                game.answers_left();
            }
            cout << "Your word is: " << game.get_display_word() << endl;

            string already_guessed = game.get_guessed_chars();
            if (already_guessed.size() == 0) {
                cout << "You have not yet guessed any letters." << endl;
            } else {
                cout << "You have already guessed these letters: ";
                cout << already_guessed << endl;
            }

            cout << "You have " << game.get_guesses_remaining();
            cout << " guesses remaining." << endl << endl;

            char guess = get_letter("What is your next guess?");
            while (game.was_char_guessed(guess)) {
                cout << endl << "You already guessed that!" << endl;
                guess = get_letter("What is your next guess?");
            }
            cout << endl;

            bool good_guess = game.process_guess(guess);
            if (good_guess) {
                cout << "Good guess!" << endl;
            } else {
                cout << "Sorry, that letter isn't in the word." << endl;
            }

            if (game.is_won()) {
                cout << "Congratulations! You won the game!" << endl;
            }

            else if (game.is_lost()) {
                cout << "Oh no! You lost!!!" << endl;
            }
        }

        cout << endl;
        if (!get_yesno("Would you like to play again (y/n)?\t")){
            break;
        }
        else{
            game.reset();
        }
    }

    cout << endl << "Thank you for playing Hangman." << endl;

    return 0;
}

// Prompt for a positive integer response, re-prompting if invalid
// input is given. This is not super-robust - it really should work
// harder to filter out responses like "123foo", but oh well.
int get_integer(string msg) {
    while (true) {
        string input;    
        int result = 0;

        cout << msg << "  ";
        cin >> input;

        result = atoi(input.c_str());
        if (result > 0) return result;

        cout << "I didn't understand that. Please enter a positive integer.";
        cout << endl;
    }
}
    
// Prompt for a letter of the alphabet, re-prompting if invalid
// input is given.
char get_letter(string msg) {
    while (true) {
        string input;    
 
        cout << msg << endl;
        cin >> input;

        input = trim(input);

        if (input.size() == 1) {
            char result = tolower(input[0]);
            if (result >= 'a' && result <= 'z') return result;
        }
        
        cout << "I didn't understand that. ";
        cout << "Please enter a letter of the alphabet.";
        cout << endl;
    }
}


// Prompt for a yes/no response, re-prompting if invalid
// input is given.
bool get_yesno(string msg) {
    while (true) {
        string input;    
 
        cout << msg;
        cin >> input;

        input = trim(input);
        for (int i = 0; i < input.size(); i++) {
            input[i] = tolower(input[i]);
        }

        if (input == "y" || input == "yes") return true;
        if (input == "n" || input == "no") return false;
        
        cout << "I didn't understand that. ";
        cout << "Please enter y(es) or n(o).";
        cout << endl;
    }
}

string trim(string s) {
    int a, b;

    for (a = 0; a < s.size() && isspace(s[a]); a++);
    for (b = s.size() - 1; b >= a && isspace(s[b]); b--);
    
    return s.substr(a, b - a + 1);
}


