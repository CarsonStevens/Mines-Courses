#ifndef _HANGMAN_H
#define _HANGMAN_H

/*
    hangman.h
        
    Class definition for the hangman class.

    assignment: CSCI 262 Project - Evil Hangman        

    author: Carson Stevens

    last modified: 9/24/2017
*/

#include <string>
#include <vector>
#include <map>
#include <set>


using namespace std;

/******************************************************************************
   class hangman

   Maintains game state for a game of hangman.

******************************************************************************/

class hangman {
public:
    hangman();

    // start a new game where player gets num_guesses unsuccessful tries
	void start_new_game(int num_guesses, int length);

    //check if the user's word length is valid
    bool valid(int length);

    // player guesses letter c; return whether or not char is in word
    bool process_guess(char c);

    // display current state of word - guessed characters or '-'
    string get_display_word();

    // How many guesses remain?
	int get_guesses_remaining();

    // What characters have already been guessed (for display)?
    string get_guessed_chars();

    // Has this character already been guessed?
    bool was_char_guessed(char c);

    // Has the game been won/lost?  (Else, it continues.)
    bool is_won();
    bool is_lost();

    
    // Shows the amount of remaining words.
    void answers_left();
    
    // Resets the game if the user wants to play again.
    void reset();

private:

    // Stores the guessed characters
    vector<char> _guessed;
    
    // Stores the amount of guesses left
    int _guesses_left;
    
    // Stores the words according to length
    map<int, vector<string>> _dictionary;
    
    // Stores the current possible words
    vector<string> _evil;
    
    // Stores the word families and corresponding words
    map<string, vector<string> > _word_family;
    
    // Stores the max length of all words
    int _max_length;
    
    // Stores the current solved string.
    string _solved;
    
    // Added to make sure the is_lost member only prints once
    int _counter = 0;
};

#endif
