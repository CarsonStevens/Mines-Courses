/*
    hangman.cpp
        
    Method implementations for the hangman class.
    
    assignment: CSCI 262 Project - Evil Hangman        

    author: Carson Stevens

    last modified: 9/24/2017
*/

#include "hangman.h"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>



using namespace std;

// constructor
hangman::hangman() {
    _max_length = -1;
    // TODO: Read in and store words from dictionary.txt
    ifstream dict("dictionary.txt");
    string word;
    int length;
    while (!dict.eof()) {
        
        getline(dict, word);
        length = word.length();
        
        if(length > _max_length){
            _max_length = length;
        }
        
        if(_dictionary.count(length) == 1){
            _dictionary[length].push_back(word);
        }
        else{
            vector<string> words;
            words.push_back(word);
            _dictionary.emplace(length,words);
        }
    }
    dict.close();
    //cout << "Dictionary constructed" << endl;
    
    // TODO: Initialize game state variables
    _solved = "";
}


// start_new_game()
//
// Setup a new game of hangman.
void hangman::start_new_game(int num_guesses, int length) {
    _guesses_left = num_guesses;
    
    _evil = _dictionary[length];
    //cout << "Possible words:\t" << _evil.size() << endl;
    
    for(int i = 0; i < length; i++){
        _solved += "-";
    }
    
    
    
}

//check if the user's word length is valid
// DONE
bool hangman::valid(int length){
    if(_dictionary.count(length) == 0){
        return false;
    }
    return true;
}


// process_guess()
//
// Process a player's guess - should return true/false depending on whether
// or not the guess was in the hidden word.  If the guess is incorrect, the
// remaining guess count is decreased.
bool hangman::process_guess(char c) {
    
    //Clears the passed stored _word_families from previous guess
    _word_family.clear();
    
    //Creates a string of 1s or 0s corresponding to matching characters from
    // the possible words left and then stores them into appropriate word families
    for(auto i : _evil){
        string positions = "";
        for(auto j : i){
            if(j == c){
                positions += '1';
            }
            else{
                positions += '0';
            }
        }
        //cout << i << " : " << positions << endl;
        
        if(_word_family.count(positions) == 1){
            _word_family[positions].push_back(i);
        }
        else{
            vector<string> words;
            words.push_back(i);
            _word_family.emplace(positions, words);
        }
    }
    
    //Finds the word family with the most possible words
    int mostFreq = -1;
    string wordFamily = "";
    for(auto i : _word_family){
        //cout << i.first << " : " << i.second.size() << endl;
        int currentFreq = i.second.size();
        
        if(currentFreq > mostFreq){
            wordFamily.clear();
            mostFreq = i.second.size();
            wordFamily.append(i.first);
        }
    }
    
    //Clears the last possible word list
    _evil.clear();
    
    //Creates the new possible word list
    _evil = _word_family[wordFamily];

   
    // cout << "Amount of possible words is now:\t" << _evil.size() << endl;
    // for(auto i : _evil){
    //     cout << i << " ";
    // }
    // cout << endl << endl;
    
    
    //Adds the guessed character to the list of guessed characters
    _guessed.push_back(c);
    
    //Edits the _solved string to replace chars with the guess if the user's
    //guess was correct
    string key = "";
    int index = 0;
    for(auto i : wordFamily){
        if(i == '1'){
            _solved[index] = c;
        }
        key += '0';
        ++index;
    }
    
    if(key == wordFamily){
        return false;
    }
    else{
        return true;
    }
}


// get_display_word()
//
// Return a representation of the hidden word, with unguessed letters
// masked by '-' characters.
string hangman::get_display_word() {
    return _solved;
}


// get_guesses_remaining()
//
// Return the number of guesses remaining for the player.
//DONE
int hangman::get_guesses_remaining() {
    _guesses_left--;
    return _guesses_left;
}


// get_guessed_chars()
//
// What letters has the player already guessed?  Return in alphabetic order.
//DONE
string hangman::get_guessed_chars() {
    set <char> characters;
    string guessed_chars = "";
    
    //Places the characters in a set to order them alphabetically
    for(int i = 0; i < _guessed.size(); i++){
        characters.emplace(_guessed.at(i));
    }
    
    //Places the characters into a string to return
    for(auto it : characters){
        guessed_chars += it;
    }
    
    return guessed_chars;
}


// was_char_guessed()
//
// Return true if letter was already guessed.

//DONE
bool hangman::was_char_guessed(char c) {
    for(int i = 0; i < _guessed.size(); i++){
        if(c == _guessed.at(i)){
            return true;
        }
    }
    return false;
}


// is_won()
//
// Return true if the game has been won by the player.
bool hangman::is_won() {
    if((_evil.size() == 1) && (_solved == _evil.at(0))){
        cout << "The word was: " << _evil.at(0) << endl;
        return true;
    }
    return false;
}


// is_lost()
//
// Return true if the game has been lost.
bool hangman::is_lost() {
    if(_guesses_left <= 1){
        if(_counter == 0){
            int index = rand() % _evil.size();
            cout << "The word was:\t" << _evil.at(index) << endl;
            ++ _counter;
        }
        return true;
    }
    return false;
}


// Displays the number of possible words left
void hangman::answers_left(){
    cout << "Amount of possible words is now:\t" << _evil.size() << endl;
}

// Resets the game variables for a new game.
void hangman::reset(){
    _evil.clear();
    _solved = "";
    _guessed.clear();
    _counter = 0;
}
