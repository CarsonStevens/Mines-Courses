/*
    CSCI 262 Data Structures, Fall 2017, Project 4 - Markov

    word_model.cpp

    Class method implementation for word_model, the word_model Markov 
    text generation model.

    Author: Carson Stevens

    Modified: 3/13/2018
*/

#include <cstdlib>
#include <vector>
#include <map>
#include <string>
#include "word_model.h"
#include <iostream>
#include <sstream>

using namespace std;

void word_model::initialize(std::string text, int order){
    
    _order = order;
    		
	// If the user has not changed the order or text file, skip remapping
    if (lastString != text || _order != order){ 
        
		lastString = text;
		
		//used in stringstream to parse txt file into vector
		string temp;
		
		//Clears the map if the user tries to use another order or file
		markovMap.clear();
		
        stringstream ss(text);
        vector<string> textWords;
        
        //store the text file into a vector
        while (ss >> temp){
            //cout << temp <<endl;
            _data.push_back(temp);
        }
        
        // copy first order characters to back to simulate wrap-around
        for (int i = 0; i < order; i++){
            _data.push_back(_data.at(i));
        }
		//Creating the map
        for (int i = 0; i < _data.size()-_order; i++){
            textWords.clear();
            for (int j = 0; j < _order; j++){
                //cout << _data.size() << ' ' << i+j << endl;
                textWords.push_back(_data.at(i+j));
            }
            //cout << i + _order << endl;
            markovMap[textWords].push_back(_data[i + _order]);
        }
	}
    
}


string word_model::generate(int sz){
    
	// pick random k-character substring as initial seed
	int start = rand() % (_data.size() - _order);
	
	string answer;
	answer.reserve(sz);
    vector<string> seed;
    
    
    for (int i = 0; i < _order; i++){
        seed.push_back(_data.at(start + i));
    }
    
    // Generates the text
    string temp;
	for (int i = 0; i < sz; i++) {
		
		// randomally picks a word from the list of possible words
        temp = markovMap.at(seed).at(rand() % markovMap.at(seed).size()); 
        
        //Adds spaces inbetween the words generated
        answer.push_back(' ');
        answer.append(temp);
        
        // set up the seed to be the key in the next loop
        seed.erase(seed.begin());
        seed.push_back(temp);
	}

	return answer;
	
}