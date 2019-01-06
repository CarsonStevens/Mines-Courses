/*
    CSCI 262 Data Structures, Fall 2017, Project 4 - Markov

    word_model.h

    Class declaration for brute_model, word_model

    Author: Carson Stevens

    Modified: 3/13/2018
*/

#pragma once

#include "model.h"
#include <map>
#include <vector>
#include <string>

using namespace std;

class word_model : public markov_model {
    public:
    	// give the model the example text and the model order; the model
    	// should do any preprocessing in this call
    	virtual void initialize(std::string text, int order);
    
    	// produce a text in the style of the example
    	virtual std::string generate(int size);
    
    protected:
    	std::vector<std::string> _data;
    	int _order;
    	std::map <std::vector<std::string>, std::vector<std::string>> markovMap;
    	std::string lastString; // stores the last text given to the class. used to see
    					        // if the map needs to be generated again
};