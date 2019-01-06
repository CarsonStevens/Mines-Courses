/*
    CSCI 262 Data Structures, Fall 2017, Project 4 - Markov

    map_model.h

    Class declaration for map_model

    Author: Carson Stevens

    Modified: 3/13/2018
*/

#ifndef _MAP_MODEL_H
#define _MAP_MODEL_H

#include "model.h"
#include <map>
#include <string>
#include <sstream>
#include <vector>

class map_model : public markov_model {
public:
	// give the model the example text and the model order; the model
	// should do any preprocessing in this call
	virtual void initialize(std::string text, int order);

	// produce a text in the style of the example
	virtual std::string generate(int size);

protected:
	std::string _data;
	int _order;
	std::map < std::string, std::vector<char>> markovMap;
};

#endif