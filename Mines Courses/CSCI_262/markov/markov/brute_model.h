/*
    CSCI 262 Data Structures, Fall 2017, Project 4 - Markov

    brute_model.h

    Class declaration for brute_model, the brute-force Markov text generation
    model.

    Author: C. Painter-Wakefield

    Modified: 11/2/2017
*/

#ifndef _BRUTE_MODEL_H
#define _BRUTE_MODEL_H

#include "model.h"
#include <iostream>

using namespace std;

class brute_model : public markov_model {
public:
	// give the model the example text and the model order; the model
	// should do any preprocessing in this call
	virtual void initialize(std::string text, int order) {
		
		
		// copy first order characters to back to simulate wrap-around
		_data = text + text.substr(0, order);
		_order = order;
	}

	// produce a text in the style of the example
	virtual std::string generate(int size);

protected:
	std::string _data;
	int _order;
};

#endif

