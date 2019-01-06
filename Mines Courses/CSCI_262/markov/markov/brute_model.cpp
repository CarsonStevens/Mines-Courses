/*
    CSCI 262 Data Structures, Fall 2017, Project 4 - Markov

    brute_model.cpp

    Class method implementation for brute_model, the brute-force Markov 
    text generation model.

    Author: C. Painter-Wakefield

    Modified: 11/2/2017
*/

#include <cstdlib>
#include <vector>
#include "brute_model.h"

using namespace std;

// Brute force character generator
string brute_model::generate(int sz) {

	// pick random k-character substring as initial seed
	int start = rand() % (_data.length() - _order);
	string seed = _data.substr(start, _order);

	vector<char> list;
	string answer;
	answer.reserve(sz);

	for (int i = 0; i < sz; i++) {
		list.clear();

		// find first occurrence of k-gram (seed)
		int pos = _data.find(seed);

		while (pos != string::npos && pos < _data.length()) {
			// what comes after seed in the text?
			char c = _data[pos + _order];
			list.push_back(c);

			// find next occurrence of seed
			pos = _data.find(seed, pos+1);
		}

		// choose next character based on probability of occurrence in list
		char c = list[rand() % list.size()];
		answer.push_back(c);

		// update seed
		seed = seed.substr(1) + c;
	}

	return answer;
}
