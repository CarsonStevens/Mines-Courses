/*
    CSCI 262 Data Structures, Fall 2017, Project 4 - Markov

    map_model.cpp

    Class method implementation for map_model

    Author: Carson Stevens

    Modified: 3/13/2018
*/

#include <cstdlib>
#include <vector>
#include <map>
#include <string>
#include "map_model.h"

using namespace std;


void map_model::initialize(std::string text, int order){
	
	// copy first order characters to back to simulate wrap-around
	_data = text + text.substr(0, order);
	_order = order;
	
	//Clears the map so if the users tries multiple orders or txt files
	//the map won't have the previous data stored
	markovMap.clear();
	
	//Creating the markovMap
	for (int i = 0; i < _data.size() - _order; i++){
    	markovMap[_data.substr(i,_order)].push_back(_data.at(i + _order));
	}
	
}


string map_model::generate(int sz){
    
    // seed = random k-character substring from the training text --- the initial seed
    //     repeat N times to generate N random letters
    //  find the vector (value) associated with seed (key) using the map
    //  next-char = choose a random char from the vector (value)
    //  print or store next-char
    //  seed = (last k-1 characters of seed) + next-char 
    
	// pick random k-character substring as initial seed
	int start = rand() % (_data.length() - _order);
	string seed = _data.substr(start, _order);

	vector<char> list;
	string answer;
	answer.reserve(sz);
	answer.append(seed);
	
	// Generating using the map model
	for (int i = 0; i < sz; i++) {
		
		list = markovMap.at(seed);
		answer.push_back(list.at(rand() % list.size()));
		
		//DEBUG
		//cout << seed << ' ' << answer.back() << ' ';
		
		seed.erase(0,1);
		seed.push_back(answer.back());
		
		//DEBUG
		//cout << seed << endl;
	}
	
	return answer;
	
}