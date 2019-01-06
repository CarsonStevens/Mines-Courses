/*
   CSCI 262 Data Structures, Fall 2017, Project 4 - Markov

   main.cpp

   Contains the main() function, which handles command line arguments and
   launches the Markov application.

   Author: C. Painter-Wakefield

   Modified: 11/2/2017
*/

#include "markov.h"
#include "brute_model.h"
#include "map_model.h"
#include "word_model.h"

// TODO: add includes for your models

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

// markov()
// Constructs the markov application.  Most importantly, it sets up the 
// different model types.
//
// TODO: uncomment lines below as you add your models
markov::markov() {
	_model_map.emplace("brute", new brute_model);
	_model_map.emplace("map", new map_model);
	_model_map.emplace("word", new word_model);
}

// ~markov()
// Releases any models created in constructor.
markov::~markov() {
	for (auto entry: _model_map) {
		delete entry.second;
	}
}

// run()
// Launches a user-interactive Markov application session.
void markov::run() {
	cout << "-+== Welcome to the Markov random text generator! ==+-";
	cout << endl;

	while (true) {
		// Tell the user the current application state
		if (_text_file != "") {
			cout << "The current text is " << _text_file << "." << endl;
		} 
		if (_order != -1) {
			cout << "The current order is " << _order << "." << endl;
		}
		if (_model_type != "") {
			cout << "The current model is the " << _model_type;
			cout << " model." << endl;
		}
		if (_random_seed != -1) {
			cout << "The current random seed is " << _random_seed;
			cout << "." << endl;
		}
			
		cout << endl;
		cout << "Choose a menu option:" << endl;
		cout << "1. Set the model type" << endl;
		cout << "2. Load an example text" << endl;
		cout << "3. Set the model order" << endl;
		cout << "4. Set the random seed" << endl;
		cout << "5. Generate some text!" << endl;
		cout << "6. Quit" << endl;
		
		string response = "";	
		getline(cin, response);
		int option = atoi(response.c_str());
		cout << endl;

		switch (option) {
		case 1:
			set_model();
			break;
		case 2:
			set_text();
			break;
		case 3:
			set_order();
			break;
		case 4:
			set_random_seed();
			break;
		case 5:
			generate_text();
			break;
		case 6:
			return;
		default:
			cout << "Unrecognized option! " << endl;
		}
	}
}

// run_one(...)
// Sets up a model as specified, generates text, then exits.
void markov::run_one(string infile, int order, string model_type, int size, int seed) {
	if (!_set_text(infile)) { return; }
	if (!_set_model(model_type)) { return; }
	if (order > 0) { 
		_order = order;
	} else {
		cout << "Invalid order." << endl;
		return;
	}
	_random_seed = seed;
	_generate_text(size);
}

// set_text()
// Ask the user for a new text file to load as an example text.
void markov::set_text() {
	if (_text_file != "") {
		cout << "The current text file is " << _text_file << "." << endl;
	}
	cout << "Please enter text file name: ";
	string infile;
	getline(cin, infile);
	if (infile == _text_file) {
		cout << "Text file is unchanged, still " << _text_file << "." << endl;
	} else if (_set_text(infile)) {
		cout << "Text file changed to " << _text_file << "." << endl;
	} else if (_text_file != "") {
		cout << "Text file is unchanged, still " << _text_file << "." << endl;
	} else {
		cout << "Text file is not set." << endl;
	}
}
	

// set_order()
// Ask the user for a new model order.
void markov::set_order() {
	if (_order != -1) {
		cout << "The current order is " << _order << "." << endl;
	}
	cout << "Please enter desired order (an integer greater than zero): ";
	string response;
	getline(cin, response);
	int test = atoi(response.c_str());
	if (test == _order) {
		cout << "Order is unchanged, still " << _order << "." << endl;
	} else if (test >= 1) {
		_order = test;
		_initialized = false;
		cout << "Order changed to " << _order << "." << endl;
	} else if (_order >= 1) {
		cout << "Invalid order.  Order is unchanged, still " << _order << "." << endl;
	} else {
		cout << "Invalid order.  Order is not set." << endl;
	}
}

// set_model()
// Ask the user for a model type to use.
void markov::set_model() {
	if (_model_type != "") {
		cout << "The current model is the " << _model_type << " model." << endl;
	}
	cout << "Please enter new model name (";
	string choices = "";
	for (auto entry: _model_map) {
		choices += entry.first;
		choices += ", ";
	}
	choices.pop_back();
	choices.pop_back();
	cout << choices << "): ";

	string response;
	getline(cin, response);
	_set_model(response);
}

// set_random_seed()
// Pseudo-random number generators produce sequences of numbers that are very
// random-like.  However, they are quite deterministic.  If you want, you can
// "seed" the generator with a specific seed to get repeatable behavior, which
// helps a lot in debugging.
void markov::set_random_seed() {
	cout << "The current random seed value is " << _random_seed << "." << endl;
	cout << "With a value of -1, the random sequence will be different each time." << endl;
	cout << "With any other integer value, the random sequence will be the same each time." << endl;
	cout << "Please enter desired random seed value: ";
	string response;
	getline(cin, response);
	_random_seed = atoi(response.c_str());
	cout << "The random seed value is now " << _random_seed << "." << endl;
}

// init_random()
void markov::init_random() {
	if (_random_seed == -1) {
		// get a different random sequence every time
		time_t t = time(NULL);
		srand(t);
	} else {
		srand(_random_seed);
	}
}

// initialize_model() 
// Initialize (or re-initialize) the model object with the current text and
// order.
void markov::initialize_model() {
	if (!_initialized) {
		_mark_start();
		_model->initialize(_text, _order);
		_mark_stop();
		_report_time("Model initialized");
	}
	_initialized = true;
}

void markov::generate_text() {
	if (_model_type == "word") {
		cout << "How many words would you like to generate? ";
	} else {
		cout << "How many characters would you like to generate? ";
	}
	string response;
	getline(cin, response);
	_generate_text(atoi(response.c_str()));
}

// _set_text(string infile)
// Try to open infile for reading, read in text and remove excess whitespace.
// Return true on success, false on failure to open file.
bool markov::_set_text(string infile) {
	// get text from input file
	ifstream fin(infile);
	if (!fin) {
		cerr << "Error opening input file \"" << infile << "\"!" << endl;
		return false;
	}
	_text_file = infile;

	// get all strings; extra whitespace will be ignored 
	ostringstream text;
	while (!fin.eof()) {
		string s;
		fin >> s;
		text << s << ' ';
	}
	fin.close();

	_text = text.str();
	_initialized = false;

	return true;
}

// _set_model(string model_type) 
// Try to match provided model_type with a model we know about.  Return true
// if the model type is recognized, false otherwise.
bool markov::_set_model(string model_type) {
	if (model_type == "") {
		cout << "Unrecognized model type." << endl;
		return false;
	} else if (model_type == _model_type) {
		cout << "Model type unchanged, still " + model_type + " model.";
		cout << endl;
		return true;
	} else if (_model_map.count(model_type) != 0) {
		_initialized = false;
		_model_type = model_type;
		_model = _model_map[model_type];
		cout << "Model changed to " + model_type + " model." << endl;
		return true;
	} else if (_model != NULL) {
		cout << "Unrecognized model type.  Model unchanged, still ";
		cout << model_type + " model." << endl;
		return false;
	} else {
		cout << "Unrecognized model type." << endl;
		return false;
	}
}

// _generate_text(int size)
// Ask the model object to generate a random text of the requested size.  Note
// that size is interpreted differently depending on the model - it can mean
// either the number of characters or number of words to generate.
void markov::_generate_text(int size) {
	initialize_model();
	init_random();
        _mark_start();
	string text = _model->generate(size);
	_mark_stop();
	cout << endl << "GENERATED TEXT:" << endl;
	cout << text << endl << endl;
	if (_model_type == "word") {
		_report_time(string("Generated ") + to_string(size) + " words");
	} else {
		_report_time(string("Generated ") + to_string(size) + " characters");
	}
	cout << endl;
}

// _mark_start()
// Record the starting processor time.
void markov::_mark_start() {
	_start_time = clock();
}

// _mark_stop()
// Record the ending processor time.
void markov::_mark_stop() {
	_stop_time = clock();
}

// _report_time()
// Output a report of elapsed time.
void markov::_report_time(string action) {
	cout << action;
	cout << " in ";
	cout << (_stop_time - _start_time) / double(CLOCKS_PER_SEC);
	cout << " seconds.";
	cout << endl;
}

