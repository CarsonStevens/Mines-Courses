/*
   CSCI 262 Data Structures, Fall 2017, Project 4 - Markov

   markov.h

   Class declaration for the markov class, which implements the markov
   application.

   Author: C. Painter-Wakefield

   Modified: 11/2/2017
*/

#ifndef _MARKOV_H
#define _MARKOV_H

#include <string>
#include <ctime>
#include <map>

#include "model.h"

class markov {
public:
	markov();
	~markov();

    void run();
	void run_one(
	std::string infile, 
	int order, 
	std::string model_type, 
	int size, 
	int seed = -1);
	void set_text();
	void set_order();
	void set_model();
	void set_random_seed();
	void init_random();
	void initialize_model();
	void generate_text();

private:
	std::map<std::string, markov_model*> _model_map;

	std::string   _text_file    = "";
	std::string   _text         = "";
    int           _order        = -1;
    std::string   _model_type   = "";
	markov_model* _model        = NULL;
	int           _random_seed  = -1;
	bool          _initialized  = false;
	clock_t       _start_time;
	clock_t       _stop_time;

	bool _set_text(std::string infile);
	bool _set_model(std::string model_type);
	void _generate_text(int size);
	void _mark_start();
	void _mark_stop();
	void _report_time(std::string action);
};

#endif
