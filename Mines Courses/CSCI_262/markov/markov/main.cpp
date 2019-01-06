/*
   CSCI 262 Data Structures, Fall 2017, Project 4 - Markov

   main.cpp

   Contains the main() function, which handles command line arguments and
   launches the Markov application.

   Author: C. Painter-Wakefield

   Modified: 11/2/2017
*/

#include <iostream>
#include <cstdlib>
#include "markov.h"

using namespace std;

// main()
int main(int argc, char* argv[]) {
	markov app;

	// allow command line "markov filename order model size [seed]" - 
	// if these options are provided, user interaction will be skipped
	if (argc == 5) {
		app.run_one(argv[1], atoi(argv[2]), argv[3], atoi(argv[4]));
	} else if (argc == 6) {
		app.run_one(argv[1], atoi(argv[2]), argv[3], atoi(argv[4]), atoi(argv[5]));
	} else if (argc > 1) {
		cout << "usage: [filename order model size [seed]]" << endl;
	} else {
		app.run();
	}

	return 0;
}

