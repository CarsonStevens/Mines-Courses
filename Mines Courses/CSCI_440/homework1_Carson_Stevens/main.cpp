/*
 * Author: Carson Stevens
 * Date: January 31, 2019
 * Description: 	1)	take a positive	integer	N as an	argument,
 *                  2)	create an integer array of size N,
 *                  3)	populate the array	with random	integers from range	[1,1000],
 *                  4)	find the largest integer and the sum of	the	array in parallel, and
 *                  5)	print the largest integer AND the sum of the array.
 */


#include <iostream>
#include <random>
#include <ctime>
#include <string>
#include <cilk/cilk.h>

using namespace std;

int main( int argc, char* argv[] ){

    // (1)
    int n = strtol(argv[1], nullptr, 0);                                // Get user input for number of random numbers
                                                                        //  to generate
    //Stat variables
    int sum = 0;                                                        // To hold to sum of the array
    int max_number = 0;                                                 // To hold the maximum generated value

    //Random variables
    srand(time(0));                                                     // Set random generator seed to time

    // (2)
    int random_numbers[n];                                              // To hold the array of random numbers

    // (3)
    //Generating random numbers and storing in array. 'cilk_for' used to generate multiple threads.
    cilk_for(int i = 0; i < n; i++){
        random_numbers[i] = rand()%1000 + 1;
    }

    // (4)
    //METHOD: cilk_for
    cilk_for(int i = 0; i < n; i++){
        sum += random_numbers[i];
        if (random_numbers[i] > max_number) {
            max_number = random_numbers[i];
        }
    }

    // (5)
    //Print output results
    cout << "Maximum: " << max_number << " Sum: " << sum << endl;

    return 0;
}