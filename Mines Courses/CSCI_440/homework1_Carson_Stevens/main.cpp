/*
 * Author: Carson Stevens
 * Date: January 31, 2019
 * Description: 	1)	take a positive	integer	N as an	argument,
 *                  2)	create an integer array of size N,
 *                  3)	populate the array	with random	integers from range	[1,1000],
 *                  4)	find the largest integer and the sum of	the	array in parallel, and
 *                  5)	print the largest integer AND the sum of the array.
 *                  EXTRA: Metrics for seeing efficiency.
 */


#include <iostream>
#include <random>
#include <ctime>
#include <chrono>
#include <string>
#include <cilk/cilk.h>

using namespace std;

string speedup(double baseline_duration, double duration){
    double speedup = baseline_duration/duration;
    return to_string(speedup) + " times";
}


//void granular_cilk_for(int random_numbers[], int first, int last, int grain_size, int sum, int max_number){
//    if(last - first < grain_size){
//        //METHOD: cilk_for with larger granularity
//
//        // In parallel, but might lose efficiency depending on if it puts a locks on max when it doesn't need a lock.
//        cilk_for(int i = first; i < last; i++){
//            sum += random_numbers[i];
//            if (random_numbers[i] > max_number) {
//                max_number = random_numbers[i];
//            }
//        }
//        return "cilk_for stats:\n\tSum: " << sum << "\n\tMax: " << max_number;
//
//    }
//    else{
//        int mid = (last+first/2);
//        cilk_spawn granular_cilk_for(random_numbers, first, mid, grain_size, sum, max_number);
//        granular_cilk_for(random_numbers, mid, last, grain_size, sum, max_number);
//    }
//}


//Main for testing different parallel methods
int main( int argc, char* argv[] ){

    //get
    int n = strtol(argv[1], nullptr, 0);                                // Get user input for number of random numbers
                                                                        //  to generate
    //Stat variables
    int sum = 0;                                                        // To hold to sum of the array
    int baseline_sum = 0;                                               // To test sum accuracy
    int max_number = 0;                                                 // To hold the maximum generated value
    int baseline_max = 0;                                               // To test max accuracy

    //Random variables
    srand(time(0));                                                     // Set random generator seed to time
    int random_numbers[n];                                              // To hold the array of random numbers

    //BASELINE
    //Generating random numbers and storing in array. 'cilk_for' used to generate multiple threads.
    cilk_for(int i = 0; i < n; i++){
        random_numbers[i] = rand()%1000 + 1;
    }
    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < n; i++){
        baseline_sum += random_numbers[i];
        if(random_numbers[i] > baseline_max){
            baseline_max = random_numbers[i];
        }
    }
    //Baseline code to see improvement from parallelization
    auto stop = chrono::high_resolution_clock::now();
    auto baseline_duration = stop - start;
    cout << "Baseline stats:\n\tSum: " << baseline_sum << "\n\tMax: " << baseline_max << "\n\tSpeedup: " <<
         speedup(chrono::duration <double, milli> (baseline_duration).count(), chrono::duration <double, milli>
         (baseline_duration).count()) << endl << endl;
    //Reset values for next test
    sum = 0;
    cilk_for(int i = 0; i < n; i++){
        random_numbers[i] = rand()%1000 + 1;
    }


    //METHOD: cilk_for
    start = chrono::high_resolution_clock::now();
    // In parallel, but might lose efficiency depending on if it puts a locks on max when it doesn't need a lock.
    cilk_for(int i = 0; i < n; i++){
        sum += random_numbers[i];
        if (random_numbers[i] > max_number) {
            max_number = random_numbers[i];
        }
    }
    stop = chrono::high_resolution_clock::now();
    auto duration = stop - start;
    cout << "cilk_for stats:\n\tSum: " << sum << "\n\tMax: " << max_number << "\n\tSpeedup: " <<
         speedup(chrono::duration <double, milli> (baseline_duration).count(), chrono::duration <double, milli>
         (duration).count()) << endl << endl;
    //Reset values for next test.
    sum = 0;
    max_number = 0;
    cilk_for(int i = 0; i < n; i++){
        random_numbers[i] = rand()%1000 + 1;
    }


//    //METHOD: Granulated cilk_for
//    int grain_size = 5;
//    start = chrono::high_resolution_clock::now();
//    granular_cilk_for(random_numbers, 0, n, grain_size, 0, 0);
//    stop = chrono::high_resolution_clock::now();
//    duration += stop - start;
//    cout << "\n\tSpeedup: " << speedup(chrono::duration <double, milli> (baseline_duration).count(), chrono::duration <double, milli>
//            (duration).count()) << endl;


    //METHOD: Built-in functions
    /* While the mutex guarantees that the access is thread-safe, it doesn't make any guarantees about ordering,
     * so the resulting list will be jumbled and different on every run with more than 1 worker.
     */
    start = chrono::high_resolution_clock::now();

    //Parallel Part: Built-in Notation
    sum = cilk_spawn(__sec_reduce_add(random_numbers[:]));
    max_number = __sec_reduce_max(random_numbers[:]);
    cilk_sync;

    stop = chrono::high_resolution_clock::now();
    duration = stop - start;
    cout << "Built-in stats:\n\tSum: " << sum << "\n\tMax: " << max_number << "\n\tSpeedup: " <<
         speedup(chrono::duration <double, milli> (baseline_duration).count(), chrono::duration <double, milli>
         (duration).count()) << endl << endl;


    //Print output results
    cout << "Maximum: " << max_number << " Sum:	" << sum << endl;



    return 0;
}