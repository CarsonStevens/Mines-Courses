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

//// Recursive addition function for recursion for Cilk Plus
//int sum_array(int randoms[], int size, int sum){
//    if(size == 0){
//        return sum;
//    }
//    else{
//        return sum += add_array(randoms[size-1], size-1, sum);
//    }
//}
//
//// Recursive loop to find the max value
//int find_max(int randoms[], int size, int max_number){
//    // if n = 0 means whole array has been traversed
//    if (size == 1)
//        return max_number;
//    return max_number = max(randoms[size-1], find_max(randoms, size-1, max_number));
//}

string speedup(double baseline_duration, double duration, int repetitions){
    double speedup = baseline_duration/duration;
    return to_string(speedup) + " times";
}



//Main for testing different parallel methods
int main( int argc, char* argv[] ){
    int repetitions = 100000;                                             // Variable to define number of simulation runs                                                     // Iterator to stop simulation repetitions
    int stopper = 0;
    srand(time(0));                                                     // Set random generator seed to time
    int n = strtol(argv[1], nullptr, 0);                                // Get user input for number of random numbers
                                                                        //  to generate
    int sum = 0;                                                        // To hold to sum of the array
    int baseline_sum = 0;                                               // To test sum accuracy
    int max_number = 0;                                                 // To hold the maximum generated value
    int baseline_max = 0;                                               // To test max accuracy
    int random_numbers[n];                                              // To hold the array of random numbers
    int size = n;
    //sizeof(random_numbers[])/sizeof(random_numbers[0]);               // Get the size of each array element


    //BASELINE
    //Generating random numbers and storing in array. 'cilk_for' used to generate multiple threads.
    cilk_for(int i = 0; i < n; i++){
        random_numbers[i] = rand()%1000 + 1;
    }
    auto start = chrono::high_resolution_clock::now();
    while (stopper <= repetitions){
        baseline_sum = 0;
        for(int i = 0; i < n; i++){
            baseline_sum += random_numbers[i];
            if(random_numbers[i] > baseline_max){
                baseline_max = random_numbers[i];
            }
        }
        stopper++;
    }
    //Baseline code to see improvement from parallelization
    auto stop = chrono::high_resolution_clock::now();
    auto baseline_duration = stop - start;
    cout << "Baseline stats:\n\tSum: " << baseline_sum << "\n\tMax: " << baseline_max << "\n\tSpeedup: " <<
         speedup(chrono::duration <double, milli> (baseline_duration).count(), chrono::duration <double, milli>
         (baseline_duration).count(), repetitions) << endl;
    //Reset values for next test
    stopper = 0;
    sum = 0;
    cilk_for(int i = 0; i < n; i++){
        random_numbers[i] = rand()%1000 + 1;
    }


//    //METHOD: RECURSION
//    start = chrono::high_resolution_clock::now();
//    //Parallel, separate functions.
//    while(stopper <= repetitions){
//        //Get sum recursion
//        sum = cilk_spawn add(random_numbers, size, sum);
//        //Get max recursion
//        max_number = find_max(random_numbers, size, max_number);
//        cilk_sync;
//        sum = 0;
//        stopper++;
//    }
//    stop = chrono::high_resolution_clock::now();
//    auto duration = stop - start;
//    cout << "Recursion stats:\n\tSum: " << sum << "\n\tMax: " << max_number << "\n\tSpeedup: " <<
//         speedup(chrono::duration <double, milli> (baseline_duration).count(), chrono::duration <double, milli>
//         (duration).count(), repetitions) << endl;
//    //Reset values for next test.
//    sum = 0;
//    max_number = 0;
//    stopper = 0;
//    cilk_for(int i = 0; i < n; i++){
//        random_numbers[i] = rand()%1000 + 1;
//    }



    //METHOD: cilk_for
    start = chrono::high_resolution_clock::now();
    // In parallel, but might lose efficiency depending on if it puts a locks on max when it doesn't need a lock.
    while(stopper <= repetitions){
        sum = 0;
        cilk_for(int i = 0; i < n; i++){
            sum += random_numbers[i];
            if (random_numbers[i] > max_number) {
                max_number = random_numbers[i];
            }
            stopper++;

        }
    }
    stop = chrono::high_resolution_clock::now();
    auto duration = stop - start;
    cout << "ilk_for stats:\n\tSum: " << sum << "\n\tMax: " << max_number << "\n\tSpeedup: " <<
         speedup(chrono::duration <double, milli> (baseline_duration).count(), chrono::duration <double, milli>
         (duration).count(), repetitions) << endl;
    //Reset values for next test.
    sum = 0;
    max_number = 0;
    stopper = 0;
    cilk_for(int i = 0; i < n; i++){
        random_numbers[i] = rand()%1000 + 1;
    }




    //METHOD: Built-in functions
    start = chrono::high_resolution_clock::now();
    /* While the mutex guarantees that the access is thread-safe, it doesn't make any guarantees about ordering,
     * so the resulting list will be jumbled and different on every run with more than 1 worker.
     */
    while(stopper <= repetitions){
        sum = 0;
        //Built-in Notation
        sum = cilk_spawn(__sec_reduce_add(random_numbers[:]));
        //Built-in notation
        max_number = __sec_reduce_max(random_numbers[:]);
        cilk_sync;
        stopper++;
    }
    stop = chrono::high_resolution_clock::now();
    duration = stop - start;
    cout << "Built-in stats:\n\tSum: " << sum << "\n\tMax: " << max_number << "\n\tSpeedup: " <<
         speedup(chrono::duration <double, milli> (baseline_duration).count(), chrono::duration <double, milli>
         (duration).count(), repetitions) << endl;


    //Print output results
    cout << "Maximum: " << max_number << "Sum:	" << sum << endl;



    return 0;
}