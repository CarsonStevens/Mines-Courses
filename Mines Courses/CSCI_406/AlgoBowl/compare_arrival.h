#include <iostream>
#include "job.h"

using namespace std;


#pragma once

// Min Heap comparison of arrival_times
class Compare_Arrival{
    public:
        int operator() (const Job& job1, const Job& job2) { 
            return job1.arrival_time > job2.arrival_time; 
        }
};