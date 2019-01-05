#include <iostream>
#include "job.h"

using namespace std;


#pragma once

//Max Heap comparison of job lengths.
class Compare_Length{
    public:
        int operator() (const Job& job1, const Job& job2) { 
            return job1.length < job2.length; 
        }
};