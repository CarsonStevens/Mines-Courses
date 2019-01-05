#include <iostream>


using namespace std;


#pragma once


class Job{
    public:
        int start_time;     // Time job starts
        int end_time;       // Time job ends
        int length;         // Length of time of job
        int arrival_time;   // Time job arrives
        int processID;      // Process(line #) job came from
        
        // Constructor; end_time, and start_time can't be defined yet.
        Job(int arrival_time, int length, int processID){
            this->length = length;
            this->arrival_time = arrival_time;
            this->processID = processID;
        }
        
        Job();
        
        // Job operator= (Job job){
        //     job.length = this->length;
        //     job.arrival_time = this->arrival_time;
        //     job.processID = this->processID;
        //     return job;
        // }
};