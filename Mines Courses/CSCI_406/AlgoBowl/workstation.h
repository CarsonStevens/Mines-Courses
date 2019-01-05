#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include "compare_arrival.h"
#include "compare_length.h"
#include "compare_processID.h"
#include "job.h"

using namespace std;

#pragma once


class Workstation{
    public:
        vector<int> job_indexes;    // Holds job at current time slots.
        vector<Job*> job_order;     // Holds order of current jobs in workstation
        int timer;          // Current time left on process
        bool running;       // True means that the workstation is busy
        int conflict;       // Used to detect loop, restarted each time a job is added
        int pq_length_size;     // Stores the current size of pq_length when a new
                            // process is started. This is used to check for 
                            // Loops created in the queue where no job is valid
        
        //Priority arrival
        priority_queue<Job, vector<Job>, Compare_Arrival> pq_arrival;
        //Priority length
        priority_queue<Job, vector<Job>, Compare_Length> pq_length;
        //Priority processID
        priority_queue<Job, vector<Job>, Compare_ProcessID> pq_ID;
        
        
        
        // Constructor
        Workstation(){
            running = false;
            timer = 0;
            conflict = 0;
            pq_length_size = pq_length.size();
        }
        
        
        
        // Add jobs to arrival queue
        void addArrivalQueue(Job job){
                pq_arrival.push(job);
        }
        
        //Add jobs to processID queue
        void addIDQueue(){
            for(auto *job : job_order){
                pq_ID.push(*job);
            }
        }
        
        // Add new jobs to pq_length that are less than the current time
        void addLengthQueue(int currentTime){
            while((pq_arrival.top().arrival_time <= currentTime) & !(pq_arrival.empty())){
                pq_length.push(pq_arrival.top());
                pq_arrival.pop();
            }
            
            // Other possible variation if above doesn't work
            // vector<int> temp;   // Holds jobs rejected due to arrival time
            // for(int i = 0; i < this->pq_arrival.size(); i++){
            //     if(this->pq_arrival.top() <= currentTime){
            //         pq_length.push(pq_arrival.pop());
            //     }
            //     else{
            //         // Remove so if statement always checks 'real' front
            //         temp.push_back(pq_length.pop());
            //     }
            // }
            // // Add rejected elements back to pq_arrival
            // for(int i = 0; i < temp.size(); i++){
            //     pq_arrival.push(temp.pop());
            // }
        }
        
        // Checks to see if the workstation has completed all its processes
        bool check(){
            if(!(pq_arrival.empty()) && pq_length.empty()){
                return true;
            }
            else{
                return false;
            }
        }
        
        // Checks the validity of the add and if valid adds, if not,
        // checks for a loop, if a full period of a loop has gone by,
        // the function returns falls, that addNext() is no longer valid.
        bool addNext(Workstation w_other1, Workstation w_other2, int timeTotal){
            bool add = true;    // Bool to check that an addition is valid for
                                // the current state of the workstations.
            
            // Won't matter if job_indexes.size() > because running will be false
            if(!(running) && !(pq_length.empty())){
                
                // CASE: All of the queues are the same size, so no conflicts, add
                if(job_indexes.size() == w_other1.job_indexes.size() || 
                    job_indexes.size() == w_other2.job_indexes.size()){
                        add = true;
                }
                
                // CASE: Queue is less that both job_index.size, itterate from
                //       current possition length of new possible process checking
                //       each position to see if its valid to insert there to either
                //       the length of the new process or to the end of the other
                //       queue
                if((job_indexes.size() < w_other1.job_indexes.size()) &&
                    (job_indexes.size() < w_other2.job_indexes.size())){
                        int check = pq_length.top().processID;
                        for(int i = job_indexes.size(); (i < w_other1.job_indexes.size()) && (i < (job_indexes.size() + pq_length.top().length)); i++){
                            if(w_other1.job_indexes.at(i) == check){
                                add = false;
                                break;
                            }
                        }
                        for(int i = job_indexes.size(); (i < w_other2.job_indexes.size()) && (i < (job_indexes.size() + pq_length.top().length)); i++){
                            if(w_other2.job_indexes.at(i) == check){
                                add = false;
                                break;
                            }
                        }
                }
                
                // CASE: Queue is less that w_other1.job_index.size, itterate from
                //       current possition length of new possible process checking
                //       each position to see if its valid to insert there to either
                //       the length of the new process or to the end of the other
                //       queue
                if((job_indexes.size() <= w_other1.job_indexes.size()) &&
                    (job_indexes.size() == w_other2.job_indexes.size())){
                        int check = pq_length.top().processID;
                        for(int i = job_indexes.size(); (i < w_other1.job_indexes.size()) && (i < (job_indexes.size() + pq_length.top().length)); i++){
                            if(w_other1.job_indexes.at(i) == check){
                                add = false;
                                break;
                            }
                        }
                }
                
                // CASE: Queue is less that w_other2.job_index.size, itterate from
                //       current possition length of new possible process checking
                //       each position to see if its valid to insert there to either
                //       the length of the new process or to the end of the other
                //       queue
                if((job_indexes.size() <= w_other2.job_indexes.size()) &&
                    (job_indexes.size() == w_other1.job_indexes.size())){
                        int check = pq_length.top().processID;
                        for(int i = job_indexes.size(); (i < w_other2.job_indexes.size()) && (i < (job_indexes.size() + pq_length.top().length)); i++){
                            if(w_other1.job_indexes.at(i) == check){
                                add = false;
                                break;
                            }
                        }
                }
                
                // add is still valid after checking cases, now add the new job
                if(add){
                    Job jobToAdd = pq_length.top();
                    pq_length.pop();
                    
                    // Set Job Start time
                    jobToAdd.start_time = timeTotal;
                    
                    // Add to ID to job_index length
                    for(int i = 0; i < jobToAdd.length; i ++){
                        job_indexes.push_back(jobToAdd.processID);
                    }
                    
                    // Add to job_order vector
                    Job *pointerJob = &jobToAdd;
                    job_order.push_back(pointerJob);
                    
                    // Set timer for new job
                    timer = jobToAdd.length;
                    
                    // Set workstation to running
                    running = true;
                    
                    // New possible loop, reset conflicts
                    conflict = 0;
                    
                    // Set new loop checking length
                    pq_length_size = pq_length.size();
                    return false;
                }
                
                else{
                    // Handles conflict loop of no entries working. Means that 
                    // Every possible entry in pq_length failed and only pq_length
                    // values are in there now. Add space.
                    if((conflict == pq_length_size) && (pq_arrival.size() == pq_length_size)){
                        job_indexes.push_back(-1);
                    }
                    conflict++;
                    pq_arrival.push(pq_length.top());
                    pq_length.pop();
                    return true;
                }
            }
            
            // Job is running or workstation is complete
            else{
                return false;
            }
            
        }
        
        // Update timer and state for one time cycle
        void update(int timeTotal){
            timer--;
            if(timer == 0){
                running = false;
                (job_order.at(job_order.size()-1))->end_time = timeTotal;
            }
        }
        
        bool getState(){
            return running;
        }
        
        int finalTime(){
            return job_order.at(job_order.size()-1)->end_time;
        }
};
