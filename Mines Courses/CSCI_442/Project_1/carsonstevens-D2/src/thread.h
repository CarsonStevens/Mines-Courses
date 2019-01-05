#include <iostream>
#include <vector>
#include <queue>
#include "burst.h"

using namespace std;

#pragma once


class Thread{
    public:
        enum State {NEW, READY, RUNNING, BLOCKED, EXIT} state;
        int arrival_time;
        int thread_switch_overhead;
        queue <Burst> bursts;
        vector <Burst> burstVec;
        queue <Burst> blocked;
        int threadID;
        int waitTime;
        int startTime;
        int responseTime;
        int endTime;
        int turnaroundTime;
        int serviceTime;
        int IOTime;
        Thread(){
            
        }
        
        Thread(int arrival_time, int thread_switch_overhead, int threadID){
            this->arrival_time = arrival_time;
            state = NEW;
            this->thread_switch_overhead = thread_switch_overhead;
            this->threadID = threadID;
            
        }
        
        void addBlocked(Burst burst){
            blocked.push(burst);
        }
        
        void addBurst(Burst burst){
            bursts.push(burst);
            burstVec.push_back(burst);
        }
        
        queue <Burst> getBurstQueue(){
            return bursts;
        }
        
        void setWaitTime(int clock_){
            waitTime = clock_ - arrival_time;
            responseTime = startTime - arrival_time;
        }
        
        void setTurnaroundTime(){
            turnaroundTime = endTime - startTime;
        }
        
};
