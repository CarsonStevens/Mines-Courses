#include <iostream>
#include <vector>
#include <queue>
#include "burst.h"

using namespace std;

#pragma once


class Thread{
    public:
    
        struct Compare_Priority{
            int operator() (const Burst& burst1, const Burst& burst2) { 
                return burst1.priority > burst2.priority; 
            }
        };
        
    
        enum State {NEW, READY, RUNNING, BLOCKED, EXIT} state;
        int arrival_time;
        int thread_switch_overhead;
        queue <Burst> bursts;
        priority_queue<Burst, vector<Burst>, Compare_Priority> rrBurstQueue;
        priority_queue<Burst, vector<Burst>, Compare_Priority> rrBurstBlockedQueue;
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
        bool started;
        bool wait;
        bool response;
        int processID;
        int timeLeft;
        int type;
        Thread(){
            
        }
        
        Thread(int arrival_time, int thread_switch_overhead, int threadID){
            this->arrival_time = arrival_time;
            state = NEW;
            this->thread_switch_overhead = thread_switch_overhead;
            this->threadID = threadID;
            started = false;
            wait = true;
            response = true;
            
        }
        
        void addBlocked(Burst burst){
            rrBurstBlockedQueue.push(burst);
            blocked.push(burst);
        }
        
        void addBurst(Burst burst){
            bursts.push(burst);
            rrBurstQueue.push(burst);
            burstVec.push_back(burst);
        }
        
        queue <Burst> getBurstQueue(){
            return bursts;
        }
        
        void setWaitTime(int clock_){
            if(wait){
                waitTime = clock_ - arrival_time;
                wait = false;
            }
            if(response){
                responseTime = startTime - arrival_time;
                response = false;
            }
        }
        
        void setTurnaroundTime(){
            turnaroundTime = endTime - startTime;
        }
        
};
