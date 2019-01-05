#include <iostream>
#include <vector>
#include <queue>
#include "thread.h"

using namespace std;

#pragma once


class Process{
    public:
        enum Type {SYSTEM = 0, INTERACTIVE = 1, NORMAL = 2, BATCH = 3};
        int processID;
        int process_switch_overhead;
        int arrival_time;
        int timeTotal;
        int processType;
        //thread ready queue
        queue <Thread> threads;
        vector<Thread> threadVec;
        Type type;
        Process(){
            
        }
        
        Process(int processID, int type, int process_switch_overhead){
            processType = type;
            this->processID = processID;
            this->process_switch_overhead = process_switch_overhead;
            timeTotal = 0;
            switch(type){
                case 0 : this->type = Process::Type::SYSTEM;        break;
                case 1 : this->type = Process::Type::INTERACTIVE;   break;
                case 2 : this->type = Process::Type::NORMAL;        break;
                case 3 : this->type = Process::Type::BATCH;         break;
                default :                                           break;
            }
                
        }
        
        void addThread(Thread thread){
            //cout << "Adding thread to threads" << endl;
            threads.push(thread);
            threadVec.push_back(thread);
        }
        
        queue <Thread> getThreadQueue(){
            return threads;
        }
        
        Thread getNextThread(){
            Thread thread = threads.front();
            threads.pop();
            return thread;
        }
        
};
