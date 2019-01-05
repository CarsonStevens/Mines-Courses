#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include "burst.h"
#include "thread.h"
#include "process.h"
#include "event.h"
// #include "compare_arrival.h"
// #include "compare_priority.h"

using namespace std;

#pragma once

class CPU{
    public:
    
        struct Compare_Arrival{
            int operator() (const Process process1, const Process process2) { 
                return process1.arrival_time < process2.arrival_time; 
            }
        };
        
        struct Compare_Priority{
            int operator() (const Process& process1, const Process& process2) { 
                return process1.type > process2.type; 
            }
        };
        
        priority_queue<Event, vector<Event>, Event::Compare_Event> event_queue;
        priority_queue<Process, vector<Process>, Compare_Arrival> readyProcess;

        
        int clock_;
        double minTime;
        int cpu_idle_time;
        Process currentProcess;
        int eventNum;
        int serviceTimeTotal = 0;
        int IOTimeTotal = 0;
        int dispatchTotal = 0;
        vector<Process> allProcess;
        vector<Process> systemProcess;
        vector<Process> interactiveProcess;
        vector<Process> batchProcess;
        vector<Process> normalProcess;
        CPU(){
            cpu_idle_time = 0;
            clock_ = 0;
            eventNum = 1;
        }
        
        //Load proccesses to CPU;
        void processToCPU(Process process){
            allProcess.push_back(process);
            readyProcess.push(process);
            // Adds threads through vector (only used for this)
            for(Thread thread : process.threadVec){
                int serviceTime = 0;
                int IOTime = 0;

                for(Burst burst : thread.burstVec){
                    serviceTime += burst.cpu_time;
                    IOTime += burst.io_time;
                }
                thread.serviceTime = serviceTime;
                serviceTimeTotal += serviceTime;
                thread.IOTime = IOTime;
                IOTimeTotal += IOTime;
                Event event(process.type, process.processID, thread.threadID, thread.arrival_time, eventNum, Event::event_type::THREAD_ARRIVED);
                event_queue.push(event);
                ++eventNum;
                
            }
            
        }
        
        void printEventQueue(){
            while(!(event_queue.empty())){
                Event event = event_queue.top();
                event.getEvent();
                event_queue.pop();
            }
        }
        
        void addEvent(int type, int process, int thread, Event::event_type event_){
                //cout << clock_ << endl;
                Event event(type, process, thread, clock_, eventNum, event_);
                event_queue.push(event);
                ++eventNum;
        }
        
        void FirstComeFirstServe(){
            // Load next process
            while(!readyProcess.empty()){
                Process currentProcess = move(readyProcess.top());
                readyProcess.pop();
                addEvent(currentProcess.type, currentProcess.processID, 0, Event::event_type::DISPATCHER_INVOKED);
                clock_ += currentProcess.process_switch_overhead;
                dispatchTotal += currentProcess.process_switch_overhead;
                addEvent(currentProcess.type, currentProcess.processID, 0, Event::event_type::PROCESS_DISPATCH_COMPLETED);
                
                //Load thread for process in order
                while(!currentProcess.getThreadQueue().empty()){
                    //cout << clock_ << endl;
                    Thread currentThread = move(currentProcess.threads.front());
                    addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::DISPATCHER_INVOKED);
                    currentProcess.threads.pop();
                    clock_ += currentThread.thread_switch_overhead;
                    dispatchTotal += currentThread.thread_switch_overhead;
                    currentThread.startTime = clock_;
                    currentThread.setWaitTime(clock_);
                    addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::THREAD_DISPATCH_COMPLETED);
                    
                    
                    // ONly for first burst in each thread
                    Burst currentCPUBurst = move(currentThread.bursts.front());
                    currentThread.bursts.pop();
                    cout << currentCPUBurst.cpu_time << endl;
                    clock_ += currentCPUBurst.cpu_time;
                    currentCPUBurst.setCPUTime(0);
                    addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::CPU_BURST_COMPLETED);
                    currentThread.blocked.push(currentCPUBurst);
                    
                    
                    while(!currentThread.getBurstQueue().empty() && !currentThread.blocked.empty()){
                        if(currentThread.bursts.empty() && !currentThread.blocked.empty()){
                            Burst currentIOBurst = move(currentThread.blocked.front());
                            currentThread.blocked.pop();
                            clock_ += currentIOBurst.io_time;
                            cpu_idle_time += currentIOBurst.io_time;
                            addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::IO_BURST_COMPLETED);
                        }
                        else if(!currentThread.bursts.empty() && currentThread.blocked.empty()){
                            Burst currentCPUBurst = move(currentThread.bursts.front());
                            currentThread.bursts.pop();
                            clock_ += currentCPUBurst.cpu_time;
                            currentCPUBurst.setCPUTime(0);
                            currentThread.blocked.push(currentCPUBurst);
                            addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::CPU_BURST_COMPLETED);
                        }
                        else if(!currentThread.getBurstQueue().empty() && !currentThread.blocked.empty()){
                            Burst currentCPUBurst = move(currentThread.bursts.front());
                            Burst currentIOBurst = move(currentThread.blocked.front());
                            minTime = min(currentCPUBurst.getCPUTime(), currentIOBurst.getIOTime());
                            
                            if(currentCPUBurst.cpu_time == minTime && currentIOBurst.io_time == minTime){
                                clock_ += currentCPUBurst.cpu_time;
                                currentThread.bursts.pop();
                                addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::CPU_BURST_COMPLETED);
                                currentCPUBurst.setCPUTime(0);
                                currentThread.blocked.push(currentCPUBurst);
                                currentThread.blocked.pop();
                                currentIOBurst.io_time = 0;
                                addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::IO_BURST_COMPLETED);
                            }
                            else if(currentCPUBurst.getCPUTime() == minTime){
                                currentThread.bursts.pop();
                                clock_ += currentCPUBurst.cpu_time;
                                currentIOBurst.io_time -= currentCPUBurst.cpu_time;
                                currentCPUBurst.setCPUTime(0);
                                currentThread.addBlocked(currentCPUBurst);
                                addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::CPU_BURST_COMPLETED);
                            }
                            else if(currentIOBurst.getIOTime() == minTime){
                                currentThread.blocked.pop();
                                clock_ += currentIOBurst.io_time;
                                currentCPUBurst.cpu_time -= currentIOBurst.io_time;
                                addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::IO_BURST_COMPLETED);
                            }
                        }
                            
                        
                    }
                    currentThread.endTime = clock_;
                    currentThread.setTurnaroundTime();
                    addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::THREAD_COMPLETED);
                }
                if(getType(currentProcess.processType) == "SYSTEM"){
                    systemProcess.push_back(currentProcess);
                }
                if(getType(currentProcess.processType) == "INTERACTIVE"){
                    interactiveProcess.push_back(currentProcess);
                }
                if(getType(currentProcess.processType) == "NORMAL"){
                    normalProcess.push_back(currentProcess);
                }
                if(getType(currentProcess.processType) == "BATCH"){
                    batchProcess.push_back(currentProcess);
                }
            }
        }
        
        void printStats(){
            cout << "SIMULATION COMPLETED!" << endl;
            int threadCount = 0;
            int responseTime = 0;
            int turnaroundTime = 0;
            for(Process process : systemProcess){
                for(Thread thread : process.threadVec){
                    threadCount++;
                    responseTime += thread.responseTime;
                    turnaroundTime += thread.turnaroundTime;
                }
                if(threadCount == 0){
                    responseTime = 0;
                    turnaroundTime = 0;
                }
                else{
                    responseTime = responseTime/threadCount;
                    turnaroundTime = turnaroundTime/threadCount;
                }
            }
            cout << endl << "SYSTEM THREADS:" << endl << "\tTotal count:\t" << threadCount<< endl << "\tAvg response time:\t" << responseTime << endl;
            cout << "\tAvg turnaround time:\t" << turnaroundTime << endl << endl;
            
            threadCount = 0;
            responseTime = 0;
            turnaroundTime = 0;
            for(Process process : interactiveProcess){

                for(Thread thread : process.threadVec){
                    threadCount++;
                    responseTime += thread.responseTime;
                    turnaroundTime += thread.turnaroundTime;
                }
                if(threadCount == 0){
                    responseTime = 0;
                    turnaroundTime = 0;
                }
                else{
                    responseTime = responseTime/threadCount;
                    turnaroundTime = turnaroundTime/threadCount;
                }
            }
            cout << endl << "INTERACTIVE THREADS:" << endl << "\tTotal count:\t" << threadCount<< endl << "\tAvg response time:\t" << responseTime << endl;
            cout << "\tAvg turnaround time:\t" << turnaroundTime << endl << endl;
            
            threadCount = 0;
            responseTime = 0;
            turnaroundTime = 0;
            for(Process process : normalProcess){

                for(Thread thread : process.threadVec){
                    threadCount++;
                    responseTime += thread.responseTime;
                    turnaroundTime += thread.turnaroundTime;
                }
                if(threadCount == 0){
                    responseTime = 0;
                    turnaroundTime = 0;
                }
                else{
                    responseTime = responseTime/threadCount;
                    turnaroundTime = turnaroundTime/threadCount;
                }
            }
            cout << endl << "NORMAL THREADS:" << endl << "\tTotal count:\t" << threadCount<< endl << "\tAvg response time:\t" << responseTime << endl;
            cout << "\tAvg turnaround time:\t" << turnaroundTime << endl << endl;
            
            threadCount = 0;
            responseTime = 0;
            turnaroundTime = 0;
            for(Process process : batchProcess){

                for(Thread thread : process.threadVec){
                    threadCount++;
                    responseTime += thread.responseTime;
                    turnaroundTime += thread.turnaroundTime;
                }
                if(threadCount == 0){
                    responseTime = 0;
                    turnaroundTime = 0;
                }
                else{
                    responseTime = responseTime/threadCount;
                    turnaroundTime = turnaroundTime/threadCount;
                }
                
            }
            cout << endl << "BATCH THREADS:" << endl << "\tTotal count:\t" << threadCount<< endl << "\tAvg response time:\t" << responseTime << endl;
            cout << "\tAvg turnaround time:\t" << turnaroundTime << endl << endl;
            cout << "Total elapsed time:\t" << clock_ << endl;
            cout << "Total service time:\t" << to_string(serviceTimeTotal) << endl;
            cout << "Total I/O time:\t" << to_string(IOTimeTotal) << endl;
            cout << "Total dispatch time:\t" << dispatchTotal << endl;
            cout << "Total idle time:\t" << cpu_idle_time << endl << endl;
            cout << "CPU utilization:\t" << to_string((cpu_idle_time/clock_) * 100) << "%" << endl;
            cout << "CPU efficiency:\t" << to_string((serviceTimeTotal/clock_) * 100) << "%" << endl<< endl<< endl;

        }
        
        void perThread(){
            for(Process process : allProcess){
                cout << "Process " << process.processID << " [" << getType(process.processType) << "]" << endl;
                for(Thread thread : process.threadVec){
                    cout << "Thread " << thread.threadID << "\tARR: " << thread.arrival_time << "\tCPU: " << thread.serviceTime << "\tI/O: "<< thread.IOTime << "\tTRT: " << thread.turnaroundTime << "\tEND: " << thread.endTime << endl;
                }
                cout << endl;
            }
        }
        
        string getType(int type){
            if(type == 0){
                return "SYSTEM";
            }
            if(type == 1){
                return "INTERACTIVE";
            }
            if(type == 2){
                return "NORMAL";
            }
            if(type == 3){
                return "BATCH";
            }
        }
};
