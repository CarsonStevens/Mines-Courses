#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include "burst.h"
#include "thread.h"
#include "process.h"
#include "event.h"


using namespace std;

#pragma once

class CPU{
    public:
    
        ////////////////////////////////////////////////////////////////////
        //                  Structs for Priority Queues                   //
        ////////////////////////////////////////////////////////////////////
        struct Compare_Arrival{
            int operator() (const Process process1, const Process process2) { 
                return process1.arrival_time < process2.arrival_time; 
            }
        };
        
        struct Compare_Arrival_Thread{
            int operator() (const Thread thread1, const Thread thread2) { 
                return thread1.arrival_time < thread2.arrival_time; 
            }
        };
        
        struct Compare_Length{
            int operator() (const Thread thread1, const Thread thread2) { 
                return thread1.timeLeft < thread2.timeLeft; 
            }
        };
        
        struct Compare_Priority{
            int operator() (const Process& process1, const Process& process2) { 
                return process1.type > process2.type; 
            }
        };
        
        struct Compare_Priority_Thread{
            int operator() (const Thread& thread1, const Thread& thread2) { 
                return thread1.type > thread2.type; 
            }
        };
        
        ///////////////////////////////////////////////////////////////////////
        
        
        
        //Event Calender
        priority_queue<Event, vector<Event>, Event::Compare_Event> event_queue;
        
        //Used for FCFS
        priority_queue<Process, vector<Process>, Compare_Arrival> readyProcess;
        priority_queue<Process, vector<Process>, Compare_Priority> readyProcessPriority;
        
        //Used for Custom
        priority_queue<Thread, vector<Thread>, Compare_Length> STNqueue;
        priority_queue<Thread, vector<Thread>, Compare_Length> STNBlockedqueue;
        priority_queue<Thread, vector<Thread>, Compare_Priority_Thread> priorityThreadQueue;
        priority_queue<Thread, vector<Thread>, Compare_Priority_Thread> priorityThreadBlockedQueue;
        
        //Used for Priority
        priority_queue<Thread, vector<Thread>, Compare_Arrival_Thread> readyPriority0;
        priority_queue<Thread, vector<Thread>, Compare_Arrival_Thread> readyPriority1;
        priority_queue<Thread, vector<Thread>, Compare_Arrival_Thread> readyPriority2;
        priority_queue<Thread, vector<Thread>, Compare_Arrival_Thread> readyPriority3;

        //Variables for Stats
        int clock_;
        double minTime;
        int cpu_idle_time;
        Process currentProcess;
        int eventNum;
        int serviceTimeTotal = 0;
        int IOTimeTotal = 0;
        int dispatchTotal = 0;
        
        //Used to store processes and types of processes
        vector<Process> allProcess;
        vector<Process> systemProcess;
        vector<Process> interactiveProcess;
        vector<Process> batchProcess;
        vector<Process> normalProcess;
        
        //Used in RR
        queue<Thread> readyThreads;
        queue<Thread> blockedThreads;
        
        //For Custom
        int lastThreadType = 0;
        int lastProcessType = 0;
        int lastBlockedThreadType = 0;
        int lastBlockedProcessType = 0;
        

        
        CPU(){
            cpu_idle_time = 0;
            clock_ = 0;
            eventNum = 1;
        }
        
        //Load proccesses to CPU;
        void processToCPU(Process process){
            // readyProcessCustom.push(process);
            readyProcessPriority.push(process);
            readyProcess.push(process);
            // Adds threads through vector (only used for this)
            for(Thread thread : process.threadVec){
                serviceTimeTotal += thread.serviceTime;
                IOTimeTotal += thread.IOTime;
                Event event(process.type, process.processID, thread.threadID, thread.arrival_time, eventNum, Event::event_type::THREAD_ARRIVED);
                event_queue.push(event);
                ++eventNum;
                
            }
            //cout << "Service time total:\t" << serviceTimeTotal << endl; 
            //cout << "IO time total:\t" << IOTimeTotal << endl;
            
        }
        
        //Prints the events in order
        void printEventQueue(){
            while(!(event_queue.empty())){
                Event event = event_queue.top();
                event.getEvent();
                event_queue.pop();
            }
        }
        
        //Creates and adds event to event calender
        void addEvent(int type, int process, int thread, Event::event_type event_){
                //cout << clock_ << endl;
                Event event(type, process, thread, clock_, eventNum, event_);
                event_queue.push(event);
                ++eventNum;
        }
        
        //for "-t"
        void perThread(){
            for(Process process : allProcess){
                cout << "Process " << process.processID << " [" << getType(process.processType) << "]" << endl;
                for(Thread thread : process.threadVec){
                    cout << "Thread " << thread.threadID << "\tARR: " << thread.arrival_time << "\tCPU: " << thread.serviceTime << "\tI/O: "<< thread.IOTime << "\tTRT: " << thread.turnaroundTime << "\tEND: " << thread.endTime << endl;
                }
                cout << endl;
            }
        }
        
        //used to return name for comparison and printing
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
                    clock_ += currentCPUBurst.cpu_time;
                    currentCPUBurst.setCPUTime(0);
                    addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::CPU_BURST_COMPLETED);
                    currentThread.blocked.push(currentCPUBurst);
                    
                    //While thread has thinks to process
                    while(!currentThread.getBurstQueue().empty() && !currentThread.blocked.empty()){
                        
                        //Handle cases
                        
                        // IO Burst Completed
                        if(currentThread.bursts.empty() && !currentThread.blocked.empty()){
                            Burst currentIOBurst = move(currentThread.blocked.front());
                            currentThread.blocked.pop();
                            clock_ += currentIOBurst.io_time;
                            cpu_idle_time += currentIOBurst.io_time;
                            addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::IO_BURST_COMPLETED);
                        }
                        // CPU Burst Completed
                        else if(!currentThread.bursts.empty() && currentThread.blocked.empty()){
                            Burst currentCPUBurst = move(currentThread.bursts.front());
                            currentThread.bursts.pop();
                            clock_ += currentCPUBurst.cpu_time;
                            currentCPUBurst.setCPUTime(0);
                            currentThread.blocked.push(currentCPUBurst);
                            addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::CPU_BURST_COMPLETED);
                        }
                        
                        //Bursts aren't done
                        else if(!currentThread.getBurstQueue().empty() && !currentThread.blocked.empty()){
                            Burst currentCPUBurst = move(currentThread.bursts.front());
                            Burst currentIOBurst = move(currentThread.blocked.front());
                            minTime = min(currentCPUBurst.getCPUTime(), currentIOBurst.getIOTime());
                            
                            //Both cpu and IO finish at same time
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
                            //CPU finishes first
                            else if(currentCPUBurst.getCPUTime() == minTime){
                                currentThread.bursts.pop();
                                clock_ += currentCPUBurst.cpu_time;
                                currentIOBurst.io_time -= currentCPUBurst.cpu_time;
                                currentCPUBurst.setCPUTime(0);
                                currentThread.addBlocked(currentCPUBurst);
                                addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::CPU_BURST_COMPLETED);
                            }
                            //IO finishes finishes first
                            else if(currentIOBurst.getIOTime() == minTime){
                                currentThread.blocked.pop();
                                clock_ += currentIOBurst.io_time;
                                currentCPUBurst.cpu_time -= currentIOBurst.io_time;
                                addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::IO_BURST_COMPLETED);
                            }
                        }
                            
                        
                    }
                    
                    //Finished thread
                    currentThread.endTime = clock_;
                    currentThread.setTurnaroundTime();
                    addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::THREAD_COMPLETED);
                    
                    //Add for book keeping
                    if(currentProcess.threads.empty()){
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
                        allProcess.push_back(currentProcess);
                    }
                }
                
            }
        }
        
        
        
        void RoundRobin(){
            //Time quantum fixed at 3
            int quantum = 3;
            bool cpuDone = false;
            bool ioDone = false;
            int cpuDoneTime;
            int ioDoneTime;
            Process prime = move(readyProcess.top());
            clock_ = prime.arrival_time;
            while(!prime.threads.empty()){
                Thread thread = move(prime.threads.front());
                prime.threads.pop();
                readyThreads.push(thread);
            }
            
            //Load next process
            while(!blockedThreads.empty() && !readyThreads.empty()){

                quantum = 3;
                cpuDone = false;
                ioDone = false;
                //Grab Next Process
                if(!readyProcess.empty()){
                    Process currentProcess = move(readyProcess.top());
                    //No process or threads ready, so move clock to arrival of next event and add cpu idle time
                    if(currentProcess.arrival_time > clock_ && readyThreads.empty() && blockedThreads.empty()){
                        cpu_idle_time += currentProcess.arrival_time - clock_;
                        clock_ = currentProcess.arrival_time;
                        
                    }
                    
                    //ELSE we have things to process
                    readyProcess.pop();
                    addEvent(currentProcess.type, currentProcess.processID, 0, Event::event_type::DISPATCHER_INVOKED);
                    clock_ += currentProcess.process_switch_overhead;
                    dispatchTotal += currentProcess.process_switch_overhead;
                    addEvent(currentProcess.type, currentProcess.processID, 0, Event::event_type::PROCESS_DISPATCH_COMPLETED);
    
                    //Load all processes threads that have arrived into final waiting queue
                    while(!currentProcess.threads.empty()){
                        Thread currentThread = move(currentProcess.threads.front());
                        // done when processes are loaded in (sorted by arrival times, so okay).
                        currentProcess.threads.pop();
                        currentThread.startTime = clock_;
                        currentThread.setWaitTime(clock_);
                        readyThreads.push(currentThread);
                    }  
                }

                //Handle quantum
                
                //Get next thread to process
                if(!readyThreads.empty()){
                    Thread cpuThread = readyThreads.front();
                    cpuThread.setWaitTime(clock_);
                    addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::DISPATCHER_INVOKED);
                    if(cpuThread.threadID != lastThreadType){
                        addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::THREAD_PREEMPTED);
                        clock_ += cpuThread.thread_switch_overhead;
                        dispatchTotal += cpuThread.thread_switch_overhead;
                        addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::THREAD_DISPATCH_COMPLETED);
                    }
                    if(cpuThread.type != lastProcessType){
                        clock_ += currentProcess.process_switch_overhead;
                        dispatchTotal += currentProcess.process_switch_overhead;
                    }
                
                    lastThreadType = cpuThread.threadID;
                    lastProcessType = cpuThread.type;
                    readyThreads.pop();
                    
                    Burst cpuBurst = move(cpuThread.rrBurstQueue.top());
                    cpuThread.rrBurstQueue.pop();
                    //cout << cpuBurst.cpu_time << "\t" << cpuBurst.io_time << endl;
                    if(cpuBurst.cpu_time - quantum == 0){
                        cpuBurst.setCPUTime(0);
                        cpuBurst.priority = 0;
                        addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::CPU_BURST_COMPLETED);
                        if(cpuThread.rrBurstBlockedQueue.empty() && cpuThread.rrBurstQueue.empty()){
                            //Thread done
                            cpuThread.endTime = clock_ + quantum;
                            cpuThread.setTurnaroundTime();
                            addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::THREAD_COMPLETED);
                            //Process done
                            currentProcess.timeTotal = clock_ - currentProcess.arrival_time;
                            allProcess.push_back(currentProcess);
                        }
                        else{
                            cpuThread.rrBurstBlockedQueue.push(cpuBurst);
                            blockedThreads.push(cpuThread);
                        }
                        
                    }
                    if(cpuBurst.cpu_time - quantum > 0){
                        cpuBurst.cpu_time = cpuBurst.cpu_time - quantum;
                        cpuBurst.priority = 1;
                        addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::THREAD_PREEMPTED);
                        cpuThread.rrBurstQueue.push(cpuBurst);
                        readyThreads.push(cpuThread);
                    }
                    if(cpuBurst.cpu_time - quantum < 0){
                        cpuDone = true;
                        cpuDoneTime = clock_ + (cpuBurst.cpu_time - quantum);
                        quantum = quantum - cpuBurst.cpu_time;
                        cpuBurst.cpu_time = 0;
                        cpuBurst.priority = 0;
                        addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::CPU_BURST_COMPLETED);
                        if(cpuThread.rrBurstBlockedQueue.empty() && cpuThread.rrBurstQueue.empty()){
                            //Thread done
                            cpuThread.endTime = clock_ - (cpuBurst.cpu_time - quantum);
                            cpuThread.setTurnaroundTime();
                            addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::THREAD_COMPLETED);
                            //Process done
                            currentProcess.timeTotal = clock_ - currentProcess.arrival_time;
                            allProcess.push_back(currentProcess);
                        }
                        else{
                            cpuThread.rrBurstBlockedQueue.push(cpuBurst);
                            blockedThreads.push(cpuThread);
                        }
                    }
                }

                //Handle blocked queue for quantum
                if(!blockedThreads.empty()){
                    Thread ioThread = move(blockedThreads.front());
                    addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::DISPATCHER_INVOKED);
                    blockedThreads.pop();
                    if(ioThread.threadID != lastThreadType){
                        addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::THREAD_PREEMPTED);
                        clock_ += ioThread.thread_switch_overhead;
                        dispatchTotal += ioThread.thread_switch_overhead;
                        addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::THREAD_DISPATCH_COMPLETED);

                    }
                    if(ioThread.type != lastProcessType){
                        clock_ += currentProcess.process_switch_overhead;
                        dispatchTotal += currentProcess.process_switch_overhead;
                    }
                    lastBlockedThreadType = ioThread.threadID;
                    lastBlockedProcessType = ioThread.type;
                    
                    Burst ioBurst = move(ioThread.rrBurstBlockedQueue.top());
                    ioThread.rrBurstBlockedQueue.pop();
                    
                    if(ioBurst.io_time - quantum == 0){
                        ioBurst.io_time = 0;
                        ioBurst.priority = 0;
                        //Burst done
                        addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::IO_BURST_COMPLETED);
                        if(ioThread.rrBurstBlockedQueue.empty() && ioThread.rrBurstQueue.empty()){
                            //Thread done
                            ioThread.endTime = clock_ + quantum;
                            ioThread.setTurnaroundTime();
                            addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::THREAD_COMPLETED);
                            //Process done
                            currentProcess.timeTotal = clock_ - currentProcess.arrival_time;
                            allProcess.push_back(currentProcess);
                        }
                        //Thread still has bursts to finish
                        else{
                            readyThreads.push(ioThread);
                        }
                    }
                    
                    if(ioBurst.io_time - quantum > 0){
                        ioBurst.io_time = ioBurst.io_time - quantum;
                        ioBurst.priority = 1;
                        ioThread.rrBurstBlockedQueue.push(ioBurst);
                        addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::THREAD_PREEMPTED);
                        blockedThreads.push(ioThread);
                    }
                    

                    if(ioBurst.io_time - quantum < 0){
                        ioDone = true;
                        ioDoneTime = clock_ - (ioBurst.cpu_time - quantum);
                        ioBurst.io_time = 0;
                        //Burst done
                        addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::IO_BURST_COMPLETED);
                        if(ioThread.rrBurstBlockedQueue.empty() && ioThread.rrBurstQueue.empty()){
                            //Thread done
                            ioThread.endTime = clock_ - (ioBurst.io_time - quantum);
                            ioThread.setTurnaroundTime();
                            addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::THREAD_COMPLETED);
                            //Process done
                            currentProcess.timeTotal = clock_ - currentProcess.arrival_time;
                            allProcess.push_back(currentProcess);
                        }
                        //Thread still has bursts to finish
                        else{
                            readyThreads.push(ioThread);
                        }
                    }
                }
                
                //Finished quantum early, so advance
                if(cpuDone && ioDone){
                    if(cpuDoneTime >= ioDoneTime){
                        clock_ = cpuDoneTime;
                    }
                    else{
                        cpu_idle_time += ioDoneTime - cpuDoneTime;
                        clock_ = ioDoneTime;
                    }
                    continue;
                }
                else{
                    clock_ += quantum;
                }
                
            }
        }
       

        void Priority(){
            bool priority0 = false;
            bool priority1 = false;
            bool priority2 = false;
            bool priority3 = false;
            while(!readyProcess.empty()){

                Process currentProcess = readyProcess.top();
                readyProcess.pop();
                if(currentProcess.arrival_time > clock_){
                    cpu_idle_time += currentProcess.arrival_time - clock_;
                    clock_ = currentProcess.arrival_time;
                    addEvent(currentProcess.type, currentProcess.processID, 0, Event::event_type::DISPATCHER_INVOKED);
                    clock_ += currentProcess.process_switch_overhead;
                    dispatchTotal += currentProcess.process_switch_overhead;
                    addEvent(currentProcess.type, currentProcess.processID, 0, Event::event_type::PROCESS_DISPATCH_COMPLETED);
                }
                else{
                    addEvent(currentProcess.type, currentProcess.processID, 0, Event::event_type::DISPATCHER_INVOKED);
                    clock_ += currentProcess.process_switch_overhead;
                    dispatchTotal += currentProcess.process_switch_overhead;
                    addEvent(currentProcess.type, currentProcess.processID, 0, Event::event_type::PROCESS_DISPATCH_COMPLETED);
                }
                currentProcess.threads.pop();
                Thread currentThread = currentProcess.threads.front();
                if( currentProcess.type == 0){
                    while(!currentProcess.threads.empty()){
                        currentProcess.threads.pop();
                        readyPriority0.push(currentThread);
                    }
                }
                if( currentProcess.type == 1){
                    while(!currentProcess.threads.empty()){
                        currentProcess.threads.pop();
                        readyPriority1.push(currentThread);
                    }
                }
                if( currentProcess.type == 2){
                    while(!currentProcess.threads.empty()){
                        currentProcess.threads.pop();
                        readyPriority0.push(currentThread);
                    }
                }
                if( currentProcess.type == 3){
                    while(!currentProcess.threads.empty()){
                        currentProcess.threads.pop();
                        readyPriority3.push(currentThread);
                    }
                }

                do{
                    if(readyPriority0.empty()){
                        if(readyPriority1.empty()){
                            if(readyPriority2.empty()){
                                priority0 = false;
                                priority1 = false;
                                priority2 = false;
                                priority3 = true;
                                //use readyPriority3
                            }
                            else{
                                priority0 = false;
                                priority1 = false;
                                priority2 = true;
                                priority3 = false;
                                //use readyPriority2
                            }
                        }
                        else{
                            priority0 = false;
                            priority1 = true;
                            priority2 = false;
                            priority3 = false;
                            //use readyPriority1
                        }
                    }
                    else{
                        priority0 = true;
                        priority1 = false;
                        priority2 = false;
                        priority3 = false;
                        //use readypriority0
                    }
    
                    if(priority0){
                        Thread currentThread= move(readyPriority0.top());
                        readyPriority0.pop();
                    }
                    else if(priority1){
                        Thread currentThread = move(readyPriority1.top());
                        readyPriority1.pop();
                    }
                    else if(priority2){
                        Thread currentThread = move(readyPriority2.top());
                        readyPriority2.pop();
                    }
                    else if(priority3){
                        Thread currentThread = move(readyPriority3.top());
                        readyPriority3.pop();
                    }
                    
                    
                    addEvent(currentProcess.type, currentProcess.processID, currentThread.threadID, Event::event_type::DISPATCHER_INVOKED);
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
                        
                    // Basically same as FCFS for thread handling
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
                    if(currentProcess.threads.empty()){
                        allProcess.push_back(currentProcess);
                    }
                } while(!(readyPriority0.empty()) && !(readyPriority1.empty()) && !(readyPriority2.empty()) && !(readyPriority3.empty()));
            }
        }
        
        
        //if queue length < 5 increase quantum, hold at 7, if length > 5, increase decrease quantum and increase
        // use STN
        // sort by priority: pop all of same current front priority it to STN for both blocked and ready
        // Detailed README description
        void Custom(){
            //Starts with a quantum of time 3;
            int quantum = 3;
            bool cpuDone = false;
            bool ioDone = false;
            int cpuDoneTime;
            int ioDoneTime;
            //Load next process
            //while all the queues aren't empty
            while(!priorityThreadQueue.empty() && !priorityThreadBlockedQueue.empty() && !STNqueue.empty() && !STNBlockedqueue.empty() && !readyProcess.empty()){
            
                cpuDone = false;
                ioDone = false;
                //Grab Next Process
                Process currentProcess = move(readyProcess.top());
                //No process or threads ready, so move clock to arrival of next event and add cpu idle time
                if(currentProcess.arrival_time > clock_ && !priorityThreadQueue.empty() && !priorityThreadBlockedQueue.empty() && !STNqueue.empty() && !STNBlockedqueue.empty() && !readyProcess.empty()){
                    cpu_idle_time += currentProcess.arrival_time - clock_;
                    clock_ = currentProcess.arrival_time;
                    continue;
                }
                
                //ELSE we have things to process
                readyProcess.pop();
                //addEvent(currentProcess.type, currentProcess.processID, 0, Event::event_type::DISPATCHER_INVOKED);
                clock_ += currentProcess.process_switch_overhead;
                dispatchTotal += currentProcess.process_switch_overhead;
                addEvent(currentProcess.type, currentProcess.processID, 0, Event::event_type::PROCESS_DISPATCH_COMPLETED);

                
                //Load all processes threads that have arrived into final waiting queue
                while(!currentProcess.threads.empty()){
                    Thread currentThread = move(currentProcess.threads.front());
                    currentProcess.threads.pop();
                    currentThread.startTime = clock_;
                    currentThread.setWaitTime(clock_);
                    priorityThreadQueue.push(currentThread);
                }   
                
                
                //Add all threads of top priority to STN queue
                while(!priorityThreadQueue.empty()){
                    Thread thread = move(priorityThreadQueue.top());
                    STNqueue.push(thread);
                    if(priorityThreadQueue.top().type != thread.type){
                        break;
                    }
                }
                
                // Adjust quantum to size of STNqueue
                if(STNqueue.size() < 5 && quantum < 7){
                    quantum++;
                }
                if(STNqueue.size() > 5 && quantum > 2){
                    quantum--;
                }
                
                //Get next shortest thread to process
                Thread cpuThread = STNqueue.top();
                addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::DISPATCHER_INVOKED);
                if(cpuThread.threadID != lastThreadType){
                    addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::THREAD_PREEMPTED);
                    clock_ += cpuThread.thread_switch_overhead;
                    dispatchTotal += cpuThread.thread_switch_overhead;
                    addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::THREAD_DISPATCH_COMPLETED);
                }
                if(cpuThread.type != lastProcessType){
                    clock_ += currentProcess.process_switch_overhead;
                    dispatchTotal += currentProcess.process_switch_overhead;
                }
                
                lastThreadType = cpuThread.threadID;
                lastProcessType = cpuThread.type;
                STNqueue.pop();
                
                //Handle current CPU Burst
                Burst cpuBurst = move(cpuThread.rrBurstQueue.top());
                cpuThread.rrBurstQueue.pop();
                if(cpuBurst.cpu_time - quantum == 0){
                    cpuThread.timeLeft -= quantum;
                    cpuBurst.setCPUTime(0);
                    cpuBurst.priority = 0;
                    cpuThread.rrBurstBlockedQueue.push(cpuBurst);
                    addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::CPU_BURST_COMPLETED);
                    priorityThreadBlockedQueue.push(cpuThread);
                }
                if(cpuBurst.cpu_time - quantum > 0){
                    cpuThread.timeLeft -= (cpuBurst.cpu_time - quantum);
                    cpuBurst.cpu_time = cpuBurst.cpu_time - quantum;
                    cpuBurst.priority = 1;
                    cpuThread.rrBurstQueue.push(cpuBurst);
                    priorityThreadQueue.push(cpuThread);
                }
                
                if(cpuBurst.cpu_time - quantum < 0){
                    cpuThread.timeLeft -= (cpuBurst.cpu_time - quantum);
                    cpuDone = true;
                    cpuDoneTime = clock_ + (cpuBurst.cpu_time - quantum);
                    cpuBurst.cpu_time = 0;
                    cpuBurst.priority = 0;
                    addEvent(currentProcess.type, currentProcess.processID, cpuThread.threadID, Event::event_type::CPU_BURST_COMPLETED);
                    cpuThread.rrBurstBlockedQueue.push(cpuBurst);
                    priorityThreadBlockedQueue.push(cpuThread);
                }
                
                //Add all threads of top priority to STN blocked queue
                while(!priorityThreadBlockedQueue.empty()){
                    Thread thread = move(priorityThreadQueue.top());
                    STNBlockedqueue.push(thread);
                    if(priorityThreadBlockedQueue.top().type != thread.type){
                        break;
                    }
                }
                
                //Adjust quantum if only IO left
                if(STNqueue.size() == 0){
                    // Adjust quantum to size of STNqueue
                    if(STNBlockedqueue.size() < 5 && quantum < 7){
                        quantum++;
                    }
                    if(STNBlockedqueue.size() > 5 && quantum > 2){
                        quantum--;
                    }
                }
                
                
                
                //Handle blocked queue for quantum
                if(!STNBlockedqueue.empty()){
                    Thread ioThread = move(STNBlockedqueue.top());
                    addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::DISPATCHER_INVOKED);
                    STNBlockedqueue.pop();
                    
                    if(ioThread.threadID != lastThreadType){
                        addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::THREAD_PREEMPTED);
                        clock_ += ioThread.thread_switch_overhead;
                        dispatchTotal += ioThread.thread_switch_overhead;
                        addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::THREAD_DISPATCH_COMPLETED);

                    }
                    if(ioThread.type != lastProcessType){
                        clock_ += currentProcess.process_switch_overhead;
                        dispatchTotal += currentProcess.process_switch_overhead;
                    }
                    lastBlockedThreadType = ioThread.threadID;
                    lastBlockedProcessType = ioThread.type;

                    Burst ioBurst = move(ioThread.rrBurstBlockedQueue.top());
                    ioThread.rrBurstBlockedQueue.pop();
                    
                    if(ioBurst.io_time - quantum == 0){
                        ioThread.timeLeft -= quantum;
                        ioBurst.io_time = 0;
                        ioBurst.priority = 0;
                        //Burst done
                        addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::IO_BURST_COMPLETED);
                        if(ioThread.rrBurstBlockedQueue.empty() && ioThread.rrBurstQueue.empty()){
                            //Thread done
                            ioThread.endTime = clock_ + quantum;
                            addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::THREAD_COMPLETED);
                            ioThread.endTime = clock_;
                            ioThread.setTurnaroundTime();
                            //Process done
                            currentProcess.timeTotal = clock_ - currentProcess.arrival_time;
                            allProcess.push_back(currentProcess);
                        }
                        //Thread still has bursts to finish
                        else{
                            priorityThreadQueue.push(ioThread);
                        }
                    }
                    
                    if(ioBurst.io_time - quantum > 0){
                        ioThread.timeLeft -= (ioBurst.io_time - quantum);
                        ioBurst.io_time = ioBurst.io_time - quantum;
                        ioBurst.priority = 1;
                        ioThread.rrBurstBlockedQueue.push(ioBurst);
                        priorityThreadBlockedQueue.push(ioThread);
                    }
                    

                    if(ioBurst.io_time - quantum < 0){
                        ioThread.timeLeft -= (ioBurst.io_time - quantum);
                        ioDone = true;
                        ioDoneTime = clock_ - (ioBurst.cpu_time - quantum);
                        ioBurst.io_time = 0;
                        //Burst done
                        addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::IO_BURST_COMPLETED);
                        if(ioThread.rrBurstBlockedQueue.empty() && ioThread.rrBurstQueue.empty()){
                            //Thread done
                            ioThread.endTime = clock_ - (ioBurst.io_time - quantum);
                            ioThread.setTurnaroundTime();
                            addEvent(currentProcess.type, currentProcess.processID, ioThread.threadID, Event::event_type::THREAD_COMPLETED);
                                //Process done
                            currentProcess.timeTotal = clock_ - currentProcess.arrival_time;
                            allProcess.push_back(currentProcess);
                        }
                        //Thread still has bursts to finish
                        else{
                            priorityThreadBlockedQueue.push(ioThread);
                        }
                    }
                }
                
                if(cpuDone && ioDone){
                    if(cpuDoneTime >= ioDoneTime){
                        clock_ = cpuDoneTime;
                    }
                    else{
                        cpu_idle_time += ioDoneTime - cpuDoneTime;
                        clock_ = ioDoneTime;
                    }
                    continue;
                }
                else{
                    clock_ += quantum;
                }
            }
        }
        
        void printStats(){
            cout << "SIMULATION COMPLETED!" << endl;
            double threadCount = 0;
            double responseTime = 0;
            double turnaroundTime = 0;
            for(Process process : allProcess){
                if(getType(process.processID) == "SYSTEM"){
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
                
            }
            cout << endl << "SYSTEM THREADS:" << endl << "\tTotal count:\t" << threadCount<< endl << "\tAvg response time:\t" << responseTime << endl;
            cout << "\tAvg turnaround time:\t" << turnaroundTime << endl << endl;
            
            threadCount = 0;
            responseTime = 0;
            turnaroundTime = 0;
            for(Process process : allProcess){
                if(getType(process.processID) == "INTERACTIVE"){
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
                
            }
            cout << endl << "INTERACTIVE THREADS:" << endl << "\tTotal count:\t" << threadCount<< endl << "\tAvg response time:\t" << responseTime << endl;
            cout << "\tAvg turnaround time:\t" << turnaroundTime << endl << endl;
            
            threadCount = 0;
            responseTime = 0;
            turnaroundTime = 0;
            
            //Look at add thread to a vector when they're complete and use that instead.
            for(Process process : allProcess){
                if(getType(process.processID) == "NORMAL"){
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
            }
            cout << endl << "NORMAL THREADS:" << endl << "\tTotal count:\t" << threadCount<< endl << "\tAvg response time:\t" << responseTime << endl;
            cout << "\tAvg turnaround time:\t" << turnaroundTime << endl << endl;
            
            threadCount = 0;
            responseTime = 0;
            turnaroundTime = 0;
            for(Process process : batchProcess){
                if(getType(process.processID) == "BATCH"){
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
};
