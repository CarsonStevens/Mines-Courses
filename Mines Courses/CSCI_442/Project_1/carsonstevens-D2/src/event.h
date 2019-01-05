#include <iostream>
using namespace std;

#pragma once

class Event{
    public:
        struct Compare_Event{
            bool operator()(Event event1, Event event2){
                return event1.event_start_time > event2.event_start_time;
            }
        };
        
        enum event_type { THREAD_ARRIVED,
                        THREAD_DISPATCH_COMPLETED,
                        PROCESS_DISPATCH_COMPLETED,
                        CPU_BURST_COMPLETED,
                        IO_BURST_COMPLETED,
                        THREAD_COMPLETED,
                        THREAD_PREEMPTED,
                        DISPATCHER_INVOKED };
                        
        int event_start_time;
        int event_ID;
        int eventNum;
        int threadID;
        int processID;
        Process::Type processType;
        event_type type;
        Event(int processType, int processID, int threadID, int start, int ID, int eventNum){
            event_start_time = start;
            event_ID = ID;
            this->eventNum = eventNum;
            this->processID = processID;
            this->threadID = threadID;
            switch(processType){
                case 0 : this->processType = Process::Type::SYSTEM;        break;
                case 1 : this->processType = Process::Type::INTERACTIVE;   break;
                case 2 : this->processType = Process::Type::NORMAL;        break;
                case 3 : this->processType = Process::Type::BATCH;         break;
                default :                                           break;
            }
        }
        
        
        void getEvent(){
            string transition = "";
            cout << "At time " << event_start_time << ":" << endl;
            switch(eventNum){
                case THREAD_ARRIVED   : cout << "\tTHREAD_ARRIVED" << endl; transition = "NEW to READY";   break; 
                case THREAD_DISPATCH_COMPLETED: cout << "\tTHREAD_DISPATCH_COMPLETED" << endl; transition = "READY to RUNNING"; break; 
                case PROCESS_DISPATCH_COMPLETED : cout << "\tPROCESS_DISPATCH_COMPLETED" << endl; transition = "READY to RUNNING"; break; 
                case CPU_BURST_COMPLETED  : cout << "\tCPU_BURST_COMPLETED" << endl; transition = "RUNNING to BLOCKED";  break; 
                case IO_BURST_COMPLETED : cout << "\tIO_BURST_COMPLETED" << endl; transition = "BLOCKED to READY"; break;
                case THREAD_COMPLETED : cout << "\tTHREAD_COMPLETED" << endl; transition = "RUNNING to EXIT"; break;
                case THREAD_PREEMPTED : cout << "\tTHREAD_PREEMPTED" << endl; transition = "RUNNING to BLOCKED"; break;
                case DISPATCHER_INVOKED : cout << "\tDISPATCHER_INVOKED" << endl; transition = "Selected from " + to_string(threadID) + " threads; will run to completion of burst"; break;
                default    : cout << "\tNONE" << endl; break;
            }
            cout << "\tThread " << threadID << " in process " << processID << " [" << getType(processType) << "]" << endl << "\tTransitioned from " << transition << endl << endl;
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