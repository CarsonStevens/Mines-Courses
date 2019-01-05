#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <queue>
#include <sstream>
#include <fstream>

#include "burst.h"
#include "thread.h"
#include "process.h"
#include "event.h"

#include "CPU.h"

using namespace std;

int main( int argc, char* argv[] ){
    
    // argv[1] = "-v";
    // argv[2] = "input.txt";
    
    //Object that contains operations for scheduling
    CPU cpu;
    
    //Variables for reading input
    string file = "";
    string line = "";
    int num_processes;
    int thread_switch_overhead;
    int process_switch_overhead;
    int processID;
    int num_threads;
    int num_bursts;
    int cpu_time;
    int io_time;
    int type;
    int timeLeft = 0;
    int serviceTime;
    int ioTime;
    
    //Scheduling Variables
    vector <Process> processes;

    
//Load file
    
    // Get File Name
    // cout << "Please enter the file for input:\t";
    // cin >> file;
    file = argv[argc-1];
    //open the file from which to read the data
    ifstream data;
    data.open(argv[argc-1]);
    
    // if the file is empty or won't load, it outputs the message
    if (!data){
        cout << "Error Loading Input File: " << file << "." << endl;  
        return 1;
    } 

    // Read in first line specifying situtation
    getline(data,line);
    istringstream process(line);
    process >> num_processes >> thread_switch_overhead >> process_switch_overhead;
    getline(data,line);
    
    
    //Parsing and Instantiation for Input File
    while(!data.eof()){
        getline(data,line);
        istringstream process(line);
        process >> processID >> type >> num_threads;
        Process currentProcess = Process(processID, type, process_switch_overhead);
        Thread thread;
        int threadNumber = 0;
        for(int j = 0; j < num_threads; j++){
            getline(data,line);
            istringstream info(line);
            info >> thread.arrival_time >> num_bursts;
            thread.threadID = threadNumber;
            thread.thread_switch_overhead = thread_switch_overhead;
            thread.processID = processID;
            threadNumber++;
            for(int i = 0; i < num_bursts-1; i++){
                getline(data,line);
                istringstream burst_info(line);
                burst_info >> cpu_time >> io_time;
                // Add data to burst
                Burst burst(cpu_time, io_time);
                serviceTime += cpu_time;
                ioTime += io_time;
                timeLeft += cpu_time + io_time;
                thread.addBurst(burst);
            }
            getline(data,line);
            istringstream burst_info(line);
            burst_info >> cpu_time;
            Burst burst(cpu_time, 0);
            serviceTime += cpu_time;
            timeLeft += cpu_time;
            thread.timeLeft = timeLeft;
            thread.serviceTime = serviceTime;
            thread.IOTime = ioTime;
            serviceTime = 0;
            ioTime = 0;
            timeLeft = 0;
            getline(data,line);
            thread.addBurst(burst);
            thread.type = type;
            currentProcess.addThread(thread);

        }
        processes.push_back(currentProcess);
        cpu.processToCPU(currentProcess);

        

    }
    data.close();
    

    // cpu.RoundRobin();
    //cpu.Custom();

    
    
    //In all cases, regardless of the flags a user passes, your simulation must output the following information:
    bool chosen = false;
    for(int i = 0; i < argc; i++){
        string arguement = argv[i];
        
         //Handles help flag help text
        if (arguement == "-vh" || arguement == "-h" || arguement == "-th" || arguement == "-ht" || arguement == "-hv" || arguement == "--help"){
            cout << endl << "Welcome to the CPU Scheduling's help page" << endl << endl;
            cout << "To run the program, type:\t ./simulator [flags] input_file_name\t into the terminal." << endl << endl;
            cout << "-v, --verbose" << endl << "\tOutput information about every state-changing event and scheduling decision." << endl << endl;
            cout << "-t, --per_thread" << endl << "\tOutput additional per-thread statistics for arrival time, service time, etc" << endl << endl;
            cout << "-h --help" << endl << "\tDisplay a help message about these flags and exit" << endl<< endl;
            return(0);
        }
        
        //Handles algorithm choice
        
        if(arguement == "-a" || "--algorithm"){
            string algorithm = argv[i+2];
            chosen = true;
            if(algorithm == "FCFS"){
                cpu.FirstComeFirstServe();
            }
            if(algorithm == "RR"){
                cpu.RoundRobin();
            }
            if(algorithm == "PRIORITY"){
                cpu.Priority();
            }
            if(algorithm == "CUSTOM"){
                cpu.Custom();
            }
        }
    }
    
    //Algorithm not specified, so choose FCFS
    if(chosen == false){
        cpu.FirstComeFirstServe();
    }
    
    
    //Handles stat printing
    for(int i = 0; i < argc; i++){
        string arguement = argv[i];
        
        if(arguement == "-v" || arguement == "-vt" || arguement == "--verbose" || arguement == "-tv"){
            cpu.printEventQueue();
        }
        
        if(arguement == "-t" || arguement == "--per_thread"  || arguement == "-tv" || arguement == "-vt"){
            cpu.perThread();
        }
    }
    
    cpu.printStats();
}