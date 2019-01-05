#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <queue>
#include <sstream>
#include <fstream>

#include "job.h"
#include "workstation.h"

using namespace std;

//Checks to see if all the workstation queues are empty
bool complete(Workstation w1, Workstation w2, Workstation w3){
    if(w1.check() && w2.check() && w3.check()){
        return true;
    }
    else{
        return false;
    }
}

bool full(Workstation w1, Workstation w2, Workstation w3){
    if(w1.getState() && w2.getState() && w3.getState()){
        return true;
    }
    else{
        return false;
    }
}

int main(){
    string file = "";   // Name of Input File
    ifstream data;      // ifstream that holds the Input File
    int n;              // Number of Jobs
    int m;              // Number of Workstations
    string line = "";   //Line used to parse input
    Workstation w1;
    Workstation w2;
    Workstation w3;
    int arrival_time;
    int processID = 1;
    int length1, length2, length3;  //Length of jobs sent to Workstations
    int timeTotal = 0;
    
//Load file
    
    // Get File Name
    // cout << "Please enter the file for input:\t";
    // cin >> file;
    file = "small.txt";
    
    //open the file from which to read the data
    data.open(file);
    
    // if the file is empty or won't load, it outputs the message
    if (!data){
        cout << "Error Loading Input File." << endl;  
        return 1;
    } 

    // Read in first line specifying situtation
    getline(data,line);
    istringstream process(line);
    process >> n >> m;
    
//Parsing and Instantiation for Input File
    while(!data.eof()){

        getline(data,line);
        istringstream process(line);
        process >> arrival_time >> length1 >> length2 >> length3;
        Job w1_job(arrival_time, length1, processID);
        Job w2_job(arrival_time, length2, processID);
        Job w3_job(arrival_time, length3, processID);
        w1.addArrivalQueue(w1_job);
        w2.addArrivalQueue(w2_job);
        w3.addArrivalQueue(w3_job);
        
        // Iterate line number (processID)
        processID++;
    }

//Close file
    data.close();
    
    bool add = true;
//Start of   
    while(!(complete(w1, w2, w3))){
        cout << "while loopie" << endl;
        // Run new cycle if all workstation aren't currently running
        if(!(full(w1, w2, w3))){
            
            //Add new processes that arrive at current time
            w1.addLengthQueue(timeTotal);
            w2.addLengthQueue(timeTotal);
            w3.addLengthQueue(timeTotal);
            
            //Try to add to workstation1
            while(add){
                add = w1.addNext(w2, w3, timeTotal);
            }
            add = true;
            
            //Try to add to workstation2
            while(add){
                add = w2.addNext(w1, w3, timeTotal);
            }
            add = true;
            
            //Try to add to workstation3
            while(add){
                add = w3.addNext(w1, w2, timeTotal);
            }
            add = true;
            
            // Update timers and states
            w1.update(timeTotal);
            w2.update(timeTotal);
            w3.update(timeTotal);
        }
        
        //Update time
        timeTotal++;
    }
    
    w1.addIDQueue();
    w2.addIDQueue();
    w3.addIDQueue();
    
    cout << "Final total time:\t" << timeTotal << endl;
    
    int timeTotal_workstations = w1.finalTime();
    if(timeTotal_workstations < w2.finalTime()){
        timeTotal_workstations = w2.finalTime();
    }
    if(timeTotal_workstations < w3.finalTime()){
        timeTotal_workstations = w3.finalTime();
    }

    
    
    cout << "Final workstation time:\t" << timeTotal_workstations << endl;
    
    for(int i = 0; i < w1.pq_ID.size(); i++){
        int start1 = w1.pq_ID.top().start_time;
        int start2 = w2.pq_ID.top().start_time;
        int start3 = w3.pq_ID.top().start_time;
        w1.pq_ID.pop();
        w2.pq_ID.pop();
        w3.pq_ID.pop();
        
        cout << start1 << " " << start2 << " " << start3 << endl;
    }
}

