//SIS

SIM

#include <queue>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <ctime>
#include <vector>

using namespace std;

typedef enum { JOB_ARRIVAL, FEEDBACK_ARRIVAL, JOB_COMPLETION } Type;

struct Event{
    Type type;
    double start_time;
    double service_time;
};

struct Compare_Arrival{
    int operator() (const Event event1, const Event event2) { 
        return event1.start_time < event2.start_time; 
    }
};

int main(int argc, char* argv[]){
    
    priority_queue<Event, vector<Event>, Compare_Arrival> event_queue;    
    double event_clock = 0;
    int const EVENTS = 1000;
    bool inService = false;
    int newJobCount = 0;
    int jobsServiced = 0;
    queue<Event> event_list;