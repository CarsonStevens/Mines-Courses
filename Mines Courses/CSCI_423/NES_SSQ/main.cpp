//SSQ SIM

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
    
    default_random_engine generator (time(0));
    exponential_distribution<double> Xdistribution (1.0);
    uniform_real_distribution<double> Udistribution(1.0);
    Event event;
    event.start_time = Xdistribution(generator)*2;
    event.service_time = Udistribution(generator)*2;
    Type type_ = Type::JOB_ARRIVAL;
    event.type = type_;
    event_queue.push(event);
    event_list.push(event);
    
    while(!event_queue.empty()){
    //ONLY ADD to event queue when event_clock < 1000;   
        //Get next job in queue
        Event current_event = event_queue.top();
        event_queue.pop();
        
        //Update clock to next event
        event_clock = event.start_time;
        
        if(event.type == Type::JOB_ARRIVAL || event.type == Type::FEEDBACK_ARRIVAL){
            if(event.type == Type::JOB_ARRIVAL){
                newJobCount++;
                double nextArrival = event_clock + Udistribution(generator) * 2;
                Event nextEvent;
                if(nextArrival < EVENTS){
                    newJobCount++;
                    Type type_ = Type::JOB_ARRIVAL;e
                    nextEvent.type = type_;
                    nextEvent.service_time = Udistribution(generator) * 2;
                    event_queue.push(nextEvent);
                }
                event_list.push(current_event);
            }
            else if(event.type == Type::JOB_COMPLETION){
                if(Udistribution(generator) <= 0.25){
                    Event nextEvent;
                    nextEvent.start_time = Xdistribution(generator);
                    Type type_ = Type::FEEDBACK_ARRIVAL;
                    nextEvent.type = type_;
                    nextEvent.service_time = event.service_time/2;
                    event_list.push(nextEvent);
                }
                Event done;
                Type type_ = Type::JOB_COMPLETION;
                done.type = type_;
                done.start_time = event_clock;
                inService = false;
            }
            if(!inService && event_queue.size() > 0){
                Event nextEvent = event_queue.top();
                event_queue.pop();
                
            }
        }
        
    }
    cout << "t:\t" << event_clock << endl << "Jobs Serviced:\t" << jobsServiced << endl << "Number of new Jobs:\t" << newJobCount<< endl;
    return 0;
}
