#include <iostream>
#include <queue>
#include <ctime>
#include <random>

using namespace std;

struct Event{
    enum Type{THINKING=1, TYPING=2, OUTPUT=3} type;
    double event_time;
    int info = 0;
};

struct Compare_Event{
    int operator() (const Event& event1, const Event& event2) { 
        if(event1.event_time == event2.event_time){
            return event1.type < event2.type;
        }
        return !(event1.event_time < event2.event_time);
    }
};





Event getThinking(double &event_clock, double thinking){
    //Handle Thinking
    
    Event thinkingEvent;
    thinkingEvent.type = Event::Type::THINKING;
    thinkingEvent.event_time = event_clock + thinking;
    event_clock += event_clock +thinking;
    return thinkingEvent;
}

Event getStroke(double &event_clock, double stroke_time){
    //Handles Typing
    Event typingEvent;
    typingEvent.event_time = event_clock + stroke_time;
    event_clock += event_clock + stroke_time;
    typingEvent.type = Event::Type::TYPING;
    // typingEvent.info = strokeNumber - i;
    return typingEvent;
}
    
Event getOutput(double &event_clock){   
    //Handles Output
    Event outputEvent;
    double output_time = event_clock + double(1/120);
    outputEvent.event_time = event_clock + output_time;
    event_clock += event_clock + double(1/120);
    outputEvent.type = Event::Type::OUTPUT;
    cout << "output" << endl;
    // outputEvent.info = outputNumber - i;
    return outputEvent;
}
    

int main(){
    
    default_random_engine generator;
    uniform_real_distribution<double> thinkTime(0.0,10.0);
    uniform_int_distribution<int> numberOfStrokes(5,15);
    uniform_real_distribution<double> strokeTime(0.15,0.35);
    uniform_int_distribution<int> numberOfOutput(50,300);
    
    
    double event_clock = 0;
    double const END_TIME = 1000;
    int userNumber = 1;
    int n = 0;
    queue<Event> event_calender;
    priority_queue<Event, vector<Event>, Compare_Event> event_queue;
    bool run = true;
    
    //Prime SIM
    for(int i = 0; i < userNumber; i++){
        double thinking = thinkTime(generator);
        Event event = getThinking(event_clock, thinking);
        event_queue.push(event);
        event_calender.push(event);
        n++;
    }

    while(event_clock < END_TIME && run == true){

        Event nextEvent = event_queue.top();
        event_queue.pop();

        
        //Handle strokes
        if(nextEvent.type == Event::Type::THINKING){
            double stroke_time = strokeTime(generator);
            Event eventTyping = getStroke(event_clock, stroke_time);
            eventTyping.info = numberOfStrokes(generator) -1;
            event_queue.push(eventTyping);
            event_calender.push(eventTyping);
            continue;
        }
        
        if(nextEvent.type == Event::Type::TYPING){
            //Handle Strokes
            if(nextEvent.info >= 0){
                double stroke_time = strokeTime(generator);
                Event eventTyping = getStroke(event_clock, stroke_time);
                eventTyping.info--;
                event_queue.push(eventTyping);
                event_calender.push(eventTyping);
            }
            //Handle Output Creation
            else{
                Event outputEvent = getOutput(event_clock);
                outputEvent.info = numberOfOutput(generator) -1;
                event_queue.push(outputEvent);
                event_calender.push(outputEvent);
            }
            continue;
        }
        
        if(nextEvent.type == Event::Type::OUTPUT){
            //Handles Outputting
            if(nextEvent.info > 0){
                Event outputEvent = getOutput(event_clock);
                outputEvent.info--;
                event_queue.push(outputEvent);
                event_calender.push(outputEvent);
            }
            //Handles thinking
            else{
                double thinking = thinkTime(generator);
                Event thinkingEvent = getThinking(event_clock, thinking);
                event_queue.push(thinkingEvent);
                event_calender.push(thinkingEvent);
                n++;
            }
            continue;
        }
        
    }
    
    int thinkNumber = 0;
    int typeNumber = 0;
    int outputNumber = 0;
    int totalEvents = 0;
    
    while(!event_calender.empty()){
        Event event = event_calender.front();
        event_calender.pop();
        
        if(event.type == Event::Type::THINKING){
            thinkNumber++;
        }
        if(event.type == Event::Type::TYPING){
            typeNumber++;
        }
        if(event.type == Event::Type::OUTPUT){
            outputNumber++;
        }
        totalEvents++;
        
    }
    
    cout << "Thinking Events:\t" << thinkNumber << endl;
    cout << "Typing Events:\t" << typeNumber << endl;
    cout << "Outputting Events:\t" << outputNumber << endl;
    cout << event_clock;
    
}