
#include "DateEvent.h"

Date::Date(){
    //Default constructor.
    _day = 30;
    _month = 12;
    _year = 1950;
    //Extra Credit constructor info.
    _location = "Aarhus, Denmark";
    _title = "Bjarne Stroustrup's Birth";
    _date = to_string(_month) + "/" + to_string(_day) + "/" + to_string(_year) + ": " + _title + " (" + _location + ")";
}

Event::Event(){
    //Default event constructor.
    _event = _date;
}

//Sets the event.
void Event::setEvent(string date){
    _event = date;
    return;
}

//Gets the event when called.
string Event::getEvent(int m, int d, int y){
    //Parameterized Extra Credit Shout out.
    if((d == 1) && (m == 8) && (y == 1876)){
        cout << "Colorado became a state!" << endl;
    }
    return _event;
}

//Gives the intro when called.
void Date::introduction(){
    cout << "Please enter the date info you would like to store:" << endl;
    return;
}

//Gets the date when called.
string Date::getDate(){
    return _date;
}

//Sets the Date if the variables are acceptable.
void Date::setDate(int day, int month, int year, string title, string location){
    if((day > 0) && (day <= 31)){
        _day = day;
    }
    else{
        
    }
        if((month > 0) && (month <= 12)){
        _month = month;
    }
    if((year >= 0)){
        _year = year;
    }
    
    _title = title;
    _location = location;
    
    _date = to_string(_month) + "/" + to_string(_day) + "/" + to_string(_year) + ": " + _title + " (" + _location + ")";
    return;
}
