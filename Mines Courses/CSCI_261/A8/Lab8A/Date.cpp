
#include "Date.h"

Date::Date(){
    //sets the default date.
    _day = 30;
    _month = 12;
    _year = 1950;
    _date = to_string(_month) + "/" + to_string(_day) + "/" + to_string(_year);
    //Extra Credit default shout out
    cout << "Bjarne Stroustrup's Birth";
    
}


void Date::introduction(){
    //Gives the introduction to the user when called.
    cout << "Please enter the date you would like to store:" << endl;
    return;
}
string Date::getDate(){
    //Extra Credit Shout out.
    if(_date == "8/1/1876"){
        cout << "Colorado became a state!..." << endl;
    }
    return _date;
}

//Sets the day, month, and year if they are acceptable values.
void Date::setDate(int day, int month, int year){
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
    _date = to_string(_month) + "/" + to_string(_day) + "/" + to_string(_year);
    return;
}
