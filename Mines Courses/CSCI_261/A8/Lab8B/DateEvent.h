#include <string>
#include <iostream>
#include <vector>
using namespace std;
#pragma once

class Event{
    public:
        Event();
        void setEvent(string date);
        string getEvent(int m, int d, int y);
        string _date;
    private:
        string _event;
};


class Date{
    
    public:
        Date();
        void introduction();
        string getDate();
        void setDate(int day, int month, int year, string title, string location);
        string _date;
    private:
        int _day;
        int _month;
        int _year;
        string _title;
        string _location;
    
};