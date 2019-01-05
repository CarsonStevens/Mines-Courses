#include <string>
#include <iostream>
#include <vector>
using namespace std;
#pragma once

class Date{
    
    public:
        Date();
        void introduction();
        string getDate();
        void setDate(int day, int month, int year);
    private:
        int _day;
        int _month;
        int _year;
        string _date;
        
  
  
    
};