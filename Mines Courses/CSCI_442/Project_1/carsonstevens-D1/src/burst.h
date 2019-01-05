#include <iostream>

using namespace std;

#pragma once


class Burst{
    public:
        int cpu_time;
        int io_time;
        
        Burst(int cpu_time, int io_time){
            this->cpu_time = cpu_time;
            this->io_time = io_time;
        }
        Burst(){
            
        }
        
        int getCPUTime(){
            return cpu_time;
        }
        
        int getIOTime(){
            return io_time;
        }
        
        void setIOTime(int t){
            io_time = t;
        }
        
        void setCPUTime(int t){
            cpu_time = t;
        }
};