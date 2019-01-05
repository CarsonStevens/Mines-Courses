#include <cmath>
#include <iostream>

using namespace std;

#pragma once

// Class to represent points.
class Point {
        private:
                double x, y;
        public:
                
                // Constructor
                Point(double x = 0.0, double y = 0.0) {
                        this->x = x;
                        this->y = y;
                }
                
                // Copy constructor 
                Point(const Point &point) {x = point.x; y = point.y; } 
        
                // Getters.
                double getX(){
                    return this->x; 
                }
                double getY(){
                    return this->y; 
                }
        
                // Distance to another point.  Pythagorean thm.
                double getDistance(Point other){
                    double xd = this->x - other.x;
                    double yd = this->y - other.y;
                    return sqrt(xd*xd + yd*yd);
                }
        
                // Prints point
                void print(){
                        cout << "(" << this->x << "," << this->y << ")";
                }
                
                void operator = (const Point& other){
                        x = other.x;
                        y = other.y;
                }
                
                bool operator == (const Point& other) {
                        if((x == other.x) && (y == other.y)){
                                return true;
                        }
                        return false;
                }
};