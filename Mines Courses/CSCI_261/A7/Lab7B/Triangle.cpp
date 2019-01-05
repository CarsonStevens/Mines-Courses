#include <cmath>
#include <iostream>
using namespace std;
#include "Triangle.h" 


Triangle::Triangle(){
    //Default constructor
    side1 = 1.0;
    side2 = 1.0;
    side3 = 1.0;
    perimeter = side1 + side2 + side3;
    area =  pow(((perimeter/2.0) * ((perimeter/2.0) -side1) * ((perimeter/2.0) - side2) * ((perimeter/2.0) - side3)), 0.5);
}

double Triangle::getSideOne() {
    return _sideOne;
}

double Triangle::getSideTwo() {
    return _sideTwo;
}

double Triangle::getSideThree() {
    return _sideThree;
}

void Triangle::setSideOne(double side1) {
    if(side1 > 0) {
        _sideOne = side1;
    }
    return;
}

void Triangle::setSideTwo(double side2) {
    if(side2 > 0) {
        _sideTwo = side2;
    }
    return;
}

void Triangle::setSideThree(double side3) {
    if(side3 > 0) {
        _sideThree = side3;
    }
    return;
}

bool Triangle::validate(){
    //Determine if it's a triangle
    if (((_sideOne + _sideTwo) <= (_sideThree)) || ((_sideOne + _sideThree) <= (_sideTwo)) || ((_sideThree + _sideTwo) <= (_sideOne))) {
        cout << "That isn't a triangle!" << endl;
        return false;
    }
    else{
        return true;
    }
}

double Triangle::findPerimeter(){
    perimeter = _sideOne + _sideTwo + _sideThree;
    return perimeter;
}


double Triangle::findArea(){
    
    double trianglePerimeter = findPerimeter();
    
    //Area of triangle using the Heron formula
        
    double heronsVariable = (trianglePerimeter)/2.0;
        
    double heronsVariable2 = ((heronsVariable) * ((heronsVariable) - _sideOne) * ((heronsVariable) - _sideTwo) * (heronsVariable - _sideThree));
        
    return pow(heronsVariable2,0.5);
    
}

bool Triangle::isLarger(double userArea){
    if(area < userArea){
        return false;
    }
    else{
        return true;
    }
}

void getTriangleInfo(){
    
    Triangle triangle1;
    Triangle triangle2;
    
    double s1;
    double s2;
    double s3;
    
    cout << "The default edges of the triangle are:" << endl << "\t Side One: " << triangle2.side1;
    cout << endl << "\t Side Two: " << triangle2.side2 << endl << "\t Side Three: " << triangle2.side3 << endl;
    cout << "The default perimeter of the triangle you entered is: \t" << triangle2.perimeter<< endl;
    cout << "The default area of the triangle you entered is: \t" << triangle2.area << endl << endl;

    
    
    
    
    cout << "Please enter the dimensions of your triangle: " << endl << "\t Side One: ";
    cin >> s1;
    triangle1.setSideOne(s1);
    cout << "\t Side Two: ";
    cin >> s2;
    triangle1.setSideTwo(s2);
    cout << "\t Side Three: ";
    cin >> s3;
    triangle1.setSideThree(s3);
    
    
    if (triangle1.validate() == true){
        cout << "The perimeter of the triangle you entered is: \t" << triangle1.findPerimeter() << endl;
        cout << "The area of the triangle you entered is: \t" << triangle1.findArea() << endl;
    }
    
    cout << endl;
    
    if(triangle2.isLarger(triangle1.findArea()) == true){
        cout << "Your area is less than the default triangle." << endl;
    }
    else{
        cout << "Your area is larger than the default triangle." << endl;
    }
    return;
}