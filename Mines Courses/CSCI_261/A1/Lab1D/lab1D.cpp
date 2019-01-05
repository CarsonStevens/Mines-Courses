/* CSCI 261 Lab 1C: Prettying it up
 *
 * Author: Carson Stevens
 *
 * Changing the code from lab 1C to make it prettier.
 */

#include <iostream>
#include <ios>
#include <iomanip>
using namespace std;

int main() {
   
    //Variables for any structure below

    float radius = 0;
    float circleArea = 0;
    const float PI = 3.1415;
    
    //Statements to enter the dimensions
    
    cout << "Enter the length, width, and height to find the volume of any structure." << endl;
    cin >> anyLength >> anyWidth >> anyHeight;
    cout << "To find the area of a circle, please enter a radius." << endl;
    cin >> radius;
    
    //Equations from the volume and area. 
    
    anyVolume = anyLength * anyWidth * anyHeight;
    circleArea = PI * radius * radius;
    
    //Output statements that give the answers
    
    cout << setprecision(5) << showpoint;//Setting the precision of the answers
    
    //Print statements with setfill to pretty it up.
    cout << endl << endl << "Box Volume:  " << setw(20) << setfill(' ') << left << anyVolume << endl;
    cout <<  "Circle Area: " << setw(20) << setfill(' ') << left << circleArea << endl;
    
    return 0;
}