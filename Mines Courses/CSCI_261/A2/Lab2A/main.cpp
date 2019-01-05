/*Lab2A, Pair Programming
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 *Program 2 equations from the list with constants and everything defined.
 */
 
 
#include<iostream>
#include <ios>
#include <cmath>
#include <iomanip>
using namespace std;

int main() {
    
    //The global variables required for the distance formula
    double distance = 0;
    double length = 0;
    double width = 0;
    
    // Global variables required for the volume of a sphere
    double volume = 0;
    double radius = 0;
    const double PI = 3.14159;
    
    // Prompt user for input on radius, width and length 
    cout << "Please enter the length, width, and radius: " << '\n' << "Length: ";
    cin >> length;
    cout << "Width: ";
    cin >> width;
    cout << "Radius: ";
    cin >> radius;
    
    //Formula for the distance
    
    distance = sqrt ((length * length) + (width * width));
    
    // Formula for volume
    
    volume = (4.0 / 3.0) * (PI) * (radius * radius);
    
    cout << '\n';
    cout << '\n';
    
    //To print the distance and the volume
    
    cout << "The distance between " << length << " and " << width << " is: " << distance << endl;
    
    cout << "The volume of the sphere with the radius of " << radius << " is: " << volume << endl;
    
    return 0;
}