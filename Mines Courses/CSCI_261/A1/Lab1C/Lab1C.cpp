/* CSCI 261 Lab 1C: GEOMETRIC CALCULATIONS
 *
 * Author: Carson Stevens
 *
 * Edit the code to calculate the volume of a specific square prism.
 */

#include <iostream>
using namespace std;

int main() {

    //Variables are listed below
    int length = 17;
    int width = 17;  
    int height = 2; 
    int volume = 0;
    int area = 0;
    
    //Area of the base is length times width
    
    area = length * width;

    // Volume of a box is length times width times height. 
    volume = length * width * height;
    
    cout << "The volume of a square prism with a length " << length << ", a width " << width << ", and a height " << height << " is " << volume << "." << endl << endl;
    // Part 2, Code that calculates any box like structure
    
    //Variables for any structure below
    float anyLength = 0;
    float anyWidth = 0; 
    float anyHeight = 0;
    float anyVolume = 0;
    
    cout << "Enter the length, width, and height to find the volume of any structure." << endl << "\tLength: ";
    cin >> anyLength;
    cout << "\tWidth: ";
    cin >> anyWidth;
    cout << "\tHeight: ";
    cin >> anyHeight;
    
     anyVolume = anyLength * anyWidth * anyHeight;
    
    cout << endl << "The volume of the structure is "<< anyVolume << "." << endl << endl << endl;
    
    //Part 3, Calcuate the area of a circle.
    
    //Variables are listed below, including the constant Pi.
    
    int radius = 0;
    float circleArea = 0;
    const float PI = 3.1415;
    
    cout << "To find the area of a circle, please enter a radius." << endl << "\tRadius: ";
    
    cin >> radius;
    
    circleArea = PI * radius * radius;
    
    cout << endl << "The area of a circle with a radius of " << radius << " is " << circleArea << "." << endl;
    

    return 0; // signals the operating system that our program ended OK.
}