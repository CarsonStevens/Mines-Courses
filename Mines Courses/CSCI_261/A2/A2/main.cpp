/*A2 / Triangle Assignment
 *
 *Author: Carson Stevens
 *
 *Ask for triangle side inputs and classify the triangle
 */
 
 
#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
#include <ctime>
#include <string>
#include <cmath>
using namespace std;

int main() {
    
    //Global variables
    double sideOne = 0.0;
    double sideTwo = 0.0;
    double sideThree = 0.0;
    double tempSide = 0.0;//Temporary side for switching numbers
    double trianglePerimeter = 0.0;
    double heronsVariable = 0.0;
    double heronsVariable2 = 0.0;
    double triangleArea = 0.0;
    const double TOLERANCE = 0.0001;
    int maxLength= 0.0;
    int randomMaxLength1 = 0.0;
    int randomMaxLength2 = 0.0;
    int randomMaxLength3 = 0.0;
    double randomMaxLength = 0.0;
    
    cout << "Please enter the dimensions of your triangle: " << endl << "\t Side One: ";
    cin >> sideOne;
    cout << "\t Side Two: ";
    cin >> sideTwo;
    cout << "\t Side Three: ";
    cin >> sideThree;
    
    //Determine if it's a triangle
    if (((sideOne + sideTwo) <= (sideThree)) || ((sideOne + sideThree) <= (sideTwo)) || ((sideThree + sideTwo) <= (sideOne))) {
        cout << "That isn't a triangle!" << endl;
    }
    
    else {
    //Ordering sides in length
        if (sideOne > sideThree) {
            tempSide = sideOne;
            sideOne = sideThree;
            sideThree = tempSide;
        }
        if (sideTwo > sideThree) {
            tempSide = sideTwo;
            sideTwo = sideThree;
            sideThree = tempSide;
        }
    
    
    cout << "The side lengths in increasing order are " << sideOne << " , " << sideTwo << " , " << sideThree << endl;
    
    //Checks to see if its a right triangle
        if ((fabs(((pow(sideOne,2.0) + (pow(sideTwo,2.0) - (pow(sideThree,2.0))))))) <= TOLERANCE) {
            cout << "Triangle is a right triangle!" << endl;
        }
    //Checks to see if its an acute triangle
        if (fabs(((pow(sideOne,2.0) + (pow(sideTwo,2.0)))) > (pow(sideThree,2.0)))) {
            cout << "Triangle is an acute triangle!" << endl;
        }
    //Checks to see if its an obtuse triangle
        if (fabs(((pow(sideOne,2.0) + (pow(sideTwo,2.0)))) < (pow(sideThree,2.0)))) {
            cout << "Triangle is an obtuse triangle!" << endl;
        }
    
    }
    //Part 2: Triangle Stats
    
    //Triangle perimeter formula
        trianglePerimeter = sideOne + sideTwo + sideThree;
    
    //Area of triangle using the Heron formula
        
        heronsVariable = (trianglePerimeter)/2.0;
        
        heronsVariable2 = ((heronsVariable) * ((heronsVariable) - sideOne) * ((heronsVariable) - sideTwo) * (heronsVariable - sideThree));
        
        triangleArea = pow(heronsVariable2,0.5);
        
        cout << "\t The perimeter of the triangle is: " << trianglePerimeter << endl;
        cout << "\t The area of the triangle is: " << triangleArea << endl;
    
    
    
    //Part 3: Randomized Triangle
    srand( time(0));
    
    cout << "Please enter the maximum side length for the triangle you want the statistic of: " << endl << "\t Max Length: ";
    cin >> maxLength;
    for (int i = 0; i <= 0; ++i) {
        randomMaxLength1 = (rand () % maxLength);
        randomMaxLength2 = (rand () % maxLength);
        randomMaxLength3 = (rand () % maxLength);
        cout << "The random sides are : " << randomMaxLength1 << " , " << randomMaxLength2  << " , " << randomMaxLength3 << endl;
    
    
    //Change all variables to the random variables and execute.
    
    if (((randomMaxLength1 + randomMaxLength2) <= (randomMaxLength3)) || ((randomMaxLength1 + randomMaxLength3) <= (randomMaxLength2)) || ((randomMaxLength3 + randomMaxLength2) <= (randomMaxLength1))) {
        cout << "That isn't a triangle!" << endl;
    }
    
    else {
    //Ordering sides in length
        if (randomMaxLength1 > randomMaxLength3) {
            tempSide = randomMaxLength1;
            randomMaxLength1 = randomMaxLength3;
            randomMaxLength3 = tempSide;
        }
        if (randomMaxLength2 > randomMaxLength3) {
            tempSide = randomMaxLength2;
            randomMaxLength2 = randomMaxLength3;
            randomMaxLength3 = tempSide;
        }
    
    
     cout << "The side lengths in increasing order are " << randomMaxLength1 << " , " << randomMaxLength2 << " , " << randomMaxLength3 << endl;
    
    //Checks to see if its a right triangle
        if ((fabs(((pow(randomMaxLength1,2.0) + (pow(randomMaxLength2,2.0) - (pow(randomMaxLength3,2.0))))))) <= TOLERANCE) {
            cout << "Triangle is a right triangle!" << endl;
        }
    //Checks to see if its an acute triangle
        if (fabs(((pow(randomMaxLength1,2.0) + (pow(randomMaxLength2,2.0)))) > (pow(randomMaxLength3,2.0)))) {
            cout << "Triangle is an acute triangle!" << endl;
        }
    //Checks to see if its an obtuse triangle
        if (fabs(((pow(randomMaxLength1,2.0) + (pow(randomMaxLength2,2.0)))) < (pow(randomMaxLength3,2.0)))) {
            cout << "Triangle is an obtuse triangle!" << endl;
        }
    
    
    //Part 2: Triangle Stats
    
    //Triangle perimeter formula
        trianglePerimeter = randomMaxLength1 + randomMaxLength2 + randomMaxLength3;
    
    //Area of triangle using the Heron formula
        
        heronsVariable = (trianglePerimeter)/2.0;
        
        heronsVariable2 = ((heronsVariable) * ((heronsVariable) - randomMaxLength1) * ((heronsVariable) - randomMaxLength2) * (heronsVariable - randomMaxLength3));
        
        triangleArea = pow(heronsVariable2,0.5);
        
        cout << "\t The perimeter of the triangle is: " << trianglePerimeter << endl;
        cout << "\t The area of the triangle is: " << triangleArea << endl;
    
        }
    }
}