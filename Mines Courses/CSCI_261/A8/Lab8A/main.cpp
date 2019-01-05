/* CSCI261 Lab8A
 *
 * Description: store important dates.
 *
 * Author: Carson Stevens
 *
 */
 
#include <iostream>
#include <vector>
using namespace std;
#include "Date.h"
 
int main(){
    
    Date original;
    cout << " is: " << original.getDate() << endl << endl;
    
    //Declaring the day, month, and year variables
    int d;
    int m;
    int y;
    
    //Gets the User's date to save
    original.introduction();
    cout << "Month:\t";
    cin >> m;
    cout << "Day:\t";
    cin >> d;
    cout << "Year:\t";
    cin >> y;

    //Sets the user's date
    original.setDate(d, m, y);
    
    //Tells the users the date that is saved.
    cout << "Your saved date is:\t" << original.getDate();
    
     
     return 0;
}