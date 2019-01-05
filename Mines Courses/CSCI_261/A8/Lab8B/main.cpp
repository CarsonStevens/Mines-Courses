/* CSCI261 Lab8B
 *
 * Description: store important dates.
 *
 * Author: Carson Stevens
 *
 */
 
#include <iostream>
#include <vector>
using namespace std;
#include "DateEvent.h"
 
int main(){
    
    Date original;
    //Extra Credit Shout out.
    cout << "SHOUT OUT\t" << original.getDate() << endl << endl;
    
    //Variables needed to define the event.
    int d;
    int m;
    int y;
    string title;
    string location;
    original.introduction();
    
    //Gets user input for the event.
    cout << "Title:\t";
    getline(cin, title);
    cout << "Location:\t";
    getline (cin, location);
    cout << "Month:\t";
    cin >> m;
    cout << "Day:\t";
    cin >> d;
    cout << "Year:\t";
    cin >> y;
    
    //Declares the event
    Event dateToSave;
    //Sets the user's input
    original.setDate(d, m, y, title, location);
    //Sets the event after the date is set.
    dateToSave.setEvent(original.getDate());
    //Prints the user's event.
    cout << "Event to save is: " << endl << dateToSave.getEvent(m, d, y);
    
//NEED TO INCLUDE SHOUT OUTS
     
     return 0;
}