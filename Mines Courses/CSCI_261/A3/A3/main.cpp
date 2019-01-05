/*Assignment 3: Guess the Number
 *
 *Author: Carson Stevens
 *
 *Set a looping program that loops until the user guesses the correct number.
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
    
    //Seeding the random number with ctime
    srand( time(0));
    
    //Identidication of Global Variables
    int randomNumber = 0;
    int userGuess = 101;
    int greatThanRandom;
    int lessThanRandom;
    
    //Generate the random number
    randomNumber = rand() % 101;
    
    //Identify variables from specific hints
    greatThanRandom = (randomNumber + 25);
    lessThanRandom = (randomNumber - 25);
    
    cout << endl << "Hold onto your pants, we're about to play guess-the-numbah!" << endl;
    
    
    // Define when to give hints
    
    while (userGuess != randomNumber) {
        
        cout << "Pick a number between 0 and 100: " << endl;
        cin >> userGuess;
        
        if (((abs(userGuess - randomNumber)) <= 5) && (userGuess != randomNumber)) {
            cout << "Oooh you're close! ";
        }
        
        if ((userGuess) >= (greatThanRandom)) {
            cout << "Not even close! ";
        }
        
        if ((userGuess) <= (lessThanRandom)) {
            cout << "Not even close! ";
        }
        
        if (userGuess > randomNumber) {
            cout << "Too high. ";
        }
        
        if (userGuess < randomNumber) {
            cout << "Too low. ";
        }
    }
    
    if (userGuess == randomNumber) {
        cout << "That's the correct number!" << endl;
    }
}