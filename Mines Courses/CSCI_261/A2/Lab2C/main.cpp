/*Lab2C, Pair Programming / State your Choice
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 *Define what the cryptic variables mean and set them equal to things that the user can understand.
 */
 
 
#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
#include <ctime>
#include <string>
using namespace std;

int main() {
    
    srand(time(0));
    
    //Defining variables for user
    char Paper = 'P';
    char Rock = 'R';
    char Scissors = 'S';
    string playerChoice;
    int computerChoice;
    string computerChoiceWord;
    
    //Prompt user for input
    cout << "Welcome to Rock, Paper, Scissors! Please enter P, R, or S:" << endl;
    cin >> playerChoice;
    
    if (playerChoice == "P") {
        playerChoice = "Paper";
    }
    if (playerChoice == "R") {
        playerChoice = "Rock";
    }
    if (playerChoice == "S") {
        playerChoice = "Scissors";
    }
    if (playerChoice == "p") {
        playerChoice = "Paper";
    }
    if (playerChoice == "r") {
        playerChoice = "Rock";
    }
    if (playerChoice == "s") {
        playerChoice = "Scissors";
    }
    
    
    //Print user choice
    cout << '\n' << "Player choose " << playerChoice << endl;
    
    //Randomly generate computer choice number
    
    for (int i = 0; i <= 0; ++i) {
        computerChoice = rand () % 3;
    }

    
    // Changes computerChoice to the actual word
    
    if (computerChoice == 0) {
            computerChoiceWord = "Paper";
    }
    if (computerChoice == 1) {
            computerChoiceWord = "Rock";
    }
    if (computerChoice == 2) {
            computerChoiceWord = "Scissors";
    }
    
    cout << "Computer choose " << computerChoiceWord << endl;
    
    return 0;
}