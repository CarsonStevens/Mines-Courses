/*Lab2D, Pair Programming / Rock Paper Scissors Logic
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 *Set conditions for wins and losses
 */
 
 
#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
#include <ctime>
#include <string>
using namespace std;

int main() {
    srand( time(0) );
    
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
    
    
    //Logic behind which choice beats what
    
    if(playerChoice == computerChoiceWord) {
         cout << "It's a draw. Nobody wins!" << endl;
    }
    
    if ((computerChoiceWord == "Paper") && (playerChoice == "Rock")) {
        cout << "Paper beats Rock! Computer is the winner!" << endl;
    }
    
        if ((computerChoiceWord == "Paper") && (playerChoice == "Scissors")) {
        cout << "Scissors beats Paper! Player is the winner!" << endl;
    }
    
        if ((computerChoiceWord == "Scissors") && (playerChoice == "Rock")) {
        cout << "Rock beats Scissors! Player is the winner!" << endl;
    }
    
        if ((computerChoiceWord == "Rock") && (playerChoice == "Scissors")) {
        cout << "Rock beats Scissors! Computer is the winner!" << endl;
    }
    
        if ((computerChoiceWord == "Rock") && (playerChoice == "Paper")) {
        cout << "Paper beats Rock! Player is the winner!" << endl;
    }
    
        if ((computerChoiceWord == "Scissors") && (playerChoice == "Paper")) {
        cout << "Scissors beats Paper! Computer is the winner!" << endl;
    }
    
    return 0;
}