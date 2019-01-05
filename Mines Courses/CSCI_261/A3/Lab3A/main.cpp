/*Lab3A, Pair Programming / Rock Paper Scissors Logic Loop
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 *Set conditions for wins and losses while looping
 */
 
 
#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
#include <ctime>
#include <string>
using namespace std;

int main() {
    
     //Defining variables for user
    char paper = 'P';
    char rock = 'R';
    char scissors = 'S';
    string playerChoice;
    int computerChoice;
    string computerChoiceWord;
    char userChoice = 'Y';
    char playerOutcome;
    int wins = 0;
    int losses = 0;
    int ties = 0;
    
    
    
    while (userChoice == 'Y') {
     //Prompt user for input
    
    cout << "Welcome one and all to a round of Rock, Paper, Scissors! Please enter P, R, or S:" << endl;
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
    
    srand( time(0));
    
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
            ties++;
            cout << "It's a draw. Nobody wins!" << endl;
        }
    
        if ((computerChoiceWord == "Paper") && (playerChoice == "Rock")) {
            losses++;
            cout << "Paper beats Rock! Computer is the winner!" << endl;
        }
    
        if ((computerChoiceWord == "Paper") && (playerChoice == "Scissors")) {
            wins++;
            cout << "Scissors beats Paper! Player is the winner!" << endl;
        }
    
        if ((computerChoiceWord == "Scissors") && (playerChoice == "Rock")) {
            wins++;
            cout << "Rock beats Scissors! Player is the winner!" << endl;
        }
    
        if ((computerChoiceWord == "Rock") && (playerChoice == "Scissors")) {
            losses++;
            cout << "Rock beats Scissors! Computer is the winner!" << endl;
        }
    
        if ((computerChoiceWord == "Rock") && (playerChoice == "Paper")) {
            wins++;
            cout << "Paper beats Rock! Player is the winner!" << endl;
        }
    
        if ((computerChoiceWord == "Scissors") && (playerChoice == "Paper")) {
            losses++;
            cout << "Scissors beats Paper! Computer is the winner!" << endl;
        }
        
        cout << " Do you want to keep playing? Enter (Y/N) " << endl;
        cin >> userChoice;
        
        if (userChoice == 'N') {
        cout << "Thank for playing!" << endl << "You won " << wins << " game(s), lost " << losses << " game(s), and tied " << ties << " game(s)." << endl;
        }
    }
    
    
    
    
   
        
    return 0;
}