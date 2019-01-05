/*Lab6A, Rock Paper Scissors Part V
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 * 
 */
 
#include <fstream>
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
    char outcome;
    char userChoiceAny = 'Y';
    char computerChoiceChar;
    
    
    
    
    ofstream gameData("RockPaperScissors.txt");
    
    if ( gameData.fail() ) {
           cout << "Error opening output file";
           return 1;
    }
    
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
                computerChoiceChar = 'P';
                computerChoiceWord = "Paper";
            }
            if (computerChoice == 1) {
                computerChoiceChar = 'R';
                computerChoiceWord = "Rock";
            }
            if (computerChoice == 2) {
                computerChoiceChar = 'S';
                computerChoiceWord = "Scissors";
            }
        
            cout << "Computer choose " << computerChoiceWord << endl;
        
        
        //Logic behind which choice beats what
        
            if(playerChoice == computerChoiceWord) {
                ties++;
                outcome = 'T';
                cout << "It's a draw. Nobody wins!" << endl;
            }
        
            if ((computerChoiceWord == "Paper") && (playerChoice == "Rock")) {
                losses++;
                outcome = 'L';
                cout << "Paper beats Rock! Computer is the winner!" << endl;
            }
        
            if ((computerChoiceWord == "Paper") && (playerChoice == "Scissors")) {
                wins++;
                outcome = 'W';
                cout << "Scissors beats Paper! Player is the winner!" << endl;
            }
        
            if ((computerChoiceWord == "Scissors") && (playerChoice == "Rock")) {
                wins++;
                outcome = 'W';
                cout << "Rock beats Scissors! Player is the winner!" << endl;
            }
        
            if ((computerChoiceWord == "Rock") && (playerChoice == "Scissors")) {
                losses++;
                outcome = 'L';
                cout << "Rock beats Scissors! Computer is the winner!" << endl;
            }
        
            if ((computerChoiceWord == "Rock") && (playerChoice == "Paper")) {
                wins++;
                outcome = 'W';
                cout << "Paper beats Rock! Player is the winner!" << endl;
            }
        
            if ((computerChoiceWord == "Scissors") && (playerChoice == "Paper")) {
                losses++;
                outcome = 'L';
                cout << "Scissors beats Paper! Computer is the winner!" << endl;
            }
            
            cout << " Do you want to keep playing? Enter (Y/N) " << endl;
            cin >> userChoiceAny;
            userChoice = toupper(userChoiceAny);
            
            
            if (userChoice == 'N') {
            cout << "Thank for playing!" << endl << "You won " << wins << " game(s), lost " << losses << " game(s), and tied " << ties << " game(s)." << endl;
            }
            
            gameData << "Time: " << time(0) << " Human: " << playerChoice << ". Computer: " << computerChoiceChar << ". = " << outcome << endl;
        }
    gameData.close();
    
    
    
   
        
    return 0;
}