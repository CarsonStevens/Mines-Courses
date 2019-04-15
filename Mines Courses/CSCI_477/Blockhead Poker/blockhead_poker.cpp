//========================================================
//
// File Name: blockhead_poker.cpp
//
// Author: Carson Stevens
//
// Course and Assignment: CSCI 477 Blockhead Poker
//
// Description: Create a blockhead poker driver that has AI
//              components, with Monte Carlo Simulation to
//              test stats
//
//=========================================================

#include <iostream>
#include <string>
#include "Player.h"
#include "Game.h"

using namespace std;


int main(){

    bool quit = false;
    string mode;
    Game game;
    cout << "Welcome to Blockhead Poker!" << endl;
    cout << "Choose an option to get started:" << endl
         << "\tALPHA_AI     : Play against the AI implemented for an Alpha Player" << endl
         << "\tBETA_AI      : Play against the AI implemented for a Beta Player" << endl
         << "\tMONTE_CARLO  : Run Simulation to test AI success rate" << endl
         << "\tOPTIONS      : Display commands again" << endl
         << "\tQUIT         : Exit the program" << endl << endl;
    while(!quit){

        cout << "Enter a command to begin:\t";
        cin >> mode;

        if (mode == "ALPHA_AI"){
            cout << endl << "Playing against Alpha AI" << endl;
            //Play game with alpha player
            PlayerType p0 = PlayerType::HUMAN;
            PlayerType p1 = PlayerType::ALPHA;
            game.playGame(p0, p1, 1000, 1000, true);

        }
        else if (mode == "BETA_AI"){
            cout << "Not Implemented Yet. Please select another command." << endl << endl;
        }
        else if (mode == "MONTE_CARLO"){
            cout << "Not Implemented Yet. Please select another command." << endl << endl;
        }
        else if (mode == "OPTIONS"){
            cout << "Choose an option to get started:" << endl
                 << "\tALPHA_AI     : Play against the AI implemented for an Alpha Player" << endl
                 << "\tBETA_AI      : Play against the AI implemented for a Beta Player" << endl
                 << "\tMONTE_CARLO  : Run Simulation to test AI success rate" << endl
                 << "\tOPTIONS      : Display commands again" << endl
                 << "\tQUIT         : Exit the program" << endl << endl;
        }
        else if (mode == "QUIT"){
            cout << ">>> EXITING <<<" << endl;
            quit = true;
        }
        else{
            cout << "Not a valid command." << endl;
        }
    }
    return 0;
}