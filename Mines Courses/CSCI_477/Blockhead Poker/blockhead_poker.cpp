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
#include "Game.h"

using namespace std;


int main(){

    bool quit = false;
    string = mode;

    cout << "\033[1;33m Welcome to Blockhead Poker!\033[0m" << endl;
    cout << "\033[1;32m Choose an option to get started:" << endl
         << "\tALPHA_AI     : Play against the AI implemented for an Alpha Player" << endl
         << "\tBETA_AI      : Play against the AI implemented for a Beta Player" << endl
         << "\tMONTE_CARLO  : Run Simulation to test AI success rate" << endl
         << "\tOPTIONS      : Display commands again" <<
         << "\tQUIT         : Exit the program\033[0m" << endl << endl;
    while(!quit){

        cout << "Enter a command to begin:\t";
        cin >> mode;

        if mode == "ALPHA_AI"{
            //Play game with alpha player
        }
        else if mode == "BETA_AI"{
            cout << "\033[1;31m Not Implemented Yet. Please select another command. \033[0m" << endl << endl;
        }
        else if mode == "MONTE_CARLO"{
            cout << "\033[1;31m Not Implemented Yet. Please select another command. \033[0m" << endl << endl;
        }
        else if mode == "OPTIONS"{
            cout << "\033[1;32m Choose an option to get started:" << endl
                 << "\tALPHA_AI     : Play against the AI implemented for an Alpha Player" << endl
                 << "\tBETA_AI      : Play against the AI implemented for a Beta Player" << endl
                 << "\tMONTE_CARLO  : Run Simulation to test AI success rate" << endl
                 << "\tOPTIONS      : Display commands again" <<
                 << "\tQUIT         : Exit the program\033[0m" << endl << endl;
        }
        else if mode == "QUIT"{
            cout << "\033[1;31m >>> EXITING <<< \033[0m" << endl;
            quit = true;
        }
        else{
            cout << "\033[1;31mNot a valid command.\033[0m" << endl;
        }
    }
    return 0;
}