//-----------------------------------------------------------
// Name: Bet
//
// Description: Represent a single player’s single bet.
//
//-----------------------------------------------------------

#ifndef MINES_COURSES_BET_H
#define MINES_COURSES_BET_H

#include <iostream>

class Bet(){
public:

    //Constructor
    Bet(int amount, int player);

    // player who made bet (0 or 1)
    int getPlayer();

    // Amount of bet if it is a raise
    int getAmount();

private:
    int amount;
    int player;
}
#endif //MINES_COURSES_BET_H
