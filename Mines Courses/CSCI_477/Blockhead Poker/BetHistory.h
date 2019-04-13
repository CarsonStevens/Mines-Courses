//-----------------------------------------------------------
// Name: BetHistory
//
// Description: The history of bets in a single round (you
// need it for human and AI decision making).
//
//-----------------------------------------------------------

#ifndef MINES_COURSES_BETHISTORY_H
#define MINES_COURSES_BETHISTORY_H

#include "Bet.h"
#include <vector>
#include <string>
#include <iostream>

class BetHistory(){
public:

    // clears the bet history
    void clearHistroy();

    // amount of bet
    void addBet(Bet bet);

    // number of bets in history
    int getCount();

    // get the nth bet in the history
    Bet getBet(int n);

private:
    vector<Bet> betHistory = new vector<Bet>;

}
#endif //MINES_COURSES_BETHISTORY_H
