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

using namespace std;

class BetHistory{
public:

    // clears the bet history
    void clearHistory();

    // amount of bet
    void addBet(Bet bet);

    // number of bets in history
    int getCount();

    // get the nth bet in the history
    Bet getBet(int n);

private:

    vector<Bet> betHistory;
};
#endif //MINES_COURSES_BETHISTORY_H
