//-----------------------------------------------------------
// Name: BetHistory
//
// Description: The history of bets in a single round (you
// need it for human and AI decision making).
//
//-----------------------------------------------------------


#include "BetHistory.h"

using namespace std;

void BetHistory::clearHistory(){
    betHistory.clear();
}

void BetHistory::addBet(Bet bet){
    betHistory.push_back(bet);
}

int BetHistory::getCount(){
    return betHistory.size();
}

Bet BetHistory::getBet(int n){
    return betHistory[n];
}