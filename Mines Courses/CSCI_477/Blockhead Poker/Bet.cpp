//-----------------------------------------------------------
// Name: Bet
//
// Description: Represent a single player’s single bet.
//
//-----------------------------------------------------------

#include "Bet.h"

using namespace std;

Bet::Bet(int amount, int player){
    this->amount = amount;
    this->player = player;
}

int Bet::getAmount(){
    return amount;
}

int Bet::getPlayer(){
    return player;
}
