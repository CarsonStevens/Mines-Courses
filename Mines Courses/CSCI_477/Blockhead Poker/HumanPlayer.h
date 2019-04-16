//-----------------------------------------------------------
// Name: HumanPlayer
//
// Description: This is a derived class of Player that
// presents domain information to the current human player
// through I/O and then allows the player to input their bet.
// Code should be implemented both to communicate to the game
// player the current status of the game (i.e. current hands
// showing, pot, bet history, etc.) and to validate the bets
// of the human player before returning the proper bet value.
// This uses the getBet() method.
//
//-----------------------------------------------------------

#ifndef MINES_COURSES_HUMANPLAYER_H
#define MINES_COURSES_HUMANPLAYER_H

#include "Player.h"

class HumanPlayer : public Player{
public:

    HumanPlayer(int id, int chips) : Player(id, chips){
    }

    int getBet(Hand opponent, BetHistory bh, int bet2player, bool canRaise, int pot) override;

};
#endif //MINES_COURSES_HUMANPLAYER_H
