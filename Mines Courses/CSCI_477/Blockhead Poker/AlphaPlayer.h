//-----------------------------------------------------------
// Name: AlphaPlayer
//
// Description: This is a derived class of Player that
// evaluates domain information for the current AI player and
// decides on a player bet. This uses the getBet() method.
// The Alpha AI should use the attached rules in deciding the
// bet for the AI player.
//
//-----------------------------------------------------------

#ifndef MINES_COURSES_ALPHAPLAYER_H
#define MINES_COURSES_ALPHAPLAYER_H

#include "Player.h"

class AlphaPlayer : public Player{
public:

    AlphaPlayer(int id, int chips) : Player(id, chips){
    }

    int getBet(Hand opponent, BetHistory bh, int bet2player, bool canRaise, int pot);

};
#endif //MINES_COURSES_ALPHAPLAYER_H
