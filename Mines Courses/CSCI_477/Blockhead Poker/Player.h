//-----------------------------------------------------------
// Name: Player
//
// Description: This represents either a human or AI player.
// This is a base abstract class. You then should have
// derived classes for human, Alpha AI players, and Beta AI
// players (or more). The human player class will basically
// be I/O functions that presents info to the game player and
// guess decisions from a game player. The AI classes should
// make their own decisions without human input.
//
//-----------------------------------------------------------

#ifndef MINES_COURSES_PLAYER_H
#define MINES_COURSES_PLAYER_H

#include "Bet.h"
#include "Hand.h"

class Player(){
public:

    /*
     * Constructor:  initialize the layer with id 0 or 1
     * and starting chips.
     */
    Player(int id, int chips);

    /*
     * This should be an abstract method that passes all
     * of the necessary domain information of the game to
     * the player. Other information about the player like
     * the players hand should already be part of the Player
     * object. The method should then return the bet made by
     * the player. This bet represents the amount to be put
     * in the pot, so it would include the amount bet2player
     * which is the previous players raise. A bet of -1 is a
     * command to quit (only comes from the human player).
     * A bet of 0 is a fold IF there is a bet to the player
     * (otherwise, it's just a call).
     * int getBet( Hand opponent, BetHistory bh, int bet2player,
     *              bool canRaise, int pot)
     */
    int getBet(Hand opponent, BetHistory bh, int bet2player,
            bool canRaise, int pot);

    // get id number
    int getID();

    // clear the player's hand
    void clearHand();

    // Add card to the player's hand
    void dealCard(Card c);

    // get player's hand
    Hand getHand();

    // add (or subtract) chips from player
    void addChips(int chips);

    // get player's chip count
    int getChips();

private:
    int id;
    int chips;
    Hand hand;
}
#endif //MINES_COURSES_PLAYER_H
