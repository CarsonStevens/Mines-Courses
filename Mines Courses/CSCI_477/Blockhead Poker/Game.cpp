//-----------------------------------------------------------
// Name: Game
//
// Description: Plays a game between a player of type p0
// and p1. Chips is the number of chips each player
// respectively has at start and end of the game. reportFlag
// is just a flag to turn on and off I/O within the game
// (so you can do Monte Carlo runs without a lot of output).
// The method returns true if someone wants to quit the program.
//
//-----------------------------------------------------------


#include "Game.h"

bool Game::playGame(PlayerType p0, PlayerType p1,
        int &chips0, int &chips1, bool reportFlag){

    //initialize the deck
    //Gives cards their suit and rank
    int k = 0;
    for (int i = 0; i < 52; ++i){


        // Assigns the first 13 cards their suit and gives them their rank. Rank is iterated by ++k. When the process has
        // gone through all the cards for the specific suit, it reset k back to 1. This happens when k is 14 because there
        // are 13 cards in each suit.
        ++k;
        if (k == 14){
            k = 1;
        }

        if (i < 13 ){
            cards.at(i).suit = "Hearts";
            cards.at(i).rank = k;
        }
        if ((i >=13) && (i < 26)){
            cards.at(i).suit = "Spades";
            cards.at(i).rank = k;
        }
        if ((i < 39) && (i >= 26)){
            cards.at(i).suit = "Clubs";
            cards.at(i).rank = k;
        }
        if (i >= 39){
            cards.at(i).suit = "Diamonds";
            cards.at(i).rank = k;
        }

        //Gives the cards their value.

        if (cards.at(i).rank >= 10){
            cards.at(i).value = 10;
        }
        else {
            (cards.at(i).value) = (cards.at(i).rank);
        }
    }
}
