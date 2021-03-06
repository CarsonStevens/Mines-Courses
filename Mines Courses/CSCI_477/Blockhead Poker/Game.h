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

#ifndef MINES_COURSES_GAME_H
#define MINES_COURSES_GAME_H

#include <iostream>
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand
#include "Card.h"
#include "Player.h"
#include "BetHistory.h"



class Game{
public:

    bool playGame(PlayerType p0, PlayerType p1, int &chips0, int &chips1, bool Flag);
    void shuffleDeck();
    void dealCards(int partOfRound, Player &p0, Player &p1);

private:

    vector<Card> deck;
    BetHistory history;
    int handCounter = 0;
    int pot = 0;
};
#endif //MINES_COURSES_GAME_H
