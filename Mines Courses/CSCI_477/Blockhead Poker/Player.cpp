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

#include "Player.h"

using namespace std;

Player::Player(int id, int chips){
    this->id = id;
    this->chips = chips;
}

int Player::getID(){
    return this->id;
}

void Player::clearHand(){
    this->hand.clear();
    return;
}

void Player::dealCard(Card c){
    hand.addCard(c);
}

Hand Player::getHand(){
    return this->hand;
}

void Player::addChips(int chips){
    this->chips += chips;
}

int Player::getChips(){
    return this->chips;
}

int Player::getHandValue(){
    int sum = 0;
    for(Card card : this->hand.getHand()){
        sum += card.getValue();
    }
    return sum;
}