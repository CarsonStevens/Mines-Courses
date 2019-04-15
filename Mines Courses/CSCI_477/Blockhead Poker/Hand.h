//-----------------------------------------------------------
// Name: Hand
//
// Description: represents a hand or partial hand of cards
//
//-----------------------------------------------------------

#ifndef MINES_COURSES_HAND_H
#define MINES_COURSES_HAND_H

#include <iostream>
#include <string>
#include <vector>
#include "Card.h"

using namespace std;

class Hand{
public:

    // clears the hand
    void clear();

    // add a card to the hand
    void addCard(Card);

    // how many cards are in the hand
    int getCount();

    // Gets the n'th card in the hand.
    Card getCard(int n);

    // gets the visble part of a hand as a new hand
    Hand getVisible();

    // what is the value of the hand
    int evaluate();

    vector<Card> getHand();

private:
    vector<Card> hand;
};
#endif //MINES_COURSES_HAND_H
