//-----------------------------------------------------------
// Name: Hand
//
// Description: represents a hand or partial hand of cards
//
//-----------------------------------------------------------

#include "Hand.h"

using namespace std;

void Hand::clear(){
    hand.clear();
}

void Hand::addCard(Card toAdd){
    hand.push_back(toAdd);
}

int Hand::getCount(){
    return hand.size();
}

Card Hand::getCard(int n){
    return hand[n];
}

// TODO
Hand Hand::getVisible(){
    // gets the visble part of a hand as a new hand
}

// TODO
int Hand::evaluate(){
    // what is the value of the hand
}
