//-----------------------------------------------------------
// Name: Hand
//
// Description: represents a hand or partial hand of cards
//
//-----------------------------------------------------------

#include "Hand.h"

using namespace std;

void Hand::clear(){
    this->hand.clear();
}

void Hand::addCard(Card toAdd){
    this->hand.push_back(toAdd);
}

int Hand::getCount(){
    return this->hand.size();
}

Card Hand::getCard(int n){
    return this->hand[n];
}


Hand Hand::getVisible(){
    Hand visible;
    for(Card card : this->hand){
        if(card.isFaceup()){
            visible.addCard(card);
        }
    }
    return visible;
}

vector<Card> Hand::getHand(){
    return this->hand;
}


int Hand::evaluate(){
    int sum = 0;
    for(Card card : this->hand){
        sum += card.getValue();
    }
    return sum;
}
