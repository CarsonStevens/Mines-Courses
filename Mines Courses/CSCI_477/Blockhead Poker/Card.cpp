//-----------------------------------------------------------
// Name: Card
//
// Description: represents a single card in a deck
//
//-----------------------------------------------------------

#include "Card.h"
#include <string>

using namespace std;

Card::Card(string cardName, int cardValue, bool face){
    name = cardName;
    value = cardValue;
    faceUp = false;
    this->face = face;
}

string Card::getName(){
    return name;
}

int Card::getValue(){
    return value;
}

bool Card::isFaceup(){
    return faceUp;
}

void Card::setFaceup(bool faceup){
    faceUp = faceup;
}

bool Card:: getFace(){
    return face;
}