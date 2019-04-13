//-----------------------------------------------------------
// Name: Card
//
// Description: represents a single card in a deck
//
//-----------------------------------------------------------

#include "Card.h"

using namespace std;

Card::Card(string cardName, int cardValue){
    name = cardName;
    value = cardValue;
    faceUp = false;
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