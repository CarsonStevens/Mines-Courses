//-----------------------------------------------------------
// Name: Card
//
// Description: represents a single card in a deck
//
//-----------------------------------------------------------

#ifndef MINES_COURSES_CARD_H
#define MINES_COURSES_CARD_H

#include <iostream>
#include <string>

class Card(){
public:

    //Constructor
    Card(string cardName, int cardValue);

    //Return name of Card
    string getName();

    //Return value of Card
    int getValue();

    // True if all players can see the card
    bool isFaceup();

    // Set faceup attribute to true
    void setFaceup(bool faceup);

    // Get whether card is a face card
    bool getFace();

private:
    string name;
    int value;
    bool faceUp;
    bool face;
}



#endif //MINES_COURSES_CARD_H
