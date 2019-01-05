#include "blackjack.h"

void shuffleDeck(vector<Card> &cards){
    srand(time(0));
    
    
    //Chooses two random cards and swaps the 10000 times 
    for (int i = 0; i < 10000; i++){
        int j = rand() % cards.size();
        int k = rand() % cards.size();
        swap(cards.at(j), cards.at(k));
    }
    return;
}

    

vector<Card> dealNextCard(vector<Card> &cards, int &total){
    //Sets i equal to the card at the back of the deck.
    int i = (cards.size())-1;
    
    //Updates the Player of Dealer total with the drawn card.
    total = total + cards.at(i).value;
    return cards;
}

void printCard(vector<Card> &cards){
    // for (int i = 0; i < cards.size(); ++i){
    int i = cards.size()-1;
        if ((cards.at(i).rank == 1)) {
            cards.at(i).properName = "Ace";
            //cout << cards.at(i).properName << " of " << cards.at(i).suit << endl;   // " which has a value of " << cards.at(i).value << endl;
        }
        else if ((cards.at(i).rank == 11)){
            cards.at(i).properName = "Jack";
            //cout << "You were dealt the Jack of " << cards.at(i).suit << endl; // " which has a value of " << cards.at(i).value << endl;
        }
        else if ((cards.at(i).rank == 12)){
            cards.at(i).properName = "Queen";
            //cout << "You were dealt the Queen of " << cards.at(i).suit << endl; // " which has a value of " << cards.at(i).value << endl;
        }
        else if ((cards.at(i).rank == 13)){
            cards.at(i).properName = "King";
            //cout << cards.at(i).suit << endl;  which has a value of " << cards.at(i).value << endl;
        }
        else{
            cards.at(i).properName = to_string(cards.at(i).rank);
            //cout << "You were dealt the " << cards.at(i).properName << " of " << cards.at(i).suit << endl; // " which has a value of " << cards.at(i).value << endl; 
        }
        
        //Prints out the card that was dealt.
        cout << cards.at(i).properName << " of " << cards.at(i).suit << endl;
        //after printing/showing the user and dealer the card, removes it from the deck.
        cards.pop_back();
    // }
    return;
    

}


void cardsReset(vector<Card> &cards){
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
    return;
}

void result(int &wins, int &losses, int &playerScore, int &dealerScore, int &payout, int debt) {
    if ((playerScore > 21) && (dealerScore < 22)){
        cout << "You busted! Dealer wins!" << endl;
        payout = payout - debt;
        ++losses;
    }
    else if (playerScore == dealerScore){
        cout << "You both had the same score! Dealer wins!" << endl;
        payout = payout - debt;
        ++losses;
    }
    else if ((playerScore == 21) && (dealerScore == 21)){
        cout << "You both had 21. Dealer wins!" << endl;
        payout = payout - debt;
        ++losses;
    }
    else if ((playerScore > 21) && (dealerScore > 21)){
        cout << "You both busted. Dealer wins!" << endl;
        payout = payout - debt;
        ++losses;
    }
    else if ((playerScore < 21) && (dealerScore < playerScore)){
        cout << "You were closer to 21! You win!" << endl;
        payout = payout + 0.5 * debt;
        ++wins;
    }
    else if ((playerScore < 21) && (dealerScore < 21) && (dealerScore > playerScore)){
        cout << "Dealer was closer to 21! You lose!" << endl;
        payout = payout - debt;
        ++losses;
    }
    else if ((playerScore == 21) && (dealerScore != 21)){
        cout << "BLACKJACK! You win!" << endl;
        payout = payout + 0.5 * debt;
        ++wins;
    }
    else if ((playerScore < 22) && (playerScore != 21) && (dealerScore > 21)){
        cout << "Dealer busted! You win!"<< endl;
        payout = payout + 0.5 * debt;
        ++wins;
    }
    else if ((dealerScore == 21) && (playerScore != 21)){
        cout << "BLACKJACK! Dealer wins!" << endl;
        payout = payout - debt;
        ++losses;
    }
    cout << endl;
    return;
}