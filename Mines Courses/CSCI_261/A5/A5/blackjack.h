#include <iostream>
#include <vector>
using namespace std;

#pragma once

struct Card {
    int rank;
    string suit;
    int value;
    string properName;
};

void shuffleDeck(vector<Card> &cards);   

vector<Card> dealNextCard(vector<Card> &cards, int &total);

void printCard(vector<Card> &cards);

void cardsReset(vector<Card> &cards);

void result(int &wins, int &losses, int &playerScore, int &dealerScore, int &payout, int debt);