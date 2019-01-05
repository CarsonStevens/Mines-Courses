/*Assignment 5 / Blackjack
 *
 *Author: Carson Stevens
 *
 *Create a game of BlackJack
 */
 
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>
#include <ios>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cctype>



using namespace std;
#include "blackjack.h"



int main () {
    
    vector<Card> deck(52);
    int dealerValue = 0;
    bool wantToPlay = true;
    int playerValue = 0;
    string userChoice;
    string likeToPlay;
    int counter = 0;
    int winCounter = 0;
    int lossCounter = 0;
    srand(time(0));
    
    int winnings = 0;
    int bet = 0;
    
    
    //intializes the deck
    cardsReset(deck);
    
    
    // shuffleDeck(deck);
    // dealNextCard(deck, dealerValue);
    // cout << "Dealer total is: " << dealerValue << endl;
    // dealNextCard(deck, dealerValue);
    // cout << "Dealer total is: " << dealerValue << endl;
    //  printCard(deck);
    // for (int i = 0; i < deck.size(); ++i){
    //     cout << deck.at(i).value << endl;
    // }
    
    
    //Prints out how many cards are left after the dealer deals each time.
    // for (int i = 0; i < 52; ++i){
    //     shuffleDeck(deck);
    //     dealNextCard(deck, dealerValue);
    //     printCard(deck);
    // cout << "Dealer total is: " << dealerValue << endl;
    // cout << "There are " << deck.size() << " cards left" << endl;
    // }
    
    
    
    cout << "Welcome to BlackJack!" << endl << endl;
    cout << "Payout is 3-to-2" << endl;
    
    //Asks for intial bet value
    cout << "Please enter a bet you'd like to place:\t$";
    cin >> bet;
    cout << endl;
    
    //Will ask for bets until the bet is valid
    while (bet < 0){
        cout << endl << "Invalid bet! Please enter a positive integer!\t$";
        cin >> bet;
    }
    
    //Shuffles the deck intially.
    shuffleDeck(deck);
    
    //Deals the dealer the first card, prints the card, and shows the dealers total.
    dealNextCard(deck, dealerValue);
    cout << "Dealer showed the ";
    printCard(deck);
    cout << "Dealer total is: " << dealerValue << endl << endl;
    
    while ((wantToPlay == true)){
        if (counter > 0){
            cout << "Would you like to play again? Please type 'yes' or 'no'!\t";
            cin >> likeToPlay;
            
            if (likeToPlay == "yes"){
                
                //Resets the deck to have all the cards and shuffles them again.
                deck.clear();
                deck.resize(52);
                cardsReset(deck);
                
                //Resets the player and dealer scores to 0.
                dealerValue = 0;
                playerValue = 0;
                
                //Asks for the next bet
                cout << endl << endl;
                cout << "Please enter a bet you'd like to place:\t$";
                cin >> bet;
                
                //Makes sure that the bet is valid
                while (bet < 0){
                    cout << endl << "Invalid bet! Please enter a positive integer!\t$";
                    cin >> bet;
                }
                
                //Shuffles the deck and restarts the game.
                shuffleDeck(deck);
                dealNextCard(deck, dealerValue);
                cout << "Dealer showed the "; 
                printCard(deck);
                cout << endl <<  "Dealer total is: " << dealerValue << endl << endl;
            }
            else {
                
                //When the player is done, thanks them for player and tells them the results.
                cout << endl << endl << "Thanks for playing! ";
                cout << "You had " << winCounter << " win(s) and " << lossCounter << " loss(es)!";
                if (winnings < 0){
                    cout << endl << "You lost $" << abs(winnings) << "!" << endl;
                }
                else {
                    cout << endl << "You won $" << winnings << "!" << endl;
                }
                break;
            }
        }
        while (dealerValue < 22){
            
                //Deals the player the first card
                dealNextCard(deck, playerValue);
                cout << "You were dealt the ";
                printCard(deck);
                
                //Deals the player the second card
                dealNextCard(deck, playerValue);
                cout << "You were dealt the ";
                printCard(deck);
                
                //Prints the players total and asks if they want another card
                cout << "Your total is: " << playerValue << endl << endl;
                cout << "Would you like to hit or stand? Please type in 'hit' or 'stand'! \t";
                cin >> userChoice;
                
                while ((userChoice == "hit") && (playerValue < 22)){
                    
                    //Deals the player another card
                    dealNextCard(deck, playerValue);
                    cout << "You were dealt the ";
                    printCard(deck);
                    
                    //Prints the players total and asks if they want another card.
                    cout << "Your total is: " << playerValue << endl << endl;
                    if (playerValue < 22){
                        cout << "Would you like to hit or stand? Please type in 'hit' or 'stand'! \t";
                        cin >> userChoice;
                    }
                }
                
                if ((userChoice == "stand") || (playerValue < 22)){
                    while (dealerValue < 17){
                        
                        //Deals the dealer cards until they bust or over 17.
                        dealNextCard(deck, dealerValue);
                        cout << endl << "Dealer dealt the ";
                        printCard(deck);
                        cout << "Dealer total is: " << dealerValue << endl;
                    }    
                }
                break;    
            }
            
            //Prints who won, adds/subtracts bet, adds to wins or losses
            result(winCounter, lossCounter, playerValue, dealerValue, winnings, bet);
            
            //Gives the counter needed to ask the player if they want to play again.
            ++counter;
    }   
    return 0;
}


