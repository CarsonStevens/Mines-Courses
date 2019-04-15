//-----------------------------------------------------------
// Name: HumanPlayer
//
// Description: This is a derived class of Player that
// presents domain information to the current human player
// through I/O and then allows the player to input their bet.
// Code should be implemented both to communicate to the game
// player the current status of the game (i.e. current hands
// showing, pot, bet history, etc.) and to validate the bets
// of the human player before returning the proper bet value.
// This uses the getBet() method.
//
//-----------------------------------------------------------
#include "HumanPlayer.h"


using namespace std;

int HumanPlayer::getBet(Hand opponent, BetHistory bh, int bet2player, bool canRaise, int pot){
    cout << endl << "\033[1;33m Pot:\t" << pot << "\033[0m" << endl << endl;

    // Handling AI if there is an AI;
    cout << "\033[1;31mOPPONENT:" << endl
         << "Bet: " << bet2player << endl
         << "Cards:\033[0m" << endl;
    cout << "\033[1;32m";
    for(Card card : opponent.getHand()){
        if(!card.isFaceup()){
            cout << "\t>>> FACE DOWN <<<" << endl;
        }
        else{
            if(card.getFace()){
                cout << "\t" << card.getName() << endl;
            }
            else{
                cout << card.getValue() << " of " << card.getName() << endl;
            }
        }
    }
    cout << "\033[0m" << endl;

    // Handle actual player
    cout << "\033[1;34mYour stats:" << endl <<
         "Chips:\t" << chips << endl << "Cards:\033[0m" << endl;
    cout << "\033[1;32m";
    for(Card card : hand.getHand()){
        if(card.getFace()){
            cout << "\t" << card.getName() << endl;
        }
        else{
            cout << card.getValue() << " of " << card.getName() << endl;
        }

    }
    cout << "\033[0m" << endl;
    if(canRaise) {
        bool valid = false;
        int raise1 = 0;
        while (!valid) {
            cout << "How much would you like to bet?" << endl << ">>> ";
            cin >> raise1;
            if (raise1 > 10 + bet2player) {
                cout << "That bet was too big. Maximum raise size is 10 chips." << endl;
            }
            else if (raise1 == -1) {
                cout << ">>> QUITING <<<" << endl;
                return -1;
            }
            else if (raise1 < bet2player && bet2player != 0) {
                cout << "Folded the hand." << endl;
                return 0;
            }
            else if (raise1 == bet2player) {
                cout << "Called the hand." << endl;
                chips -= bet2player;
                return bet2player;
            }
            else {
                cout << "Raised bet of: " << to_string(raise1 - bet2player) << " chips" << endl;
                chips -= (raise1 + bet2player);
                return raise1 + bet2player;
            }
        }
    }
    else{
        int bet = 0;
        cout << "Maximum number of raises met this round. Place a bet to call or fold:\t";
        cin >> bet;
        // Handle quiting
        if(bet == -1){
            cout << ">>> QUITING <<<" << endl;
            return -1;
        }
        //Handle folding
        else if (bet < bet2player){
            cout << "Folded the hand." << endl;
            return 0;
        }
        //Handle calling
        else{
            cout << "Called the hand." << endl;
            chips -= bet2player;
            return bet2player;
        }
    }
}
