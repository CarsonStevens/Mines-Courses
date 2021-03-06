//-----------------------------------------------------------
// Name: Game
//
// Description: Plays a game between a player of type p0
// and p1. Chips is the number of chips each player
// respectively has at start and end of the game. reportFlag
// is just a flag to turn on and off I/O within the game
// (so you can do Monte Carlo runs without a lot of output).
// The method returns true if someone wants to quit the program.
//
//-----------------------------------------------------------


#include "Game.h"
#include "HumanPlayer.h"
#include "AlphaPlayer.h"

using namespace std;

bool Game::playGame(PlayerType p0, PlayerType p1,
        int &chips0, int &chips1, bool report){
    int temp;
    bool fold;
    int player1Bet;
    int player0Bet;
    HumanPlayer player0(0,chips0);
    AlphaPlayer player1(1, chips0);
//    if(p0 == HUMAN){
//        HumanPlayer player0(0, chips0);
//        AlphaPlayer player1(1, chips1);
//    }
//    else{
//        AlphaPlayer player0(0, chips0);
//        BetaPlayer player1(1, chips1);
//    }

    while(handCounter < 20) {
        cout << "Hand " << (handCounter+1) << " out of 20" << endl;
        int round = 1;
        player0.clearHand();
        player1.clearHand();
        //Reshuffle Deck for next hand.
        shuffleDeck();

        //Remove cost to enter pot: -10
        player0.addChips(-10);
        player1.addChips(-10);
        pot += 20;

        //Player0 goes first for hand
        if(handCounter % 2 == 0) {
            while(round < 4){

                // Values for each round
                bool call = false;
                fold = false;
                bool start = true;
                int raises = 0;
                bool bet = true;
                player0Bet = 0;
                player1Bet = 0;
                this->history.clearHistory();
                // Deal cards to Players for specific round
                dealCards((round - 1), player0, player1);
                cout << "Moving to bidding round " << round << endl << endl;
                // Does 1 round
                while (!fold) {
                    //Update if a player can bet
                    if (raises < 3) {
                        bet = true;
                    }
                    else{
                        bet = false;
                    }

                    //Player0 Turn
                    temp = player0.getBet(player1.getHand(), history, player1Bet, bet, pot);

                    //If human player, to quit
                    if (temp == -1) {
                        return true;
                    }
                    // Handles check if first bet of round is 1.
                    else if (temp == 0 && start) {
                        call = true;
                        player0Bet = temp-player1Bet;
                        pot += temp;
                        chips0 -= temp;
                    }

                    //Handle calling
                    else if (temp == player1Bet && !start) {
                        if(call){
                            round++;
                            break;
                        }
                        else{
                            call = true;
                            player0Bet = temp-player1Bet;
                            pot += temp;
                            Bet player_bet(temp, player0.getID());
                            history.addBet(player_bet);
                            chips0 -= temp;
                            if(raises == 3){
                                round++;
                                break;
                            }
                        }
                    }
                    else if (temp == 0 && call && !start){
                        round++;
                        break;
                    }

                    // Handles folding
                    else if (temp == 0 && !call && !start && player1Bet > 0) {
                        //Folded: hand ends pot goes to player1
                        fold = true;
                        player0Bet = 0;
                        //To break out of loop for hand
                        round = 4;
                        break;
                    }

                    //Handle raising
                    else {
                        // Chips subtracted from player in getBet() call
                        call = false;
                        fold = false;
                        raises++;
                        player0Bet = temp - player1Bet;
                        pot += temp;
                        Bet player_bet(temp, player0.getID());
                        history.addBet(player_bet);
                        chips0 -= temp;
                    }
                    start = false;

                    //Player1 Turn
                    //Update if the player can bet
                    if (raises < 3) {
                        bet = true;
                    }
                    else{
                        bet = false;
                    }

                    //Get bet from player1
                    temp = player1.getBet(player0.getHand(), history, player0Bet, bet, pot);

                    // If both players call
                    if (temp == player0Bet) {

                        player1Bet = temp - player0Bet;
                        pot += temp;
                        Bet player_bet(temp, player1.getID());
                        history.addBet(player_bet);
                        chips1 -= temp;
                        if (call) {
                            if(report){
                                cout << "Opponent also called your bet. Moving to next betting round." << endl;
                            }
                            round++;
                            break;
                        }
                        else{
                            call = true;
                            if(raises == 3){
                                round++;
                                if(report){
                                    cout << "Oponent called your bet." << endl;
                                    cout << "Moving to next betting round." << endl;
                                }
                                break;
                            }
                        }
                    }
                    else if (temp == 0  && player0Bet > 0) {
                        fold = true;
                        player1Bet = 0;
                        round = 4;
                        if(report){
                            cout << "Opponent folded this round." << endl;
                        }
                        break;
                        // Pot goes to player 1
                    }
                    else {
                        call = false;
                        fold = false;
                        raises++;
                        player1Bet = temp - player0Bet;
                        pot += temp;
                        Bet player_bet(temp, player1.getID());
                        history.addBet(player_bet);
                        chips1 -= temp;
                        if(report){
                            cout << "Opponent raised your bet " << player1Bet << " chips." << endl;
                        }
                    }
                }
            }



        }
        // Player1 starts hand
        else {
            while (round < 4) {

                // Values for each round
                bool call = false;
                fold = false;
                bool start = true;
                int raises = 0;
                bool bet = true;
                player0Bet = 0;
                player1Bet = 0;
                this->history.clearHistory();
                // Deal cards to Players for specific round
                dealCards((round - 1), player0, player1);
                cout << "Moving to bidding round " << round << endl << endl;

                // Does 1 round
                while (!fold) {
                    //Update if a player can bet
                    if (raises < 3) {
                        bet = true;
                    }else {
                        bet = false;
                    }

                    //Player1 Turn
                    temp = player1.getBet(player0.getHand(), history, player0Bet, bet, pot);

                    // Handles check if first bet of round is 1.
                    if (temp == 0 && start) {
                        if (report) {
                            cout << "Opponent checked." << endl;
                        }
                        call = true;
                        player1Bet = temp - player0Bet;
                        pot += temp;
                        chips1 -= temp;
                    }

                        //Handle calling
                    else if (temp == player0Bet && !start) {
                        if (call) {
                            cout << "Opponent also called. Moving to next betting round." << endl;
                            round++;
                            break;
                        }else {
                            call = true;
                            player1Bet = temp - player0Bet;
                            pot += temp;
                            Bet player_bet(temp, player1.getID());
                            history.addBet(player_bet);
                            chips1 -= temp;
                            if (raises == 3) {
                                round++;
                                if (report) {
                                    cout << "Moving to next betting round." << endl;
                                }
                                break;
                            }
                        }
                    } else if (temp == 0 && call && !start) {
                        if (report) {
                            cout << "Opponent also called. Moving to next bidding round." << endl;
                        }
                        round++;
                        break;
                    } else if (temp == 0 && !call && !start && player0Bet > 0) { //Handles Folds
                        //Folded: hand ends pot goes to player1
                        fold = true;
                        player1Bet = 0;
                        if (report) {
                            cout << "Opponent folded hand." << endl;
                        }
                        //To break out of loop for hand
                        round = 4;
                        break;
                    } else { //Handle raising
                        // Chips subtracted from player in getBet() call
                        call = false;
                        fold = false;
                        raises++;
                        player1Bet = temp - player0Bet;
                        pot += temp;
                        Bet player_bet(temp, player1.getID());
                        history.addBet(player_bet);
                        chips1 -= temp;
                        if (report) {
                            cout << "Opponent raised " << player1Bet << " chips." << endl;
                        }
                    }
                    start = false;

                    //Player0 Turn
                    //Update if the player can bet
                    if (raises < 3) {
                        bet = true;
                    }else {
                        bet = false;
                    }

                    //Get bet from player0
                    temp = player0.getBet(player1.getHand(), history, player1Bet, bet, pot);

                    if(temp == -1){
                        return true;
                    } else if (temp == player1Bet) { // If both players call
                        player0Bet = temp - player1Bet;
                        pot += temp;
                        Bet player_bet(temp, player0.getID());
                        history.addBet(player_bet);
                        chips0 -= temp;
                        if (call) {
                            round++;
                            break;
                        }else {
                            call = true;
                            if (raises == 3) {
                                round++;
                                break;
                            }
                        }
                    } else if (temp == 0 && player1Bet > 0) {
                        fold = true;
                        player0Bet = 0;
                        round = 4;
                        break;
                        // Pot goes to player 1
                    } else {
                        call = false;
                        raises++;
                        player0Bet = temp - player1Bet;
                        pot += temp;
                        Bet player_bet(temp, player0.getID());
                        history.addBet(player_bet);
                        chips0 -= temp;
                    }
                }
            }
        }
        //Hand over
        handCounter++;
        if(report){
            cout << "Your hand's total:\t" << player0.getHandValue() << endl;
            cout << "OPPONENT hand's total:\t" << player1.getHandValue() << endl;
        }
        //Determine winner and give pot;

        //For folds
        if(fold and player1Bet == 0){
            player0.addChips(pot);
            cout << endl << "You won the pot of " << pot << " chips!" << endl << endl;
            chips0 += pot;
            pot = 0;
        } else if(fold && player0Bet == 0){
            player1.addChips(pot);
            cout << endl << "You lost the pot of " << pot << " chips to your opponent!" << endl << endl;
            chips1 += pot;
            pot = 0;
        } else if(player0.getHandValue() == player1.getHandValue()){
            //tie so pot continues.
            cout << endl << "You tied! The pot of " << pot << " chips will continue to the next hand!" << endl << endl;
            continue;
        } else if(player0.getHandValue() > player1.getHandValue()){
            player0.addChips(pot);
            cout << endl << "You won the pot of " << pot << " chips!" << endl << endl;
            chips0 += pot;
            pot = 0;
        } else{
            player1.addChips(pot);
            cout << endl << "You lost the pot of " << pot << " chips to your opponent!" << endl << endl;
            chips1 += pot;
            pot = 0;
        }
    }
    return true;
}

void Game::shuffleDeck(){
    this->deck.clear();
    //initialize the deck
    //Gives cards their suit and rank
    int k = 0;
    for (int i = 0; i < 52; ++i){
        string suit = "";
        int rank = 0;
        bool face = false;

        // Assigns the first 13 cards their suit and gives them their rank. Rank is iterated by ++k. When the process has
        // gone through all the cards for the specific suit, it reset k back to 1. This happens when k is 14 because there
        // are 13 cards in each suit.
        ++k;
        if (k == 14){
            k = 1;
        }

        if (i < 13 ){
            suit = "Hearts";
            rank = k;
        }
        if ((i >=13) && (i < 26)){
            suit = "Spades";
            rank = k;
        }
        if ((i < 39) && (i >= 26)){
            suit = "Clubs";
            rank = k;
        }
        if (i >= 39){
            suit = "Diamonds";
            rank = k;
        }

        if (rank > 10 || rank == 1){
            if(rank == 1){
                suit = "Ace of " + suit;
            }
            if(rank == 11){
                suit = "Jack of " + suit;
            }
            if(rank == 12){
                suit = "Queen of " + suit;
            }
            if(rank == 13){
                suit = "King of " + suit;
            }
            rank = 10;
            face = true;
        }

        this->deck.push_back(Card(suit, rank, face));
    }

    //Shuffle deck
    srand(time(0));
    std::random_shuffle(this->deck.begin(), this->deck.end());
    return;
}

void Game::dealCards(int partOfRound, Player &p0, Player &p1){

    if(partOfRound == 0){
        //Deal two face up and one down to each player

        // Deals the face down cards
        p0.dealCard(this->deck.back());
        this->deck.pop_back();
        p1.dealCard(this->deck.back());
        this->deck.pop_back();

        // Deals one set of face up cards
        this->deck.back().setFaceup(true);
        p0.dealCard(this->deck.back());
        this->deck.pop_back();
        this->deck.back().setFaceup(true);
        p1.dealCard(this->deck.back());
        this->deck.pop_back();

        //Deals another set of face up cards;
        this->deck.back().setFaceup(true);
        p0.dealCard(this->deck.back());
        this->deck.pop_back();
        this->deck.back().setFaceup(true);
        p1.dealCard(this->deck.back());
        this->deck.pop_back();

    } else{
        //Deals another set of face up cards;
        this->deck.back().setFaceup(true);
        p0.dealCard(this->deck.back());
        this->deck.pop_back();
        this->deck.back().setFaceup(true);
        p1.dealCard(this->deck.back());
        this->deck.pop_back();
    }
}
