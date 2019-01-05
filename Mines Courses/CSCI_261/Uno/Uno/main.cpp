#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

using namespace std;

class Card{
    public:
  
    //Constructor
    Card(){}
    
    //Setters
    void setFace(string face){
        this->face = face;
    }
    void setValue(int value){
        this->value = value;
    }
    void setColor(string color){
        this->color = color;
    }
    void setChosenCardIndex(int index){
        chosenCardIndex = index;
    }
    
    //Getters
    string getFace(){
        return face;
    }
    int getValue(){
        return value;
    }
    string getColor(){
        return color;
    }
    int getChosenCardIndex(){
        return chosenCardIndex;
    }

    
    void printCard(){
        //Determine Card Color
        int color = -1;
        if(getColor() == "red")    color = 0;
        if(getColor() == "blue")   color = 1;
        if(getColor() == "green")  color = 2;
        if(getColor() == "yellow") color = 3;
        
        //Give Card Color
        switch(color){
            case(0) : face = ("\033[1;31m" + face + "\033[0m");  break;
            case(1) : face = ("\033[1;34m" + face + "\033[0m");  break;
            case(2) : face = ("\033[1;32m" + face + "\033[0m");  break;
            case(3) : face = ("\033[1;33m" + face + "\033[0m");  break;
            default : face =                 face;            ;  break;
        }


        //Print Card
        for (int i = 0; i < 7; i++){
            
            if(i == 0 && this->color != "none"){
                cout << face.substr(0, 17) << endl;
            }
            else if (i == 6 && this->color != "none"){
                cout << face.substr(66+7, 11+3) << endl;
            }
            else if (this-> color != "none"){
                cout << face.substr(i*11+7, 11) << endl;
            }
            else{
                cout << face.substr(i*11, 11) << endl;
            }
        }
        
        cout << "\033[0;m\033[0m";
    }
    
    private:
    
    //Card Variables
    int value;              //Holds the value of the card
    string face;            //Holds the ASCII art of the card
    string color;           //Holds the color of the card in a string
    int chosenCardIndex;    //Holds the index of the card in the player's hand
};

class Player{
    public:
    
    //Constructor
    Player(bool isHuman){
        this->isHuman = isHuman;
        vector<Card> hand;
    }
    
    //Getters
    vector<Card>& getHand(){
        return hand;
    }
    bool human(){
        return isHuman;
    }

    
    //Functions
    void removeCardFromHand(Card card){
        for(int i = 0; i < hand.size(); i++){
            if(card.getColor() == hand.at(i).getColor() && hand.at(i).getValue() == card.getValue()){
                hand.erase(hand.begin()+i);
            }
        }
    }
    
    private:
    
    //Player Variables
    bool isHuman;           //Player is human or not
    vector<Card> hand;      //Player's hand

};


class Game{
    private:
    
    //Game Variables
    double cardsPerLine = 7.0;      //Determines how many cards print per line
    int currentPlayer = 0;          //who the currentplayer index is
    bool validTopCard = true;       //Boolean to determine if a card was valid
    bool run = true;                //Boolean used to check if game is over
    int directionOfPlay = 1;        //Direction game play moves in
    int timeToSleep = 1;            //Time to sleep between plays
    
    //Game Containers
    vector<Card> discardPile;      //Holds the discard pile
    vector<Card> deck;             //Holds the card to be drawn(deck)
    vector<Player> players;        //Holds the players in the game
    
    
    public:
    
    Game(){
        vector<Card> discardPile;
        vector<Card> deck;
        vector<Player>players;
        printRules();
    }
    
    //Getters
    vector<Card>& getDeck(){
        return deck;
    }
    vector<Player>& getPlayers(){
        return players;
    }

    //////////////////////////////////////
    /*          Main Game               */
    //////////////////////////////////////
    void playGame(){
        
        //Setup Initial game
        printRules();
        dealCards();
        
        //Run Game
        while(run){
            
            //Clears screen for next output
            //cout << "\u001b[2J" << endl;
            printDisplay();
            
            if(players.at(currentPlayer).human()){
                while(!checkForAnyValidCard()){
                    cout << "You had no valid card. A card was drawn for you." << endl;
                    drawCard(0);
                    printDisplay();
                }
                humanChooseCard();
            }
            else{
                cout << "Player " << currentPlayer << "'s turn." << endl;
                computerChooseCard();
                if(players.at(currentPlayer).getHand().size() == 1){
                    cout << "UNO!" << endl;
                }
                this_thread::sleep_for(chrono::seconds(timeToSleep));
            }
            
            //Handle any function from a card put down
            Card card = discardPile.at(discardPile.size()-1);
            handleSpecialCard(card);
            
            //Update game for next turn
            run = checkEnd();
            updateCurrentPlayer();
        }
    }
    
    void handleSpecialCard(Card card){
        
        if(card.getValue() == 12){
            cout << "Direction of play has been reversed!" << endl;
            changeDirection();
        }
        
        // only print if human for draw cards
        else if(card.getValue() == 14){
            if(currentPlayer == players.size()-1){
                for(int i = 0; i < 4; i++){
                    drawCard(0); 
                }
                cout << "You had to draw 4 cards. Yikes!!" << endl;
            }
            else{
                for(int i = 0; i < 4; i++){
                    drawCard(currentPlayer+1);
                }
            }
        }
        
        else if(card.getValue() == 11){
            updateCurrentPlayer();
        }
        
        else if(card.getValue() == 10){
            if(currentPlayer == players.size()-1){
                for(int i = 0; i < 2; i++){
                    drawCard(0);
                }
                cout << "You had to draw 2 cards. Yikes!!" << endl;
            }
            else{
                for(int i = 0; i < 2; i++){
                    drawCard(currentPlayer+1);
                }
            }
        }
    }
    
    //Game Functions
    void dealCards(){
        
        random_shuffle(deck.begin(), deck.end());
        
        for(int i = 0; i < 7; i++){
            for(int j = 0; j < players.size(); j++){
                    Card card = deck.at(deck.size()-1);
                    players.at(j).getHand().push_back(card);
                    deck.pop_back();
            }
        }
        
        Card card = deck.at(deck.size()-1);
        
        if(card.getValue() == 14){
            while(deck.at(deck.size()-1).getValue() != 14){
                random_shuffle(deck.begin(), deck.end());
            }
            card = deck.at(deck.size()-1);
        }
        
        if(card.getValue() == 13){
            int check = -1;
            while(check >= 4 && check < 0){
                cout << "What color would you like to choose for your wild card?" << endl 
                << "\t(0)Red\n\t(1)Blue\n\t(2)Green\n\t(3)Yellow\n";
                cout << "Color: ";
                cin >> check;
                if(check >= 4 && check < 0){
                    cout << "That is not a valid input. Try again." << endl;
                    
                }
            }
            
            if(check == 0) card.setColor("red");
            if(check == 1) card.setColor("blue");
            if(check == 2) card.setColor("green");
            if(check == 3) card.setColor("yellow");
        }
        
        if(card.getValue() == 10 || card.getValue() == 11 || card.getValue() == 12){
            handleSpecialCard(card);
        }
        
        //Handle reverse and skip here too.
        discardPile.push_back(card);
        deck.pop_back();
    }
    
    bool checkEnd(){
        for(int i = 0; i < players.size(); i++){
            if(players.at(i).getHand().size() == 0){
                if(currentPlayer == 0){
                    cout << "You won!" << endl;
                }else{
                    cout << "Player " << currentPlayer << " won!" << endl;
                }
                return false;
            }
        }

        return true;
    }
    
    void printRules(){
        cout << endl << "WELCOME TO UNO!" << endl << endl << "Rules:" << endl;
        cout << "The first player is normally the player to the left of the dealer (you can also choose the youngest player)\nand gameplay usually follows a clockwise direction. Every player views his/her cards and tries to match the card in the Discard Pile.\n" <<
        "You have to match either by the number, color, or the symbol/Action. For instance, if the Discard Pile has a red\ncard that is an 8 you have to place either a red card or a card with an 8 on it. You can also play a Wild card (which can alter current color in play).\n" <<
        "If the player has no matches or they choose not to play any of their cards even though they might have a match,\nthey must draw a card from the Draw pile. If that card can be played, play it. Otherwise, the game moves on to the next person in turn.\n" << 
        "You can also play a Wild card, or a Wild Draw Four card on your turn." << endl << endl << endl;
    }
    
    void printDisplay(){
        
        //Print Discard Pile card
        cout << endl << "\033[1;4;35mDiscard Pile:\033[0m" << endl;
        discardPile.at(discardPile.size()-1).printCard();
        cout << endl;
        if(currentPlayer == 0){
            
            cout << "\033[1;4;35mYour Hand:\033[0m" << endl << endl;
        
            //Filter cards into color vectors
            vector<Card> reds;
            vector<Card> yellows;
            vector<Card> blues;
            vector<Card> greens;
            vector<Card> nones;
            for(int i = 0; i < players.at(0).getHand().size(); i++){
                players.at(0).getHand().at(i).setChosenCardIndex(i);
                if(players.at(0).getHand().at(i).getColor() == "red"){
                    reds.push_back(players.at(0).getHand().at(i));
                }
                if(players.at(0).getHand().at(i).getColor() == "yellow"){
                    yellows.push_back(players.at(0).getHand().at(i));
                }
                if(players.at(0).getHand().at(i).getColor() == "blue"){
                    blues.push_back(players.at(0).getHand().at(i));
                }
                if(players.at(0).getHand().at(i).getColor() == "green"){
                    greens.push_back(players.at(0).getHand().at(i));
                }
                if(players.at(0).getHand().at(i).getColor() == "none"){
                    nones.push_back(players.at(0).getHand().at(i));
                }
            }
            //Format and print rows
            //for reds
            cout << " \033[1;4;36mRed Cards:\033[0m" << endl;
            string finalRed = makeRows(reds);
            finalRed = "\033[1;31m" + finalRed + "\033[0m";
            cout << finalRed << endl;
            //yellow
            cout << " \033[1;4;36mYellow Cards:\033[0m" << endl;
            string finalYellow = makeRows(yellows);
            finalYellow = "\033[1;33m" + finalYellow + "\033[0m";
            cout << finalYellow << endl;
            //blue
            cout << " \033[1;4;36mBlue Cards:\033[0m" << endl;
            string finalBlue = makeRows(blues);
            finalBlue = "\033[1;34m" + finalBlue + "\033[0m";
            cout << finalBlue << endl;
            //green
            cout << " \033[1;4;36mGreen Cards:\033[0m" << endl;
            string finalGreen = makeRows(greens);
            finalGreen = "\033[1;32m" + finalGreen + "\033[0m";
            cout << finalGreen << endl;
            //none
            cout << " \033[1;4;36mWild Cards:\033[0m" << endl;
            string finalNone = makeRows(nones);
            cout << finalNone << endl << endl;
        }
    }
    
    
    bool checkValidCard(Card card){
        if(card.getColor() == discardPile.at(discardPile.size()-1).getColor()){
            return true;
        }
        if(card.getValue() == discardPile.at(discardPile.size()-1).getValue()){
            return true;
        }
        if(card.getValue() == 13 || card.getValue() == 14){
            return true;
        }
        return false;
    }
    
    //Make the rows for printing players hand
    string makeRows(vector<Card> cards){
        //Make all cards into one string
        string cards_input = "";
        for(int i = 0; i < cards.size(); i++){
            cards_input += cards.at(i).getFace();
        }
        
        //Variables for final string
        string final = "";
        int l = 0;  //For printing index on current row
        int m = 0;  //For printing index with multiple rows
        
        //for loop for number of rows to print
        for(int i = 0; i < ceil(cards.size()/cardsPerLine); i++){
            
            //Handles index printing
            while(l < cards.size()){
                cout << "    (" << cards.at(l + m).getChosenCardIndex() << ")    \t";
                l++;
                if(l == cardsPerLine){
                    l =0;
                    m += cardsPerLine;
                    break;
                }
            }
            cout << endl;
            
            //for the current line of each card
            for(int j = 0; j < 7; j++){
                //for the amount of cards in that line
                for(int k = 0; k < cardsPerLine && k < cards.size(); k++){
                    final += cards_input.substr(j*11+ k*77, 11) + "\t";
                }
                final += "\n";
            }
        }
        return final;
    }
    
    //Update who the current player is
    void updateCurrentPlayer(){
        currentPlayer += directionOfPlay;
        if(currentPlayer == -1){
            currentPlayer = players.size() - 1;
        }
        currentPlayer = currentPlayer % players.size();
    }
    
    //Check if currentPlayer has any valid cards
    bool checkForAnyValidCard(){
        for(int i = 0; i < players.at(currentPlayer).getHand().size(); i++){
            if(checkValidCard(players.at(currentPlayer).getHand().at(i))){
                return true;
            }
        }
        return false;
    }
    
    void changeDirection(){
        directionOfPlay *= -1;
    }
    
    //Called to make a computer player choose a card from hand
    void computerChooseCard(){
        //Chooses first valid card
        while(!checkForAnyValidCard()){
            drawCard(currentPlayer);
        }
        for(int i = 0; i < players.at(currentPlayer).getHand().size(); i++){
            if(checkValidCard(players.at(currentPlayer).getHand().at(i))){
                Card card = players.at(currentPlayer).getHand().at(i);
                if(card.getValue() == 13 || card.getValue() == 14){
                    int color = rand() % 4;
                    if(color == 0) card.setColor("red");
                    if(color == 1) card.setColor("blue");
                    if(color == 2) card.setColor("green");
                    if(color == 3) card.setColor("yellow");
                }

                discardPile.push_back(card);
                players.at(currentPlayer).getHand().erase(players.at(currentPlayer).getHand().begin()+i);
                return;
            }
        }

    }
    
    void humanChooseCard(){
        int input = -1;
        bool flag = false;
        while(input < 0 || input > players.at(0).getHand().size()-1 || !flag){
            cout << "Enter the card you wish to play: ";
            cin >> input;
            if(input < 0 || input > players.at(0).getHand().size()-1){
                cout << "That is not a valid index choice. Try again." << endl;
                continue;
            }
            flag = checkValidCard(players.at(0).getHand().at(input));
            if(!flag){
                cout << "That is not a valid card choice. Try again." << endl;
            }
        }
        Card card = players.at(0).getHand().at(input);
        if(card.getValue() == 13 || card.getValue() == 14){
            int check = -1;
            while(check > 3 || check < 0){
                cout << "What color would you like to choose for your wild card?" << endl 
                << "\t(0)Red\n\t(1)Blue\n\t(2)Green\n\t(3)Yellow\n";
                cout << "Color: ";
                cin >> check;
                if(check > 3 || check < 0){
                    cout << "That is not a valid input. Try again." << endl;
                    
                }
            }
            if(check == 0) card.setColor("red");
            if(check == 1) card.setColor("blue");
            if(check == 2) card.setColor("green");
            if(check == 3) card.setColor("yellow");
        }
        discardPile.push_back(card);
        players.at(0).getHand().erase(players.at(0).getHand().begin()+input);
    }
    
    //Draws card from deck for current player
    void drawCard(int player){
        
        //Check to make sure there is a card to draw.
        //  if not, shuffle discardPile and use it.
        if(deck.size() == 0){
            deck = discardPile;
            discardPile.clear();
            random_shuffle(deck.begin(), deck.end());
            if(deck.at(deck.size()-1).getValue() == 14){
                while(deck.at(deck.size()-1).getValue() != 14){
                    random_shuffle(deck.begin(), deck.end());
                }
            }
            Card card = deck.at(deck.size()-1);
            if(card.getValue() == 13 || card.getValue() == 14){
                if(currentPlayer == 0){
                    int check = -1;
                    while(check >= 4 && check < 0){
                        cout << "What color would you like to choose for your wild card?" << endl 
                        << "\t(0)Red\n\t(1)Blue\n\t(2)Green\n\t(3)Yellow\n";
                        cout << "Color: ";
                        cin >> check;
                        if(check >= 4 && check < 0){
                            cout << "That is not a valid input. Try again." << endl;
                            
                        }
                    }
                    if(check == 0) card.setColor("red");
                    if(check == 1) card.setColor("blue");
                    if(check == 2) card.setColor("green");
                    if(check == 3) card.setColor("yellow");
                }
                else{
                    int color = rand() % 4;
                    if(color == 0) card.setColor("red");
                    if(color == 1) card.setColor("blue");
                    if(color == 2) card.setColor("green");
                    if(color == 3) card.setColor("yellow");
                }
            }
            if(card.getValue() == 10 || card.getValue() == 11 || card.getValue() ==12){
                handleSpecialCard(card);
            }

            discardPile.push_back(card);
            deck.pop_back();
        }
        
        //Reset Wild Card if it was already used
        if(deck.at(deck.size()-1).getValue() == 13 || deck.at(deck.size()-1).getValue() == 14){
            deck.at(deck.size()-1).getColor() = "none";
        }
        //Deal card to player
        players.at(player).getHand().push_back(deck.at(deck.size()-1));
        deck.pop_back();
    }
};



//Creates the Players for the game;
void createPlayers(Game &UnoGame){
    
    //Get amount of players to play against from user
    int playerCount = 0;
    while(playerCount < 1 || playerCount > 9){
        cout << "Enter the amount of players (1-9) you wish to play against: ";
        cin >> playerCount; 
        if(playerCount < 1 || playerCount > 9){
            cout << "That was not a valid number. Please try again." << endl;
        }
        else{
            break;
        }
    }
    
    //Create user player
    Player human = Player(true);
    UnoGame.getPlayers().push_back(human);
    
    //Create computer players
    for(int i = 0; i < playerCount; i++){
        Player playa = Player(false);
        UnoGame.getPlayers().push_back(playa);
    }
}

//Sets up the cards for the deck
void setup(Game &UnoGame){
    /*
     *Array of amounts of cards to compose
     *Mapped by this order one 0 card, two 1 cards, two 2s, 3s, 4s, 5s, 6s, 7s, 8s and 9s, 
     *2 Draw Two cards, 2 Skip cards, 2 Reverse cards, 4 Wild cards, and 4 Wild Draw Four cards. 
    */
    int card_numbers [15] = {1,2,2,2,2,2,2,2,2,2,2,2,2,4,4};
    
    //Variables to read in cards
    ifstream cards_input;
    cards_input.open("cards.1.txt");
    string card_face = "";

    //Reads all types of cards
    for(int j = 0; j < 15; j++){
        
        //Reads the actual card
        getline(cards_input, card_face);
        
        //Creates the right number of cards in their colors
        for(int k = 0; k < card_numbers[j]; k++){
            Card card;
            card.setValue(j);
            
            //Colored Cards
            if(j < 13){
                
                //for red
                card.setFace(card_face);
                card.setColor("red");
                UnoGame.getDeck().push_back(card);
                
                //for blue
                card.setFace(card_face);
                card.setColor("blue");
                UnoGame.getDeck().push_back(card);
                
                //for yellow
                card.setFace(card_face);
                card.setColor("yellow");
                UnoGame.getDeck().push_back(card);
                
                //for green
                card.setFace(card_face);
                card.setColor("green");
                UnoGame.getDeck().push_back(card);
            
            }
            //Wild cards (no color)
            else{
                card.setFace(card_face);
                card.setColor("none");
                UnoGame.getDeck().push_back(card);
            }
        }
        
    }
    //TEST: should be 108
    //cout << UnoGame.getDeck().size() << endl;
}

int main(){
    
    //Create Game
    Game UnoGame;
    
    //Initialize Game
    setup(UnoGame);
    createPlayers(UnoGame);

    //Run Game
    UnoGame.playGame();
}