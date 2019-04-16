//-----------------------------------------------------------
// Name: AlphaPlayer
//
// Description: This is a derived class of Player that
// evaluates domain information for the current AI player and
// decides on a player bet. This uses the getBet() method.
// The Alpha AI should use the attached rules in deciding the
// bet for the AI player.
//
//-----------------------------------------------------------
#include "AlphaPlayer.h"

using namespace std;

int AlphaPlayer::getBet(Hand opponent, BetHistory bh, int bet2player, bool canRaise, int pot){
    //Rules for Alpha Type AI
    //1. A delta value is calculated by getting the value of the AI players hand and subtracting the value of the opponent’s visible hand.
    Hand visible = opponent.getVisible();
    double delta = this->hand.evaluate() - visible.evaluate();

    if(canRaise){

        //2. If there are still cards to draw…
        if(this->hand.getCount() < 5){
            //a. If there have been no bets prior…
            if(bh.getCount() == 0){
                //i. If delta > 10, raise 10
                if(delta > 10.0){
                    chips -= (10+bet2player);
                    return 10+bet2player;
                }
                //ii. If delta > 5, raise 5
                else if(delta > 5.0){
                    chips -= (5+bet2player);
                    return 5+bet2player;
                }
                //iii. If delta > 0, raise 1
                else if(delta > 0.0){
                    chips -= (1+bet2player);
                    return 1+bet2player;
                }
                //iv. Else call
                else{
                    chips -= bet2player;
                    return bet2player;
                }
            }
            //b. Else there have been prior bets
            else{
                //i. Calculate a pot_factor which is the size of the pot divided by 10 (int value)
                int pot_factor = pot/10;

                //ii. If the prior bet is a call
                if(bet2player == 0){
                    //1. If delta > 5-pot_factor, raise 10
                    if(delta > (5-pot_factor)){
                        chips -= (10+bet2player);
                        return 10+bet2player;
                    }
                    //2. If delta > 0-pot_factor, raise 1
                    else if(delta > (0-pot_factor)){
                        chips -= (1+bet2player);
                        return 1+bet2player;
                    }
                    //3. Else call
                    else{
                        chips -= bet2player;
                        return bet2player;
                    }
                }

                //iii. If the prior bet is a raise and is less than 1+pot_factor*2
                if(bet2player > 0 && bet2player < (1 + pot_factor*2)){
                    //1. If delta > 8-pot_factor, raise 10
                    if(delta > (8-pot_factor)){
                        chips-= (10+bet2player);
                        return 10+bet2player;
                    }
                    //2. If delta > -2-pot_factor, raise 1
                    else if(delta > (-2 - pot_factor)){
                        chips -= (1+bet2player);
                        return 1+bet2player;
                    }
                    //3. If delta > -4-pot_factor, call
                    else if(delta > (-4-pot_factor)){
                        chips -= bet2player;
                        return bet2player;
                    }
                    //4. Else fold
                    else{
                        return 0;
                    }
                }
                //iv. else
                else{
                    //1. If delta > 10-pot_factor, raise 10
                    if(delta > (10 - pot_factor)){
                        chips -= (10+bet2player);
                        return 10 + bet2player;
                    }
                    //2. If delta > 0-pot_factor, raise 1
                    else if(delta > (0-pot_factor)){
                        chips -= (1+bet2player);
                        return 1+bet2player;
                    }
                    //3. If delta > -2-pot_factor, call
                    else if(delta > (-2-pot_factor)){
                        chips -= bet2player;
                        return bet2player;
                    }
                        //4. Else fold
                    else{
                        return 0;
                    }
                }
            }
        }
        //3. Else last betting round
        else{
            //a. If there have been no bets prior…
            if(bh.getCount() == 0){
                //i. If delta > 10, raise 10
                if(delta > 10){
                    chips -= (10+bet2player);
                    return 10+bet2player;
                }
                //ii. If delta > 5, raise 5
                else if(delta > 5){
                    chips -= (5+bet2player);
                    return 5+bet2player;
                }
                //iii. Else call
                else{
                    chips -= bet2player;
                    return bet2player;
                }
            }
            //b. Else there have been prior bets
            else{
                //i. Calculate a pot_factor which is the size of the pot divided by 10 (int value)
                int pot_factor = pot/10;

                //ii. If the prior bet is a call
                if(bet2player == 0){
                    //1. If delta > 10-pot_factor, raise 10
                    if(delta > (10-pot_factor)){
                        chips -= (10+bet2player);
                        return 10+bet2player;
                    }
                    //2. Else call
                    else{
                        chips -= bet2player;
                        return bet2player;
                    }
                }
                //iii. If the prior bet is a raise and is less than 1+pot_factor*2
                else if(bet2player > 0 && bet2player < (1+pot_factor*2)){
                    //1. If delta > 6-pot_factor, raise 10
                    if(delta > (6-pot_factor)){
                        chips -= (10+bet2player);
                        return 10+bet2player;
                    }
                    //2. If delta > 2, call
                    else if(delta > 2){
                        chips -= bet2player;
                        return bet2player;
                    }
                    //3. Else fold
                    else{
                        return 0;
                    }
                }
                //iv. else
                else{
                    //1. If delta > 8-pot_factor, raise 10
                    if(delta > (8-pot_factor)){
                        chips -= (10+bet2player);
                        return 10+bet2player;
                    }
                    //2. If delta > 4, call
                    else if(delta > 4){
                        chips -= bet2player;
                        return bet2player;
                    }
                    //3. Else fold
                    else{
                        return 0;
                    }
                }
            }
        }
    }
    else{
        if(delta < 5){
            //Call
            chips -= bet2player;
            return bet2player;
        }
        //Fold
        else{
            return 0;
        }
    }
}
