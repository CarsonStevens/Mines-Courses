/* CSCI 261 Assignment 7
 *
 * Author: Stuart Shirley
 *CSCI 261 D
 *
 * cpp  file for  MagicItem class
 */
 
#include <iostream> // For cin, cout, etc.
#include "class.h"

using namespace std;

    MagicItem::MagicItem(){
        _prize = "helluva";
        _candyQ = "What candy was invented in golden?";
        _candyAnswer = "Jolly Rancher";
        _statueQ = "Who is the man that greets you when entering Golden?";
        _statueAnswer = "Buffalo Bill";
        _mQuestion = "What year was the M officially born?";
        _mAnswer = 1908;
        _userScore = 0;
        introduction();
    }
    string MagicItem::getPrize(){
        return _prize;
    }
    string MagicItem::getCandyQ(){
        return _candyQ;
    }
    
    string MagicItem::getCandyA(){
        return _candyAnswer;
    }
    
    string MagicItem::getStatueQ(){
        return _statueQ;
    }
    
    string MagicItem::getStatueA(){
        return _statueQ;
    }
    
    string MagicItem::getMQuestion(){
        return _mQuestion;
    }
    
    int MagicItem::getMAnswer(){
        return _mAnswer;
    }
    
    int MagicItem::setUserScore(int addScore){
        _userScore += addScore;
        return 0;
    }
    
    int MagicItem::getUserScore(){
        return _userScore;
    }
    
    bool MagicItem::checkM(const int& numGuess){
        return numGuess == _mAnswer;
    }
    
    bool MagicItem::checkCandy(const string& candyGuess){
        return candyGuess == _candyAnswer;
    }
    
    bool MagicItem::checkStatue(const string& userGuess){
        return userGuess == _statueAnswer;
    }
 
    void MagicItem::introduction(){
        cout << endl;
        cout << "Welcome to magic item!" << endl;
        cout << "Time to quiz your Golden knowledge!" << endl;
        cout << "Three questions will test your Knowledge, increasing in dificulty and prize value, one attempt for each question." << endl;
    }
    void MagicItem::awardPrize(MagicItem &golden){
        cout <<"Congratulations, you won " << golden.getUserScore() << " rousing chants of " << golden.getPrize() << " Engineer!" << endl;
        cout << "You're an ";
        for (int i =0; i < golden.getUserScore(); ++i){
            cout << golden.getPrize() << " ";
        }
        cout << " Engineer!" << endl;
    }