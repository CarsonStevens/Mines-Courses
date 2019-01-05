/* CSCI 261 Assignment 7
 *
 * Author: Stuart Shirley
 *CSCI 261 D
 *
 * header file for  class
 *
 */
 
 #pragma once
 #include <string>
 
 using namespace std;
 
 class MagicItem{
     public:
        MagicItem();
        string getCandyQ();
        string getCandyA();
        string getStatueQ();
        string getStatueA();
        string getMQuestion();
        string getPrize();
        int getMAnswer();
        int setUserScore(int addScore);
        int getUserScore();
        bool checkM(const int&);
        bool checkCandy(const string&);
        bool checkStatue(const string&);
        void awardPrize(MagicItem &golden);
     private:
        void introduction();
        string _prize;
        string _candyQ;
        string _candyAnswer;
        string _statueQ;
        string _statueAnswer;
        string _mQuestion;
        int _mAnswer;
        int _userScore;
        
 };