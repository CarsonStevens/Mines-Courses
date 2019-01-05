/* CSCI261 Section E Assignment 8
 *
 * Description: If a user solves the puzzle, award a prize
 *
 * Author: Carson Stevens
 * Work from: (Ashley Piccone, Section C), (Stuart Shirley, Section D), (Devvrat Singh, Section B)
 */
 
#include <iostream>
#include <cmath>
#include <string>
#include <vector>
 
using namespace std;
#include "MathPattern.h"
#include "Bunny.h"
#include "class.h"
#include "MagicItem.h"

int main(){
    
    int userChoice;
    
    //Gives the intro. Depending on the user's input number, selects the game they want to play.
    cout << "Please select the game to win a prize. Each contains a unique one!";
    cout << endl << "Please type: " << endl << "'1' for the math pattern game" << endl;
    cout << "'2' for the bunny game" << endl << "'3' for the Golden History game" << endl;
    cout << "'4' for the trivia game" << endl << "'5' for all the games" << endl;
    cin >> userChoice;
    cout << endl;
    
    if(userChoice == 1){
    
        MathPattern answer;         //Defines the users input for the class MagicItem
        
        getTheAnswer(answer);       //Prints out the riddle, and gets the user's input
        
        printResults(answer);       //Prints out the prize or the solution depending on
                                    //correctness
    }
    
    else if (userChoice == 2){
        vector<Bunny> bunVec; //makes a vector of bunnies
        Bunny firstBun;
        bunVec.push_back(firstBun);
        
        firstBun.introduction(firstBun); //calls intro function
        
        char userChar;
        while (userChar != 'q') {
            cin >> userChar;
            if (userChar == 'h') {
                firstBun.bunnyHop();
            }
            else if (userChar == 'f') {
                firstBun.feedBunny();
                firstBun.getSize();
            }
            else if (userChar == 'b') {
                firstBun.breedBunny(bunVec);
            }
            else {
                if (userChar != 'q') {
                    cout << "Type a valid letter." << endl;
                }
            }
        }
    }
    
    else if (userChoice == 3){
        MagicItem golden;
    
        //ask first question, check answer, and award points
        int numGuess;
        
        cout << golden.getMQuestion() << endl;
        cin >> numGuess;
        
        if(golden.checkM(numGuess)){
            golden.setUserScore(1);
        }else{
            cout << "incorrect year" << endl;
            cout << "your score " << golden.getUserScore() << endl;
        }
        
        //ask second question, check answer, and award points
        cin.ignore();
        string candyGuess;
        
        cout << golden.getCandyQ() << endl;
        getline(cin,candyGuess);
        
        if(golden.checkCandy(candyGuess)){
            golden.setUserScore(2);
        }else{
            cout << "incorrect candy" << endl;
            cout << "your score " <<golden.getUserScore() << endl;
        }
        //ask third question, check answer, and award points
        //cin.ignore();
        string personGuess;
        
        cout << golden.getStatueQ() << endl;
        getline(cin,personGuess);
        
        if(golden.checkStatue(personGuess)){
            golden.setUserScore(5);
        }else{
            cout << "incorrect person" << endl;
            cout << "your score " <<golden.getUserScore() << endl;
        }
        //award prize based on user points
        golden.awardPrize(golden);

    }
    
    if(userChoice == 4){
        
        JackpotTrivia diamonds; // Defining the variable titled diamonds
	    diamonds.introduction(); // Calling the introduction function using the dot product operator
	    diamonds.trivia1();  // Calling the trivia1 function using the dot product operator
        
    }
    
    if(userChoice == 5){
        //Loops through to play all of the games.
        for(int i = 1; i < 5; i++){
            userChoice = i;
            
            cout << endl << endl << "Game #" << i << endl << endl;
            
            
            if(userChoice == 1){
        
            MathPattern answer;         //Defines the users input for the class MagicItem
            
            getTheAnswer(answer);       //Prints out the riddle, and gets the user's input
            
            printResults(answer);       //Prints out the prize or the solution depending on
                                        //correctness
            }
            
            else if (userChoice == 2){
                vector<Bunny> bunVec; //makes a vector of bunnies
                Bunny firstBun;
                bunVec.push_back(firstBun);
                
                firstBun.introduction(firstBun); //calls intro function
                
                char userChar;
                while (userChar != 'q') {
                    cin >> userChar;
                    if (userChar == 'h') {
                        firstBun.bunnyHop();
                    }
                    else if (userChar == 'f') {
                        firstBun.feedBunny();
                        firstBun.getSize();
                    }
                    else if (userChar == 'b') {
                        firstBun.breedBunny(bunVec);
                    }
                    else {
                        if (userChar != 'q') {
                            cout << "Type a valid letter." << endl;
                        }
                    }
                }
            }
            
            else if (userChoice == 3){
                MagicItem golden;
            
                //ask first question, check answer, and award points
                int numGuess;
                
                cout << golden.getMQuestion() << endl;
                cin >> numGuess;
                
                if(golden.checkM(numGuess)){
                    golden.setUserScore(1);
                }else{
                    cout << "incorrect year" << endl;
                    cout << "your score " << golden.getUserScore() << endl;
                }
                
                //ask second question, check answer, and award points
                cin.ignore();
                string candyGuess;
                
                cout << golden.getCandyQ() << endl;
                getline(cin,candyGuess);
                
                if(golden.checkCandy(candyGuess)){
                    golden.setUserScore(2);
                }else{
                    cout << "incorrect candy" << endl;
                    cout << "your score " <<golden.getUserScore() << endl;
                }
                //ask third question, check answer, and award points
                //cin.ignore();
                string personGuess;
                
                cout << golden.getStatueQ() << endl;
                getline(cin,personGuess);
                
                if(golden.checkStatue(personGuess)){
                    golden.setUserScore(5);
                }else{
                    cout << "incorrect person" << endl;
                    cout << "your score " <<golden.getUserScore() << endl;
                }
                //award prize based on user points
                golden.awardPrize(golden);
        
            }
            
            if(userChoice == 4){
                
                JackpotTrivia diamonds; // Defining the variable titled diamonds
	            diamonds.introduction(); // Calling the introduction function using the dot product operator
	            diamonds.trivia1();  // Calling the trivia1 function using the dot product operator
                
            }
        }
    }
    return 0;
}