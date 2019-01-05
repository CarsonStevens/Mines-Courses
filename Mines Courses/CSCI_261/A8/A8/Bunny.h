/* CSCI261 Assignment 8: Make A Class
 *
 * Description: Declaration file for Bunny Class
 *
 * Author: Ashley Piccone, Section C
 *
 */

#ifndef BUNNY_H
#define BUNNY_H

#include <vector>
#include <string>
using namespace std;

class Bunny {
    public:
        Bunny();
        void introduction(Bunny bunners); //prints out introduction
        void bunnyHop(); //prints out acii art of bunny 
        void breedBunny(vector<Bunny>& bunnyVec); //adds another bunny to the vector of bunnies
        void feedBunny(); //feeds the bunny to increase its size
        
        bool getFluffy(); //getter functions
        bool getCute();
        int getSize();
        string getName();
        
        void setFluffy(bool fluffiness); //setter functions
        void setCute(bool cuteness);
        void setSize(int bunSize);
        void setName(string bunName);
        
    private:
        bool isFluffy;
        bool isCute;
        int size;
        string name;
};

#endif