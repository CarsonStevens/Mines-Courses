/* CSCI261 Assignment 8: Bunny Class
 *
 * Description: Definition file for Bunny Class
 *
 * Author: Ashley Piccone, Section C
 *
 */

#include "Bunny.h"
#include <iostream>
#include <cmath>
#include <string>
#include <vector>

using namespace std;

Bunny::Bunny() { //default case
    isFluffy = true;
    //name = "BunBun";
    isCute = true;
    size = 1;
}

void Bunny::introduction(Bunny bunners) { //lets user see the attributes of their initial bunny
    cout << "First things first, name your bunny!" << endl;
    string bunnyName;
    cin >> bunnyName;
    bunners.setName(bunnyName);
    cout << "Your bunny's name is: " << bunners.getName() << endl;
    cout << "You are starting with one bunny. It has the following qualities: " << endl;
    if (bunners.getFluffy() == true) {
        cout << "Your bunny is fluffy." << endl; 
    }
    else {
        cout << "Your bunny is not fluffy." << endl;
    }
    if (bunners.getCute() == true) {
        cout << "Your bunny is quite cute." << endl;
    }
    else {
        cout << "Your bunny is not cute." << endl;
    }
    bunners.getSize();
    cout << "You have three options. You can make your bunny hop (type h), feed him/her (type f), or breed him/her (type b). You may type q to quit." << endl;
}

void Bunny::bunnyHop() { //bunny hopping!!
    if (size <= 5) {
        cout << "             " << "           " << "        _   _" << endl;
        cout << "             " << "           " << "       | | | |" << endl;
        cout << "             " << "           " << "  ____(^  .  ^)" << endl;
        cout << "        _   _" << "           " << "O(_____  ___)  " << "         " << "          _   _" << endl;
        cout << "       | | | |" << "          " << " (___)  (___) " <<  "         " << "          | | | |" << endl;
        cout << "  ____(^  .  ^)" << "                                  " << "   ____(^  .  ^)" << endl;
        cout << "O(_____  ___)  " << "                                  " << " O(_____  ___)  " << endl;
        cout << " (___)  (___) " << "                                   " << "  (___)  (___) " << endl;
        cout << endl << "Your bunny hopped!" << endl;
    }
    else {
        cout << "Your bunny is too overweight to hop." << endl;
    }
}

void Bunny::breedBunny(vector<Bunny>& bunnyVec) { //adds another baby bunny to the bunny vector
    Bunny newBun;
    string newBunnyName;
    cout << "Enter the baby bunny's name: " << endl;
    cin >> newBunnyName;
    newBun.setName(newBunnyName);
    cout << "Is the bunny fluffy? (Y/N)" << endl;
    char fluff;
    cin >> fluff;
    if (fluff == 'Y') {
        newBun.setFluffy(true);
        cout << "The new bunny is fluffy." << endl;
    }
    else {
        newBun.setFluffy(false);
        cout << "The new bunny is not fluffy." << endl;
    }
    cout << "Is the new bunny cute?" << endl;
    cout << "DUH" << endl;
    newBun.getSize();
    bunnyVec.push_back(newBun);
    cout << "Now you have " << bunnyVec.size() << " bunnies! They are all adorable!" << endl;
}

void Bunny::feedBunny() { //feeds the bunny and increases its size
    size += 1;
    cout << "Your bunny grew! ";
}

bool Bunny::getFluffy() {
    return isFluffy;
}

bool Bunny::getCute() {
    return isCute;
}

int Bunny::getSize() {
    if (size == 1) {
        cout << "Your bunny is small." << endl;
    }
    else if (size == 2 || size == 3) {
        cout << "Your bunny is medium-sized." << endl;
    }
    else if (size == 4 || size == 5) {
        cout << "Your bunny is large." << endl;
    }
    else {
        cout << "Your bunny is obese. Take him/her to a vet, you awful pet owner." << endl;
    }
    return size;
}

string Bunny::getName() {
    return name;
}

void Bunny::setFluffy(bool fluffiness) {
    isFluffy = fluffiness;
}

void Bunny::setCute(bool cuteness) {
    isCute = cuteness;
}

void Bunny::setSize(int bunSize) {
    size = bunSize;
}

void Bunny::setName(string bunName) {
    name = bunName;
}


