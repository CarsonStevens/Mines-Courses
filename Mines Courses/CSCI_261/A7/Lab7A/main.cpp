/* CSCI261 Lab: Money Class
 *
 * Author: Dr. Jeffrey Paone
 *
 * Purpose: give students practice to (1) define classes with
 * public member variables and (2) define and implement constructors.
 *
 * YOU SHOULD NOT MODIFY THIS CODE.
 */

#include "Money.h"


#include <iostream>
using namespace std;

int main() {

    cout << "Who doesn't love money?" << endl << endl;

    Money myMoney;
    cout << "I have $" << myMoney.dollars << ".";
    cout << myMoney.cents << endl;

    Money yourMoney(987, 65); // dollars and cents
    cout << "You have $" << yourMoney.dollars;
    cout << "." << yourMoney.cents << endl;
    cout << endl;

    cout << "I have $" << (myMoney.dollars - yourMoney.dollars) << ".";
    cout << (myMoney.cents - yourMoney.cents) << " more money than you!" << endl;

    return 0;

}
