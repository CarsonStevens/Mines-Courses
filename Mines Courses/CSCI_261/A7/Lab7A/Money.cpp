/* CSCI261 Lab: Money Class
 *
 * Description: Definition file for Money Class
 *
 * Author: Carson Stevens and Stephanie Holzschuh
 *
 */
#include <iostream>
using namespace std;
#include "Money.h"


Money::Money(){
    dollars = 999;
    cents = 99;
    
}

Money::Money(int d, int c){
    dollars = d;
    cents = c;
}