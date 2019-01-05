/* CSCI261 Lab: Money Class
 *
 * Description: Declaration file for Money Class
 *
 * Author: Carson Stevens and Stephanie Holzschuh
 *
 */

#pragma once

class Money {
    
    // default     
    public:
        Money(); // The prototype of the constructor function
        Money(int d, int c); // Non-default constructor
        int dollars;
        int cents;
    
};