/* CSCI 261 Assignment 1: Hello World and ASCII Art
 *
 * Author: Carson Stevens
 *
 *Description: The goal was to make a smiley face using ASCII art and utilize user input to state facts.
 */

#include <iostream>
using namespace std;

int main() {
    
    cout << "Part 1" << endl << endl;
    
    //ASCII Smiley Face
    
    cout << "   ######" << endl << "  #      #" << endl << " #   * *  #    \\1/" << endl;
    cout << "#    L     #    O-" << endl << " #  \\__/  #    /" << endl << "   ######-----" << endl << endl;
    
    
//Part 2
    
    cout << "Part 2" << endl << endl;
    
    //Variables being used.
    
    int userAge = 19;
    int carSpeed = 60;
    float carTime = 5.2;
    int numCorndogs = 8;
    
    // Output of variables in a statement.
    
    cout << "Hello World!" << endl;
    cout <<"I am " << userAge << " years old." << endl;
    cout << "My car goes " << carSpeed << " miles per hour in " << carTime << " seconds." << endl;
    cout << "Yesterday, I ate " << numCorndogs << " corndogs." << endl << "Goodbye!" << endl;
    
    return 0;
}