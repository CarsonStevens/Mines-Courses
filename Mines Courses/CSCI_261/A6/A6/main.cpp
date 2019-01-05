/*A6, Green Eggs and Ham
 *
 *Author: Carson Stevens
 *
 * Count the number of occurances of each word and add them to a bar graph.
 */
 
#include <fstream>
#include <iostream>
#include <ios>
#include <iomanip>
#include <string>
#include <vector>
#include <SFML/Graphics.hpp>

using namespace std;
using namespace sf;

#include "greeneggsandham.h"

int main() {
    
    vector<WordCount> wordsInText(0);       //Declares the vector that will hold the unique words and their 
                                            //counts.
    
    getFileReady();                         //Reads in the greeneggsandham.txt file, checks for error,
                                            //and creates a new text file without the punctuation.
    
    createVector(wordsInText);              //Reads in the new text file that doesn't have punctuation.
                                            //Then checks if the words is in the vector. If not, adds it and 
                                            //gives it a count of 1. If it is in the vector, increases count
                                            //by one.
    
    alphabetize(wordsInText);               //Alphabetizes the new vector
    
    printResults(wordsInText);              //Prints out the words in alphabetical order and their counts.
                                            //Also prints the most frequent and least frequent words.
                                    
    createSFML(wordsInText);                //Prints out the bar graph of the word frequencies using SFML
    
    return 0;
}