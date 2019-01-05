#include <fstream>
#include <iostream>
#include <ios>
#include <iomanip>
#include <string>
#include <vector>
#include <SFML/Graphics.hpp>
using namespace std;
using namespace sf;

#pragma once


//Declares the struct used to store unique words and their frequencies
struct WordCount {
    int count;
    string word;
};

int findWordInVector( string& tempWord, vector<WordCount> foundWords );

int findMaxFrequency(vector<WordCount> foundWords);

int getFileReady();

void alphabetize(vector<WordCount>& uniqueWords);

int createVector(vector<WordCount>& uniqueWords);

void printResults(vector<WordCount>& uniqueWords);

void createSFML(vector<WordCount> uniqueWords);