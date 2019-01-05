#pragma once

//Checks if the character has already been used.
bool check (char answer, char inPlay[], int numTotal) {
    for(int i = 0; i <= numTotal; ++i) {
        char alreadyUsed = inPlay[i];
        if (answer == alreadyUsed) {
            return false;
        }
    }
    return true;
}

//checks to see if you have the answer correct.
bool answer (char notAnswered[], char chosenWord[], int numTotal){
    int noMatch = 0;
    for(int i = 0; i <= numTotal; ++i) {
        if (notAnswered[i] != chosenWord[i]){
            // @DEBUG
                //cout << "not: " << notAnswered[i] << "\t chosen: " << chosenWord[i] << endl;
            ++noMatch;
        }
    }
    if (noMatch > 0) {
        return false;
    }
    else {
        return true;
    }
}

//prints the blank array
void answerKey(char answer[], int length) {
    
    for (int i = 0; i < length; ++i) {
        std::cout << answer[i] << " ";
    }
    return;
}