#include "greeneggsandham.h"


//Checks to see if the word is in the Vector already of not
int findWordInVector( string& tempWord, vector<WordCount> foundWords ){
    for(int i = 0; i < foundWords.size(); i++){
        if(tempWord == foundWords.at(i).word){
            return i;
        }
    }
    return -1;
    
}

//Returns the index of the most frequent word.
int findMaxFrequency(vector<WordCount> foundWords){
    int maxIndex = -1;
    int maxNum = -1;
    for(int i = 0; i < foundWords.size(); i++){
        if(foundWords.at(i).count > maxNum){
            maxNum = foundWords.at(i).count;
            maxIndex = i;
        }
    }
    return maxIndex;
    
}



//Preps the files of text to be read and put into the vector.
int getFileReady(){
    ifstream text("greeneggsandham.txt"); //Declares text being inputed
    
    //Checks if text file fails to open
    if(text.fail()){
        cerr << "Could not read 'greeneggsandham.txt' file.";
        return 1;
    }
    
    ofstream textWithOutPunc("textWithOutPunc.txt");// Declares new text file used for the output of removing the punctuation
    
    
    //Checks if the new text file fails to open
    if(textWithOutPunc.fail()){
        cerr << "Could not read textWithOutPunc.txt' file.";
        return 1;
    }

    char c; //going to be used with get to remove punctuation.
    //Removes punctuation from the text and stores the chars in the new text file with no punctuation
    while( !text.eof()) {
        text.get(c);
            if((c == '.') || (c == '?') || (c == '!') || (c == ',') || (c == '\n')){
                textWithOutPunc << " ";
            }
            else {
                textWithOutPunc << c;
            }
    }
    
    //Closes the files after they were either read from or created.
    text.close();
    textWithOutPunc.close();
    
    return 0;
}

//Puts the strings in alphabetical order.
void alphabetize(vector<WordCount>& uniqueWords){
    for(int j = 0; j < uniqueWords.size(); j++){
        for(int k = 0; k < uniqueWords.size(); k++){
            if(uniqueWords.at(j).word < uniqueWords.at(k).word){
                swap(uniqueWords.at(j).word, uniqueWords.at(k).word);
                swap(uniqueWords.at(j).count, uniqueWords.at(k).count);
            }
            else{
                continue;
            }
        }
    }
    return;
}

//Creates the vector that will contain all the unique words
int createVector(vector<WordCount>& uniqueWords){
    
    ifstream noPuncText("textWithOutPunc.txt"); //Declares the new file as an input.
    
    //Checks to make sure the new text file can still be opened after the edit as an input.
    if(noPuncText.fail()){
        cerr << "Could not read new.txt' file.";
        return 1;
    }
    
    
    
    WordCount tempStorage;  //The words to be checked and added for either count or the word and count.
    
    int counter = 0;    //Used to initialize the first element of the uniqueWords vector.
    while(!noPuncText.eof()){
        noPuncText >> tempStorage.word;     // Reads from the file each string.
        
        //Initializes the vector with the first word
        if(counter == 0){
            tempStorage.count = 0;
            uniqueWords.push_back(tempStorage);
            ++counter;
        }
        
        int checkIndex = findWordInVector(tempStorage.word, uniqueWords ) ;
        
        //Checks the vector for the rest of the strings in the file.
        if(counter != 0){
            if( checkIndex >= 0 ) {
            // increment the counter for the WordCount value at checkIndex
                uniqueWords.at(checkIndex).count++;
            }
            else if (checkIndex < 0) {
            // set the new WordCount variable to contain word and a count of 1
            // push the new WordCount variable into vectorWordCount
                tempStorage.count = 1;
                uniqueWords.push_back(tempStorage);

            }
        }    
    }
    
    //Closes the text file when done reading the strings from it.
    noPuncText.close();
    
    return 0;
}

//Prints the results of unique words, frequencies, the least frequent ones, and the most frequent one.
void printResults(vector<WordCount>& uniqueWords){
    
    //Prints all the unique words and their counts
    for (int i = 0; i < uniqueWords.size(); i++){
        cout <<  "#" << setw(3) << left << i+1 <<  " " << setw(9) << left << uniqueWords.at(i).word << ": " << uniqueWords.at(i).count << endl;
    }
    
    //Prints the least frequent words and their frequency.
    //Stated on Piazza that since we know the least frequent is 1, we can use this since there are multiple words that occur only once.
    //Noted that if the least frequent word still appeared 2 or more times that this method would not work.
    for (int i = 0; i < uniqueWords.size(); i++){
        if (uniqueWords.at(i).count == 1){
            cout << "Least frequent word: " << uniqueWords.at(i).word << endl << "\t" << "Count: " << uniqueWords.at(i).count << endl;
        }
    }
    
    //Prints the most frequent word and its frequency.
    cout << "Most frequent word: " << uniqueWords.at(findMaxFrequency(uniqueWords)).word << endl << "\t" << "Count: " << uniqueWords.at(findMaxFrequency(uniqueWords)).count << endl;
    return;
}


//Creates the bar graph showing frequency
void createSFML(vector<WordCount> uniqueWords){
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    
// Beginning of SFML
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //Identifying the max and min frequencies.
    int maxCount = uniqueWords.at(findMaxFrequency(uniqueWords)).count;
    int minCount = 1;
    
    //Creates the window
    RenderWindow window( VideoMode( 640, 480), "SFML Window" );
    
    //While the window is open, print out the bars of the bar graph.
    while(window.isOpen()){
        
        //Sets background color of window
        window.clear( Color::Black );
        const double width = 640.0/uniqueWords.size(); //Sets the width of each bar
        double heightScaling = 480.0/(maxCount); //Scales the height of the bar to the count and window height
        



        //iterates the number of unique words
        for(int i = 0; i < uniqueWords.size(); ++i){
            
            RectangleShape bar; //Declares the shape of the bars
            int height = uniqueWords.at(i).count;   //Sets the height to the frequency of the word
            
            //Creates the x and y Position for each bar.
            double xPos = i * width;
            double yPos = 480.0 - uniqueWords.at(i).count/ maxCount * 480.0;
            
            //Sets the size and position of the bars
            bar.setSize(Vector2f(width, -(height)*heightScaling));
            bar.setPosition(Vector2f(xPos, yPos));
            
            
            //Sets the color of each bar
            if (uniqueWords.at(i).count == minCount){
                bar.setFillColor(Color::Red);
            }
            else if(uniqueWords.at(i).count == maxCount){
                bar.setSize(Vector2f(width, (height)*heightScaling));
                bar.setFillColor(Color::Yellow);
            }
            else{
                bar.setFillColor(Color::Green);
            }
            
            //Draws the bar on the window
            window.draw(bar);
        }
        
        //Allows the user to close the window.
        Event event;
        while( window.pollEvent( event) ){
            if(event.type == Event::Closed){
                window.close();
            }
        }

        //Displays the window
        window.display();
    }
}