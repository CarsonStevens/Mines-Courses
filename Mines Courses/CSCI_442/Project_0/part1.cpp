#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;


int main( int argc, char* argv[] ){
    
    // Prints command line arguements for user
    // for(int i = 0; i < argc; i++){
    //     cout << argv[i] << endl;
    // }
    
    
    for(int i = 0; i < argc; i++){
        string arguement = argv[i];
        if (arguement == "-h" || arguement == "--help" || arguement == "-ho" || arguement == "-oh"){
            cout << endl << "Welcome to ./part1's help page" << endl << endl;
            cout << "To run the program, type:\t ./part1 [flags] input_file_name\t into the terminal." << endl << endl;
            cout << "-o, --output" << endl << "Output the sum value into a file named output_part1" << endl << endl << "-h --help";
            cout << endl << "Display a help message about the flags and exit" << endl << endl;
            return(0);
        }
    }
    
    //Load file
    //Declare ifstream object for .txt file parsing.
    //open the file from which to read the data
    ifstream data( argc > (argc-1) ? argv[argc-1] : "/dev/null" );
    if( !data ) {
        cerr << "Error opening input." << endl;
        return(1);
    }
	
	int sum = 0;
	string line;
    while(!data.eof()){
        getline(data,line);
        for (int i = 0; i < line.length(); i++){
            // cout << line[i];
            if(isdigit(line[i])){
                sum += (int)line[i];
            }
            else{
                continue;
            }
        }
    }
    for(int i = 0; i < argc; i++){
        string arguement = argv[i];
        if (arguement == "-o" || arguement == "--output"){
            ofstream output;
            output.open("output_part1.txt");
            output << sum;
            output.close();
            return(0);
        }
    }

    cout << "Sum is:\t" << sum << endl;
    return(0);
}