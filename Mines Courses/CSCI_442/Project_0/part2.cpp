#include <fstream>
#include <sstream>
#include <iostream>
#include <bitset>
#include <string>

using namespace std;


int main( int argc, char* argv[] ){
    // Load file
    // Declare ifstream object for .txt file parsing.
    // open the file from which to read the data
    ifstream data( argc > 1 ? argv[1] : "/dev/null" );
    if( !data ) {
    	    cerr << "Error opening data file." << endl;
         return(1);
    }

    string answer = "";
    string bits;
    while(!data.eof()){
        data >> bits;
        bitset<8> c(bits);
        
        // DEBUG
        // cout << c << endl;
        // cout << char(c.to_ulong()) << endl;
        
        answer += char(c.to_ulong());
    }
    data.close();
    
    ofstream output( argc > 2 ? argv[2] : "/dev/null" );
    output << answer;
    output.close();
}