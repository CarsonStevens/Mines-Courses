/*Lab6B, Working with Data
 *
 *Author: Carson Stevens and Stephanie Holzschuh
 *
 * 
 */
 
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <ios>
#include <iomanip>
#include <ctime>
#include <string>
using namespace std;

int main() {
    
    char replacement;
    
    ifstream secret("secretMessage.txt");
    
    if(secret.fail()){
        cerr << "Error opening  secretMessage.txt file.";
        return 1;
    }
    
    
    ofstream decrypt("decipheredMessage.txt");
    
    if(decrypt.fail()){
        cerr << "Error opening decipheredMessage.txt file.";
        return 1;
    }
    
    
    
    while( !secret.eof()) {
        secret.get(replacement);
            if (replacement == '~'){    
                decrypt << " ";
            }
            else if (replacement == '\n') {
                decrypt << "\n";
            }
            else {
                decrypt << char(replacement  + 1);
            }
        
    }
 
    secret.close();
    decrypt.close();
}    