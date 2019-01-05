/* CSCI261 Section E Assignment XC
 *
 * Description: People Validator
 *
 * Author: Carson Stevens
 *
 */

#include <iostream>
#include <string>
#include <fstream>
using namespace std;

class Person{
    public:
        Person();
        void readData(ifstream, int *countPt);
        //void writeData(ofstream, string f, string l, char g, int a, double h, bool m, bool c);
        bool validate(string f, string l, char g, int a, double h, bool m, bool c);
        
    private:
        string _first;
        string _last;
        char _gender;
        int _age;
        double _height;
        bool _monsterLover;
        bool _cantaloupeLover;
    
};

Person::Person(){
    _first = "JimBob";
    _last = "Stevens";
    _gender = 'F';
    _age = 18;
    _height = 7.5;
    _monsterLover = true;
    _cantaloupeLover = true;
}

void Person::readData(ifstream peopleData, int *countPt){
    while(!peopleData.eof()){
        peopleData >> _first;
        peopleData >> _last;
        peopleData >> _gender;
        peopleData >> _age;
        peopleData >> _height;
        peopleData >> _monsterLover;
        peopleData >> _cantaloupeLover;
        if(validate(_first, _last, _gender, _age, _height, _monsterLover, _cantaloupeLover) == true){
           countPt++;
        }
    }
    
    return;
}

// void Person::writeData(){
    
//     return;   
// }

bool Person::validate(string f, string l, char g, int a, double h, bool m, bool c){
    if(g != 'M' || g != 'F'){
        return false;
    }
    if((a < 18) || (a > 40)){
        return false;
    }
    if(h>7.5){
        return false;
    }
    if((m != 0) || (m != 1)){
        return false;
    }
    if((c != 0) || (c != 1)){
        return false;
    }
    return true;
}


int main(){
    
    int validCount;
    int *countPt;
   
    countPt = &validCount;
    Person person;
    
    ifstream peopleData("PersonFile.dat"); //Declares the new file as an input.
    
    //Checks to make sure the new text file can be opened.
    if(peopleData.fail()){
        cerr << "Could not read PersonFile.dat file.";
        return -1;
    }
    
    person.readData(peopleData, &validCount);
    
    cout << *countPt;
    
    return 0;
}