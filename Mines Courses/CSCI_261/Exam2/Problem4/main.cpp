#include <iostream>
#include <vector>

using namespace std;

class Board{
    public:
        Board();
        Board(int h, int l, int w, int d);
        int getH();
        int getL();
        int getW();
        int getD();
        void set(int h, int l, int w, int d);
        int volume();
        int weight();
    private:
        int _height;
        int _length;
        int _width;
        int _density;
};

Board::Board(){
    _height = 1;
    _length = 1;
    _width = 1;
    _density = 1;
}

//Parameterized constructor
Board::Board(int h, int l, int w, int d){
    _height = h;
    _length = l;
    _width = w;
    _density = d;
}

//Gets all the parameters
int Board::getH(){
    return _height;
}

int Board::getL(){
    return _length;
}

int Board::getW(){
    return _width;
}

int Board::getD(){
    return _density;
}

//Sets all the properties
void Board::set(int h, int l, int w, int d){
    if(h > 0){
        _height = h;
    }
    if(l > 0){
        _length = l;
    }
    if(w > 0){
        _width = w;
    }
    if(d > 0){
        _density = d;
    }
    return;
}

//Returns volume
int Board::volume(){
    return _height*_length*_width;
}

//Returns weight
int Board::weight(){
    return _height*_length*_width*_density;
}

int main(){
    //Defines variables
    int num;
    int height;
    int width;
    int length;
    int density;
    //Gets number of boards
    cin >> num;
    //creates a vector with all the boards
    vector<Board> allBoards;
    //Sets size of vector
    allBoards.resize(num);
    for(int i = 0; i < num; ++i){
        //Gets input properties
        cin >> height;
        cin >> width;
        cin >> length;
        cin >> density;
        //Sets properties
        allBoards.at(i).set(height, length, width, density);
    }
    //Gets max load weight
    int max;
    cin >> max;
    //Defines total weight and trips
    int total;
    int trips;
    
    //Adds vector component weights for total;
    for(int j = 0; j < allBoards.size(); j++){
        total = total + allBoards.at(j).weight();
    }
    
    //Determines if an extra trip should be made
    if(total/max == 0){
        trips = total/max;
    }
    else{
        trips = (total/max) + 1;
    }
    
    //Prints results
    if(trips > 0){
        cout << trips;
    }
    else{
        cout << "Not Possible";
    }
}