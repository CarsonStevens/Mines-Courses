#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

int main(){
    
    //defines variables
    int left;
    int right;
    //Prompts for input
    cin >> left;
    cin >> right;
    
    //Does the odd cases
    if(left != right){
        //Sees if left or right is larger
        if (left > right){
            cout << "Odd " << left * 2;
        }
        if (right > left){
            cout << "Odd " << right * 2;
        }
    }
    //Does even case
    else if((left == right) && (left != 0) && (right != 0)){
        cout << "Even " << left + right;
    }
    //Does case where not possible
    else if ((left == 0) && (right == 0)){
        cout << "Not possible";
    }
    
    return 0;
}