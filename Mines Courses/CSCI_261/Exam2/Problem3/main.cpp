#include <iostream>
#include <vector>

using namespace std;



bool palindrome(vector<int> vec){
    double count = 0;
    int j = vec.size()-1;
    for(int i = 0; i < vec.size(); ++i){
            // cout << vec.at(i) << " " << vec.at(j) << endl;
            if(vec.at(j) != vec.at(i)){
                return false;
            }
        --j;
    }
    if(vec.empty() == true){
        return false;
    }
    return true;
}

int main(){
    vector<int> vec = {1,2,3,3,2,1};
    palindrome(vec);
}