/* CSCI261 Section E Assignment XCLab
 *
 * Description: Pointer Instructions
 *
 * Author: Carson Stevens & Stephanie Holzschuh
 *
 */

#include <iostream>
using namespace std;

int main(){
    
    //#1
    int iNum = 9;
    //#2
    double dNum = 14.25;
    //#3
    int* iPtr1;
    int* iPtr2;
    //#4
    double* dPtr;
    
    //#5
    iPtr1 = &iNum;
    //#6
    cout << "The address of iNum: " << iPtr1 << endl;
    cout << "\tDouble check address: " << addressof(iNum) << endl;
    //#7
    cout << "The contents of iNum using iPtr1 is: " << *iPtr1 << endl;
    //#8
    //iPtr1 = &dNum;
        //cannot convert ‘double*’ to ‘int*’ in assignment
    //#9
    dPtr = &dNum;
    //#10
    cout << "Contents of dNum is: " << *dPtr << endl;
    //#11
    iNum = 7;
    //#12
    cout << "Contents of iNum is: " << *iPtr1 << endl;
    //#13
    iPtr2 = iPtr1;
    //#14
    cout << "Address of iPtr2 is: " << iPtr2 << endl;
    //#15
    cout << "The value pointed to by iPtr2 is: " << *iPtr2 << endl;
    //#16
    iNum = 5 + *iPtr1;
    //#17
    cout << "iNum using iPtr1 to output contents is: " << *iPtr1 << endl;
    cout << "iNum using iPtr2 to output contents is: " << *iPtr2 << endl;
    cout << "iNum using iNum is: " << iNum << endl;
    
    
    
    return 0;
}