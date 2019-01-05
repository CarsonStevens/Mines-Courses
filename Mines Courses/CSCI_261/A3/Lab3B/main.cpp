/* CSCI 261: Fix Loop Errors
 *
 * Author: Carson Stevens and Stephanie Holzschuh
 *
 *    This program illustrates a variety of common loop errors.
 *    Fix the errors in each section.

Copyright 2017 Dr. Jeffrey Paone

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

 */

#include <iostream>
using namespace std;

int main() {
    cout << "Welcome to Loop World" << endl;

// SECTION I: update comment below on how you fixed this section's code, and tests run
// that show loop is working properly
// FIX = The addition of an equal sign between "<" and "5" on line 51
// TESTS: Inclusive addition is equal to 15, which is correct

    cout << endl;
    cout << "******************" << endl;
    cout << "Section I" << endl;
    cout << "******************" << endl;

    short sum = 0;  // Accumulates the total
    short i;    // Used as loop control variable
    for (i = 1; i <= 5; ++i) {
         sum += i;
     }
    cout << "The sum of the numbers from 1 to 5 (inclusive) is: " << sum << endl;

// SECTION II: update comment below on how you fixed this section's code, and tests run
// that show loop is working properly
// FIX = The total was being updated to 0 each time that it was iterated. We took out that part and intitalized total to 0 above.
// TESTS: A number of items was chosen and given prices. The prices added up to what the program said they should.

    cout << endl;
	cout << "******************" << endl;
	cout << "Section II" << endl;
	cout << "******************" << endl;

    double total = 0;     // Accumulates total
    double price;    // Gets next price from user
    short num_items;     // Number of items
    short counter = 1;  // Loop control counter

    cout << "How many items do you have? ";
    cin >> num_items;
    cout << endl;

    while (counter <= num_items) {
        cout << "Enter the price of item " << counter << ": ";
        cin >> price;
        cout << endl;
        total += price;
        counter++;
    }
    cout << "The total price is: " << total << endl;

// SECTION III: update comment below on how you fixed this section's code, and tests run
// that show loop is working properly
// FIX = The program would go until the counter equaled the sum, but we didn't want that and the counter was never incrimenting
//       Thus counter++ incrimented the counter and changing the condition to <= 4 satifies the prompt of inclusion.
// TESTS: When run the prgram prints the sum as 10 and 1+2+3+4=10.

    cout << endl;
	cout << "******************" << endl;
	cout << "Section III" << endl;
	cout << "******************" << endl;

    cout << "I will now calculate ";
    cout << "the sum of numbers from 1 to 4 (inclusive)" << endl;

    sum = 0;
    counter = 1;

    do {
        sum += counter;
        cout << "Sum so far: " << sum << endl;
        counter++;
    } while (counter <= 4);

    cout << endl << "Section III Recap" << endl;

    cout << "I calculated the sum of numbers from 1 to 4 (inclusive) as " << sum << endl;


// SECTION IV: update comment below on how you fixed this section's code, and tests run
// that show loop is working properly
// FIX : The counter started at 4 instead of 1 and the end condition was i>0 and not i <=4
// TESTS: This is correct because 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30 which is what the program prints.

    cout << endl;
	cout << "******************" << endl;
	cout << "Section IV" << endl;
	cout << "******************" << endl;

    cout << "I will now calculate ";
    cout << "the sum of squares from 1 to 4 (inclusive)" << endl;

    sum = 0;
    for (i = 1; i <= 4; i++) {
        sum += i*i;
    }

    cout << "The sum of squares from 1 to 4 is: " << sum << endl;

// SECTION V: update comment below on how you fixed this section's code, and tests run
// that show loop is working properly
// FIX = Counter was incrimenting to 10 instead of 4. And the program wasn't incrimenting the counter after each loop.
// TESTS: This is correct because 1^3 + 2^3 + 3^3 + 4^3 = 1 + 8 + 27 + 64 = 100 which is what the program prints.

    cout << endl;
	cout << "******************" << endl;
	cout << "Section V" << endl;
	cout << "******************" << endl;

    cout << "I will now calculate ";
    cout << "the sum of cubes from 1 to 4 (inclusive)" << endl;

    sum = 0;
    counter = 1;

    while (counter <= 4) {
        sum += (counter * counter * counter);
        counter++;
    }

    counter++;

    cout << "The sum of cubes from 1 to 4 is: " << sum << endl;

    cout << endl;
	cout << "******************" << endl;
	cout << "Section Done" << endl;
	cout << "******************" << endl;

	cout << endl << "Congrats!  You fixed them all (hopefully correctly!)" << endl << endl << "Goodbye" << endl << endl;

    return 0;
}
