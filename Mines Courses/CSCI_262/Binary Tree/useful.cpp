#include <iostream.h>
#include <stdlib.h>
#include "useful.h"

void eat_line()
{
	char next;
	
	do 

	cin.get(next);
	while (next != '\n');
}

bool inquire(const char query[])
{
	char answer;

	do
	{
		cout << query << " [Yes or No]" << endl;
		cin >> answer;
		answer = toupper(answer);
		eat_line();
	}
	while ((answer != 'Y') && (answer != 'N'));
	return (answer == 'Y');
}

