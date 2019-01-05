#include <iostream>
#include <stack>
#include <string>

using namespace std;

class DP_Cell{
public:
	int cost;
	int parentCity;

	DP_Cell(int cost, int parentCity){
		this->cost = cost;
		this->parentCity = parentCity;
	}

	DP_Cell(){
		this->cost = -1;
		this->parentCity = -1;
	}
};

int calc_costs(int previous_city, int current_month);
void traceback (int minCost);

const int MAX_CITIES = 3, MAX_MONTHS = 12;

int operating_costs[MAX_CITIES][MAX_MONTHS] = {
	{8, 3, 10, 43, 15, 48, 5, 40, 20, 30, 28, 24},
	{18, 1, 35, 18, 10, 19, 18, 10, 8, 5, 8, 20},
	{40, 5, 8, 13, 21, 12, 4, 27, 25, 10, 5, 15}
};

int relocation_costs[MAX_CITIES][MAX_CITIES] = {
	{0, 20, 15},
	{20, 0, 10},
	{15, 10, 0}
};



DP_Cell dp_matrix[MAX_CITIES][MAX_MONTHS];

string cities[3] = {"NY", "LA", "DEN"};



int main(int argc, char **argv) {

    //will store min cost indexed at [this_month's_city][next_month] (min cost of the future months for the city currently residing in) 
    // intialize cells to -1
     for (int i = 0; i < MAX_CITIES; i++) {
         for (int j = 0; j < MAX_MONTHS; j++) {
			DP_Cell cell(-1, i);
             dp_matrix[i][j] = cell;
         }
     }

    int minCost = calc_costs(-1, 0);
	cout << minCost << endl;
	
	for (int i = 0; i < MAX_CITIES; i++) {
        for (int j = 0; j < MAX_MONTHS; j++) {
        	std::cout << dp_matrix[i][j].cost << ", ";
        }
        std::cout << std::endl;
    }

    traceback(minCost);
}

int calc_costs(int previous_city, int current_month){ 
	int min_month_cost = 2000000000, city_to_move_to = -1, total_cost = 0;

	if(previous_city != -1){
		//if its in the DP matrix
		if (dp_matrix[previous_city][current_month].cost != -1){
			return dp_matrix[previous_city][current_month].cost;
		}
	
		//base case when we are on the last month
		if (current_month == MAX_MONTHS-1){
			for (int city = 0; city < MAX_CITIES; city++){
				int total_cost = operating_costs[city][current_month] + relocation_costs[previous_city][city];
				if (total_cost < min_month_cost){
					city_to_move_to = city;
					min_month_cost = total_cost;
				}
			}
			dp_matrix[previous_city][current_month].cost = min_month_cost;
			dp_matrix[previous_city][current_month].parentCity = city_to_move_to;
			return dp_matrix[previous_city][current_month].cost;
		}
	}

	

	//else, check each city and reccur
	for (int city = 0; city < 3; city++){
		//if just starting algorithm
		if (previous_city == -1){
			total_cost = operating_costs[city][current_month] + calc_costs(city, current_month + 1);
		}
		else {
			total_cost = operating_costs[city][current_month] + relocation_costs[previous_city][city] + calc_costs(city, current_month + 1);
		}
		
		if (total_cost < min_month_cost){
			city_to_move_to = city;
			min_month_cost = total_cost;
		}
	}

	//store answer in D.P Matrix
	dp_matrix[previous_city][current_month].cost = min_month_cost;
	dp_matrix[previous_city][current_month].parentCity = city_to_move_to;
	return min_month_cost;
    
}

void traceback (int minCost) {
	int optimal_city_start;
	for(int i = 0; i < 3; i++){
		if(dp_matrix[i][0].cost == minCost){
			optimal_city_start = dp_matrix[i][0].parentCity;
			break;
		}
	}
	for(int n = 0; n < MAX_MONTHS; n++){
	    cout << cities[dp_matrix[optimal_city_start][n].parentCity] << " ";
	    optimal_city_start = dp_matrix[optimal_city_start][n].parentCity;
	}
    return;
}