#include <iostream>
#include <ctime>
#include <ios>
#include <iomanip>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;
#include "point.h"
#include <algorithm>

using namespace std;

void PrintRoute(vector<Point> route){
    for(auto i : route){
        i.print();
        cout << " -> ";
    }
}


void Permute(Point start, vector<Point> points, int size, double n, vector<vector<Point>> &permutations){
    if(size == 1 && (points.at(0) == start)){
        points.push_back(start);
        permutations.push_back(points);
        //PrintRoute(points);
        //cout << endl;
        return;
    }
    for(int i = 0; i < size; i++){
        Permute(start, points, size-1, n, permutations);
        if(size%2 == 1){
            swap(points.at(0),points.at(points.size() - 1));
        }
        else{
            swap(points.at(i), points.at(points.size() - 1));
        }
    }
}



int ExhaustiveSeach(vector<vector<Point>> pointsExhaustive){
    int index;
    int shortest = 999999999;
    for(int i = 0; i < pointsExhaustive.size(); i++){
        double distance = 0;
        for(int j = 0; j < pointsExhaustive.at(i).size()-1; j++){
            distance += pointsExhaustive.at(i).at(j).getDistance(pointsExhaustive.at(i).at(j+1));
        }
        if(distance < shortest){
            shortest = distance;
            index = i;
        }
    }
    return index;
    
}


Point NearestNeighbor(Point p, vector<Point> &pointsNeighbor, vector<Point> &neighborRoute){
    double distance = 999999999;
    Point closest;
    int index;
    for(int i = 0; i < pointsNeighbor.size(); i++){
        if(p.getDistance(pointsNeighbor.at(i)) < distance){
            distance = p.getDistance(pointsNeighbor.at(i));
            closest = pointsNeighbor.at(i);
            index = i;
        }
    }
    pointsNeighbor.erase(pointsNeighbor.begin() + index);
    neighborRoute.push_back(closest);
    return closest;
}

int main(){
    
//Load file
    //Declare ifstream object for .txt file parsing.
    ifstream data;
    //open the file from which to read the data
    data.open("TSP.txt");
    // if the file is empty or won't load, it outputs the message
    if (!data){
        cout << "TSP.txt is empty or won't load" << endl;  
        return 1;
    } 
    
    double counter = 0;
    double n;
    double x;
    double y;
    string line;
    
    vector<Point> pointsNeighbor;
    vector<Point> neighborRoute;
    vector<Point> pointsExhaustive;
    vector<vector<Point>> exhaustiveRoute;
    
    while(!data.eof()){
        getline(data,line);
        istringstream point(line);
        if( counter == 0){
            point >> n;
            counter++;
        }
        else {
            point >> x;
            point >> y;
            Point currentPoint(x,y);
            pointsNeighbor.push_back(currentPoint);
            pointsExhaustive.push_back(currentPoint);
        }
    }
    data.close();

    cout << "Testing NearestNeighbor:" << endl;
    auto t1 = Clock::now();
    
    Point start = pointsNeighbor.at(0);
    Point currentPoint = pointsNeighbor.at(0);
    pointsNeighbor.erase(pointsNeighbor.begin() + 0);
    neighborRoute.push_back(start);
    for( int i = 0; i < n; i++){
        currentPoint = NearestNeighbor(currentPoint, pointsNeighbor, neighborRoute);
        // Point neighbor = neighborRoute.at(neighborRoute.size()-1);
        // currentPoint = neighbor;
    }
    //neighborRoute.insert(neighborRoute.begin(), start);
    neighborRoute.pop_back();
    neighborRoute.push_back(start);
    double distance = 0;
    for(int i = 0; i < neighborRoute.size()-1; i++){
        distance += neighborRoute.at(i).getDistance(neighborRoute.at(i+1));
    }
    
    auto t2 = Clock::now();
    cout << "\tDistance: " << setprecision(9) << distance << endl;
    cout << "\tDuration: " << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() << " nanoseconds" << endl;
    cout << "\tRoute: " << endl << "\t\t";
    PrintRoute(neighborRoute);
    cout << endl << endl;
    
    cout << "Testing Exhaustive:" << endl;
    auto t3 = Clock::now();
    Permute(start, pointsExhaustive, pointsExhaustive.size(), pointsExhaustive.size(), exhaustiveRoute);
    cout << "SIZE: " << exhaustiveRoute.size()<< endl << endl;
    int index = ExhaustiveSeach(exhaustiveRoute);
    distance = 0;
    for(int i = 0; i < exhaustiveRoute.at(index).size()-1; i++){
        distance += exhaustiveRoute.at(index).at(i).getDistance(exhaustiveRoute.at(index).at(i+1));
    }
    auto t4 = Clock::now();
    
    cout << "\tDistance: " << setprecision(9) << distance << endl;
    cout << "\tDuration: " << chrono::duration_cast<chrono::nanoseconds>(t4 - t3).count() << " nanoseconds" << endl;
    cout << "\tRoute: " << endl << "\t\t";
    PrintRoute(exhaustiveRoute.at(index));
    cout << endl << endl;
    
    
}