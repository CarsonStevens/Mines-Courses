/*
 * File:                main.cpp
 *
 * Author:              Carson Stevens
 *
 * Course & Assignment: CSCI_477 Assignment 6
 *
 * Description:         Implement Dijkstra's Algorithm
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <algorithm>

using namespace std;

struct Edge {
    string destination;
    int cost;

    //Constructor
    Edge(string destination, int cost){
        this->destination = destination;
        this->cost = cost;
    }
    
};


struct compare{  
   bool operator()(const pair<string, int>& l, const pair<string, int>& r) {  
       return l.second > r.second;  
   }  
};  


class Graph {
public:
    map<string, vector<Edge> > adjList;
    
    //Constructor
    Graph() {

    }

    //Clear the Graph
    void makeEmpty() {
        adjList.clear();
        cout << "Graph cleared." << endl << endl;
        return;
    }

    //Add Vertex to the graph
    void addVertex(string vertex_name) {
        //New vector to store adj vertexes and edges
        vector<Edge> new_adj_list;
        //If vertex that is already in map is added again, will delete old data
        if(adjList.count(vertex_name) == 1){
            cout << "Vertex already in graph." << endl;
        }
        else{
            adjList[vertex_name] = new_adj_list;
            cout << "Vertex " << vertex_name << " added to the graph." << endl;
        }
    }

    bool addEdge(string fromVertex, string toVertex, int cost) {
        Edge edge(toVertex, cost);

        if(adjList.count(fromVertex) == 1 && adjList.count(toVertex) == 1){
            adjList[fromVertex].push_back(edge);
            cout << "Edge from " << fromVertex << " to vertex " << toVertex << " added to the graph." << endl;
            return true;
        }
        if(adjList.count(fromVertex) == 0){
            cout << "Vertex '" << fromVertex << "' doesn't exist." << endl;
        }
        if(adjList.count(toVertex) == 0){
            cout << "Vertex '" << toVertex << "' doesn't exist." << endl;
        }
        return false;
    }

    //Return the weight of the edge
    int getWeight(string fromVertex, string toVertex) {
        if(adjList.count(fromVertex) == 1 && adjList.count(toVertex) == 1){
            vector<Edge> temp = adjList[fromVertex];
            for(Edge edge : temp){
                if(edge.destination == toVertex){
                    return edge.cost;
                }
            }
        }
        if(adjList.count(fromVertex) == 0){
            cout << "Vertex '" << fromVertex << "' doesn't exist." << endl;
        }
        if(adjList.count(toVertex) == 0){
            cout << "Vertex '" << toVertex << "' doesn't exist." << endl;
        }
        return -1;
    }

    bool getAdjacent(string fromVertex, vector<string> &vVertex) {
        if(adjList.count(fromVertex) == 1){
            for(auto edge : adjList[fromVertex]){
                vVertex.push_back(edge.destination);
            }
            return true;
        }
        cout << "Vertex '" << fromVertex << "' doesn't exist." << endl;
        return false;
    }

    void printGraph(){
        for(auto vertex : adjList){
            for(Edge edge : vertex.second){
                cout << "(" << vertex.first << " -> " << edge.destination << ", " << edge.cost << ")" << endl;
            }
            cout << endl;
        }
    }
    
    int getSize(){
        return adjList.size();
    }
    
    
    int dijkstra(string fromVertex, string toVertex, vector<string> &vVertex) { 
        
        //Make sure the vertices exist
        if(adjList.count(fromVertex) == 0){
            cout << "Vertex '" << fromVertex << "' doesn't exist." << endl;
            return -1;
        }
        if(adjList.count(toVertex) == 0){
            cout << "Vertex '" << toVertex << "' doesn't exist." << endl;
            return -1;
        }
        
        //For accessing adj nodes
        vector<string> *temp = new vector<string>;
        
        //table to keep track
        map<string, pair<string,int>> table;
        //Insert start into table
        table.emplace(fromVertex, pair<string,int>(fromVertex, 0));
        set<string>visited;
        priority_queue<pair<string, int> ,vector<pair<string, int>>, compare> pq;
        
        //insert first path vertex
        pq.push(pair<string,int>(fromVertex, 0));
        
        while(!pq.empty()){
            temp->clear();
            //Find min cost in unvisited set and set to current node
            pair<string, int> current_vertex = pq.top();
            pq.pop();
            
            //Only process if not in visited list
            if(visited.count(current_vertex.first) == 0){
                //If not in list, add to list
                visited.insert(current_vertex.first);
                
                //Process all new edges
                temp->clear();
                getAdjacent(current_vertex.first, *temp);
                for(int i = 0; i < temp->size(); i++){
                    //Add all paths not in visited into the pq
                    if(visited.count(temp->at(i)) == 0){

                        //Update table
                        //Not in table
                        if(table.count(temp->at(i))==0){
                            int total_cost = getWeight(current_vertex.first, temp->at(i));
                            total_cost += table[current_vertex.first].second;
                            //Add to table
                            pair<string,int> path(current_vertex.first, total_cost);
                            table.emplace(temp->at(i), path);
                            
                            //Add new neighbors to pq; previous node(current_vertex) and new cost
                            pair<string, int> path2(temp->at(i), total_cost);
                            pq.push(path2);
                        }
                        //In table
                        else{
                            int new_cost = getWeight(current_vertex.first, temp->at(i));
                            new_cost += table[current_vertex.first].second;
                            //Check if path cost is better
                            pair<string,int> new_path = table[temp->at(i)];
                            if(new_cost < new_path.second){
                                //Update table with the node it came from (current_vertex) and the new cost
                                pair<string,int> path (current_vertex.first, new_cost);
                                table[temp->at(i)] = path;
                                
                                //Add back to pq
                                pair<string,int> path2(temp->at(i), table[temp->at(i)].second);
                                pq.push(path2);
                            }
                        }
                    }
                }
            }
        }
        string backtrace = toVertex;
        set<string> check;
    
        vVertex.push_back(backtrace);
        while(backtrace != fromVertex){
            if(check.count(backtrace) == 1){
                cout << "No path from " << fromVertex << " to " << toVertex << "." << endl << endl;
                vVertex.clear();
                return -1;
            }
            else{
                check.insert(backtrace);
            }
            
            if(table.count(backtrace) == 0){
                cout << "No path from " << fromVertex << " to " << toVertex << "." << endl << endl;
                vVertex.clear();
                return -1;
            }
            else{
                backtrace = table[backtrace].first;
                vVertex.push_back(backtrace);
            }
            
        }
        reverse(vVertex.begin(), vVertex.end());
        return table[toVertex].second;
        
        //Steps:
            //If not visited
                //in pq (in map)
                    //Update pq and table
                //not in pq (not in map)
                    //add to pq with total cost to get to it, update table (new addition)
                    //if not in table, not in pq
            //else ignore
    }
    
};

int main(){

    int testing = true;
    string command;
    Graph graph;
    
    cout << ">>> GRAPH DRIVER <<<" << endl;
    cout << "\tCOMMANDS:" << endl;
    cout << "\t\tINIT: Reset the graph." << endl;
    cout << "\t\tADDV: Requests an input, then adds the vertex to the graph." << endl;
    cout << "\t\tADDE: Requests two input vertices and a weight, then adds the edge to the graph." << endl;
    cout << "\t\tPRINT: Prints the graph as (from_Vertex, to_Vertex, weight) spaced by source." << endl;
    cout << "\t\tTEST1: Clears the graph and then creates an example flight cost graph." << endl;
    cout << "\t\tTEST2: Clears the graph and then creates and example game map." << endl;
    cout << "\t\tPATH: Requests tow input vertices and gives the shortest path between the two." << endl;
    cout << "\t\tACCURACY: Runs a test to test the accuracy of TEST1's created graph." << endl;
    cout << "\t\tOPTIONS: Prints the input command options again." << endl;
    cout << "\t\tQUIT: Exits the program." << endl;
    while(testing){

        cout << "Input Graphing Command:\t";
        cin >> command;

        //INIT - Reset the graph
        if(command == "INIT"){
            graph.makeEmpty();
        }
        
        //Reprints the command input options
        else if(command == "OPTIONS"){
            cout << ">>> GRAPH DRIVER <<<" << endl;
            cout << "\tCOMMANDS:" << endl;
            cout << "\t\tINIT: Reset the graph." << endl;
            cout << "\t\tADDV: Requests an input, then adds the vertex to the graph." << endl;
            cout << "\t\tADDE: Requests two input vertices and a weight, then adds the edge to the graph." << endl;
            cout << "\t\tPRINT: Prints the graph as (from_Vertex, to_Vertex, weight) spaced by source." << endl;
            cout << "\t\tTEST1: Clears the graph and then creates an example flight cost graph." << endl;
            cout << "\t\tTEST2: Clears the graph and then creates and example game map." << endl;
            cout << "\t\tPATH: Requests tow input vertices and gives the shortest path between the two." << endl;
            cout << "\t\tACCURACY: Runs a test to test the accuracy of TEST1's created graph." << endl;
            cout << "\t\tOPTIONS: Prints the input command options again." << endl;
            cout << "\t\tQUIT: Exits the program." << endl << endl;
        }
        //ADDV - Request a string, then add that string as a vertex
        else if(command == "ADDV"){
            string name;
            cout << endl << "Enter a vertex name:\t";
            cin >> name;
            cout << endl;
            graph.addVertex(name);
        }

        //ADDE - Request the name of two vertices, then a weight between the vertices.
        else if(command == "ADDE"){
            string source;
            string destination;
            int weight;
            cout << "Enter the name of the source vertex:\t";
            cin >> source;
            cout << endl << "Enter the name of the destination vertex:\t";
            cin >> destination;
            cout << endl << "Enter the edge weight:\t";
            cin >> weight;
            cout << endl;
            graph.addEdge(source, destination, weight);
        }

        //PRINT - Prints each vertex in the grpah with all edges and weights associated
        else if(command == "PRINT"){
            cout << ">>> PRINTED GRAPH (separated by vertex) <<<" << endl << endl;
            graph.printGraph();
        }

        //TEST1 - Initialize the graph. Then insert the vertices and edges defined for
        //        for the example on slide 6 in lecture 24
        else if(command == "TEST1"){
            graph.makeEmpty();
            cout << ">>> CREATING GRAPH: TEST 1 <<<" << endl << endl;
            graph.addVertex("Austin");
            graph.addVertex("Dallas");
            graph.addVertex("Washington");
            graph.addVertex("Denver");
            graph.addVertex("Atlanta");
            graph.addVertex("Houston");
            graph.addVertex("Chicago");
            
            graph.addEdge("Austin", "Denver", 200);
            graph.addEdge("Austin", "Houston", 160);
            graph.addEdge("Dallas", "Denver", 780);
            graph.addEdge("Dallas", "Austin", 200);
            graph.addEdge("Dallas", "Chicago", 900);
            graph.addEdge("Washington", "Dallas", 1300);
            graph.addEdge("Washington", "Atlanta", 600);
            graph.addEdge("Atlanta", "Houston", 800);
            graph.addEdge("Atlanta", "Washington", 600);
            graph.addEdge("Houston", "Atlanta", 800);
            graph.addEdge("Denver", "Atlanta", 1400);
            graph.addEdge("Denver", "Chicago", 1000);
            graph.addEdge("Chicago", "Denver", 1000);
            cout << endl << "Graph created." << endl << endl;
        }
        
        //Test for accuracy
        else if(command == "ACCURACY"){
            double correct = 0;
            bool TEST1 = false;
            bool TEST2 = false;
            bool TEST3 = false;
            vector<string> *temp = new vector<string>;
            
            //For first Test
            cout << "Testing TEST1 graph:" << endl;
            cout << "\tTesting flight path Washington -> Dallas -> Austin: Cost should be 1500"<< endl;
            graph.getAdjacent("Washington", *temp);
            int actual = 1500;
            int g_rep = 0;
            for (int i = 0; i < temp->size(); i++) {
                if(temp->at(i) == "Dallas"){
                    g_rep += graph.getWeight("Washington", "Dallas");
                }
            }
            temp->clear();
            graph.getAdjacent("Dallas", *temp);
            for (int i = 0; i < temp->size(); i++) {
                if(temp->at(i) == "Austin"){
                    g_rep += graph.getWeight("Dallas", "Austin");
                }
            }
            if(g_rep == actual){
                correct++;
                cout << "\t\tCORRECT" << endl << endl;
            }
            else{
                cout << "\t\tINCORRECT >>> Implemented Graph Cost: " << g_rep << endl << endl;
            }
            
            //Reset Variables
            temp->clear();
            g_rep = 0;
            
            //for second Test
            cout << "Testing TEST1 graph:" << endl;
            cout << "\tTesting flight path Dallas -> Chicago -> Denver: Cost should be 1900"<< endl;
            graph.getAdjacent("Dallas", *temp);
            actual = 1900;
            for (int i = 0; i < temp->size(); i++) {
                if(temp->at(i) == "Chicago"){
                    g_rep += graph.getWeight("Dallas", "Chicago");
                }
            }
            temp->clear();
            graph.getAdjacent("Chicago", *temp);
            for (int i = 0; i < temp->size(); i++) {
                if(temp->at(i) == "Denver"){
                    g_rep += graph.getWeight("Chicago", "Denver");
                }
            }
            if(g_rep == actual){
                correct++;
                cout << "\t\tCORRECT" << endl << endl;
            }
            else{
                cout << "\t\tINCORRECT >>> Implemented Graph Cost: " << g_rep << endl << endl;
            }
            
            //Reset Variables
            temp->clear();
            g_rep = 0;
            
            //for second Test
            cout << "Testing TEST1 graph:" << endl;
            cout << "\tTesting flight path Atlanta -> Washington -> Dallas -> Austin -> Houston -> Atlanta: Cost should be 3060"<< endl;
            graph.getAdjacent("Atlanta", *temp);
            actual = 3060;
            for (int i = 0; i < temp->size(); i++) {
                if(temp->at(i) == "Washington"){
                    g_rep += graph.getWeight("Atlanta", "Washington");
                }
            }
            temp->clear();
            graph.getAdjacent("Washington", *temp);
            for (int i = 0; i < temp->size(); i++) {
                if(temp->at(i) == "Dallas"){
                    g_rep += graph.getWeight("Washington", "Dallas");
                }
            }
            temp->clear();
            graph.getAdjacent("Dallas", *temp);
            for (int i = 0; i < temp->size(); i++) {
                if(temp->at(i) == "Austin"){
                    g_rep += graph.getWeight("Dallas", "Austin");
                }
            }
            temp->clear();
            graph.getAdjacent("Austin", *temp);
            for (int i = 0; i < temp->size(); i++) {
                if(temp->at(i) == "Houston"){
                    g_rep += graph.getWeight("Austin", "Houston");
                }
            }
            temp->clear();
            graph.getAdjacent("Houston", *temp);
            for (int i = 0; i < temp->size(); i++) {
                if(temp->at(i) == "Atlanta"){
                    g_rep += graph.getWeight("Houston", "Atlanta");
                }
            }
            if(g_rep == actual){
                correct++;
                cout << "\t\tCORRECT" << endl << endl;
            }
            else{
                cout << "\t\tINCORRECT >>> Implemented Graph Cost: " << g_rep << endl << endl;
            }
            
            
            
            //Testing Inputted Variables
            cout << "Testing manual input: Vertex 1 to Vertex 2, Cost 10" << endl << endl;
            graph.addVertex("1");
            graph.addVertex("2");
            graph.addEdge("1", "2", 10);
            g_rep = graph.getWeight("1", "2");
            actual = 10;
            if(g_rep == actual){
                correct++;
                cout << "\t\tCORRECT" << endl << endl;
            }
            else{
                cout << "\t\tINCORRECT >>> Implemented Graph Cost: " << g_rep << endl << endl;
            }
            
            //Testing invalid accesss
            bool test = false;
            cout << "Testing invalid edge input:" << endl << endl;
            cout << "\tTesting '1' to '3': invalid '3'" << endl;
            test = graph.addEdge("1", "3", 1);
            if(!test){
                correct++;
                cout << "\t\tCORRECT" << endl << endl;
            }
            else{
                cout << "\t\tINCORRECT" << endl;
            }
            cout << "\tTesting '3' to '2': invalid '3'" << endl;
            test = graph.addEdge("3", "2", 1);
            if(!test){
                correct++;
                cout << "\t\tCORRECT" << endl << endl;
            }
            else{
                cout << "\t\tINCORRECT" << endl;
            }
            cout << "\tTesting '4' to '5': invalid '4' and '5'" << endl;
            test = graph.addEdge("4", "5", 1);
            if(!test){
                correct++;
                cout << "\t\tCORRECT" << endl << endl;
            }
            else{
                cout << "\t\tINCORRECT" << endl;
            }
            
            //Test for Dijkstra
            //Test 1 Shortest path from Houstin to Austin
            cout << "Testing Dijkstra" << endl << endl;
            cout << "\tTesting shortest path from Houston to Austin:" << endl
                 << "\t\t Path: Houston -> Atlanta -> Washington -> Dallas -> Austin" << endl
                 << "\t\tCost: 2900" << endl;
            temp->clear();
            g_rep = 2900;
            actual = graph.dijkstra("Houston", "Austin", *temp);
            if(actual != -1){
                cout << "Cost: " << actual << endl;
                for(int i = 0; i < temp->size(); i++){
                    if(i != temp->size()-1){
                        cout << temp->at(i) << " -> ";
                    }
                    else{
                        cout << temp->at(i);
                    }
                }
                cout << endl << endl;
            }
            if(g_rep == actual){
                correct++;
                cout << "\t\tCORRECT" << endl << endl;
            }
            else{
                cout << "\t\tINCORRECT" << endl;
            }
            
            
            cout << "\tTesting shortest path from Dallas to Houston:" << endl
                 << "\t\tPath: Dallas -> Austin -> Houston" << endl
                 << "\t\tCost: 360" << endl;
            temp->clear();
            g_rep = 360;
            actual = graph.dijkstra("Dallas", "Houston", *temp);
            if(actual != -1){
                cout << "Cost: " << actual << endl;
                for(int i = 0; i < temp->size(); i++){
                    if(i != temp->size()-1){
                        cout << temp->at(i) << " -> ";
                    }
                    else{
                        cout << temp->at(i);
                    }
                }
                cout << endl << endl;
            }
            if(g_rep == actual){
                correct++;
                cout << "\t\tCORRECT" << endl << endl;
            }
            else{
                cout << "\t\tINCORRECT" << endl;
            }
            
            //Testing invalid path
            cout << "\tTesting no valid path" << endl;
            cout << "\t\tPath: No path from Houston to Fake." << endl;
            cout << "\t\tCost: -1" << endl << endl;
            temp->clear();
            cout << endl;
            graph.addVertex("Fake");
            graph.addEdge("Fake", "Washington", 10000);
            g_rep = -1;
            actual = graph.dijkstra("Houston", "Fake", *temp);
            if(g_rep == actual){
                correct++;
                cout << "\t\tCORRECT" << endl << endl;
            }
            else{
                cout << "\t\tINCORRECT" << endl;
            }
            
            //Testing Graph Clear
            cout << "Testing graph clear:" << endl;
            graph.makeEmpty();
            int size = graph.getSize();
            if(size == 0){
                correct++;
                cout << "\t\tCORRECT" << endl << endl;
            }
            else{
                cout << "\t\tINCORRECT >>> Implemented Graph Size: " << size << endl << endl;
            }
            
            correct = (correct / 11.0) * 100;
            cout << endl << ">>> ACCURACY SCORE: " << correct << "% <<<" << endl << endl;
            graph.makeEmpty();
            
        }

        //TEST2 - Initializes the graph. Then inserts the vertices and edges defined for
        //        the example on slide 3 in lecture 24
        else if(command == "TEST2"){
            graph.makeEmpty();
            cout << ">>> CREATING GRAPH: TEST 2 <<<" << endl << endl;
            graph.addVertex("1A");
            graph.addVertex("1B");
            graph.addVertex("1C");
            graph.addVertex("1D");
            graph.addVertex("1E");
            graph.addVertex("1F");
            graph.addVertex("1G");
            graph.addVertex("1H");
            graph.addVertex("1I");
            graph.addVertex("1J");
            graph.addVertex("2A");
            graph.addVertex("2B");
            graph.addVertex("2C");
            graph.addVertex("2D");
            graph.addVertex("2E");
            graph.addVertex("2F");
            graph.addVertex("2G");
            graph.addVertex("2H");
            graph.addVertex("2I");
            graph.addVertex("2J");
            graph.addVertex("3A");
            graph.addVertex("3B");
            graph.addVertex("3C");
            graph.addVertex("3D");
            graph.addVertex("3E");
            graph.addVertex("3F");
            graph.addVertex("3G");
            graph.addVertex("3H");
            graph.addVertex("3I");
            graph.addVertex("3J");
            graph.addVertex("4A");
            graph.addVertex("4B");
            graph.addVertex("4C");
            graph.addVertex("4D");
            graph.addVertex("4E");
            graph.addVertex("4F");
            graph.addVertex("4G");
            graph.addVertex("4H");
            graph.addVertex("4I");
            graph.addVertex("4J");
            graph.addVertex("5A");
            graph.addVertex("5B");
            graph.addVertex("5C");
            graph.addVertex("5D");
            graph.addVertex("5E");
            graph.addVertex("5F");
            graph.addVertex("5G");
            graph.addVertex("5H");
            graph.addVertex("5I");
            graph.addVertex("5J");
            graph.addVertex("6A");
            graph.addVertex("6B");
            graph.addVertex("6C");
            graph.addVertex("6D");
            graph.addVertex("6E");
            graph.addVertex("6F");
            graph.addVertex("6G");
            graph.addVertex("6H");
            graph.addVertex("6I");
            graph.addVertex("6J");
            
            graph.addEdge("1A", "1B", 2);
            graph.addEdge("1A", "2A", 3);
            graph.addEdge("1B", "1A", 3);
            graph.addEdge("1B", "2B", 3);
            graph.addEdge("1B", "1C", 1);
            graph.addEdge("1C", "1B", 2);
            graph.addEdge("1C", "2C", 3);
            graph.addEdge("1C", "1D", 1);
            graph.addEdge("1D", "1C", 1);
            graph.addEdge("1D", "2D", 1);
            graph.addEdge("1D", "1E", 4);
            graph.addEdge("1E", "1D", 1);
            graph.addEdge("1E", "2E", 2);
            graph.addEdge("1J", "2J", 1);
            graph.addEdge("2A", "2B", 3);
            graph.addEdge("2A", "1A", 3);
            graph.addEdge("2B", "1B", 2);
            graph.addEdge("2B", "2A", 3);
            graph.addEdge("2B", "2C", 3);
            graph.addEdge("2C", "1C", 1);
            graph.addEdge("2C", "2B", 3);
            graph.addEdge("2C", "3C", 3);
            graph.addEdge("2C", "2D", 1);
            graph.addEdge("2D", "1D", 1);
            graph.addEdge("2D", "2C", 3);
            graph.addEdge("2D", "3D", 1);
            graph.addEdge("2D", "2E", 2);
            graph.addEdge("2E", "1E", 4);
            graph.addEdge("2E", "2D", 1);
            graph.addEdge("2E", "3E", 1);
            graph.addEdge("2E", "2F", 4);
            graph.addEdge("2F", "2E", 2);
            graph.addEdge("2F", "3F", 1);
            graph.addEdge("2I", "3I", 1);
            graph.addEdge("2I", "2J", 1);
            graph.addEdge("2J", "1J", 1);
            graph.addEdge("2J", "2I", 4);
            graph.addEdge("2J", "3J", 1);
            graph.addEdge("3C", "2C", 3);
            graph.addEdge("3C", "3D", 1);
            graph.addEdge("3C", "4C", 3);
            graph.addEdge("3D", "2D", 1);
            graph.addEdge("3D", "3C", 3);
            graph.addEdge("3D", "4D", 1);
            graph.addEdge("3D", "3E", 1);
            graph.addEdge("3E", "2E", 2);
            graph.addEdge("3E", "3D", 1);
            graph.addEdge("3E", "4E", 2);
            graph.addEdge("3E", "3F", 1);
            graph.addEdge("3F", "2F", 4);
            graph.addEdge("3F", "3E", 1);
            graph.addEdge("3F", "4F", 2);
            graph.addEdge("3F", "3G", 1);
            graph.addEdge("3G", "3F", 1);
            graph.addEdge("3G", "4G", 2);
            graph.addEdge("3G", "3H", 1);
            graph.addEdge("3H", "3G", 1);
            graph.addEdge("3H", "4H", 2);
            graph.addEdge("3H", "3I", 1);
            graph.addEdge("3I", "2I", 4);
            graph.addEdge("3I", "3H", 1);
            graph.addEdge("3I", "4I", 2);
            graph.addEdge("3I", "3J", 1);
            graph.addEdge("3J", "2J", 1);
            graph.addEdge("3J", "3I", 1);
            graph.addEdge("3J", "4J", 2);
            graph.addEdge("4C", "3C", 3);
            graph.addEdge("4C", "5C", 3);
            graph.addEdge("4C", "4D", 1);
            graph.addEdge("4D", "3D", 1);
            graph.addEdge("4D", "4C", 3);
            graph.addEdge("4D", "5D", 1);
            graph.addEdge("4D", "4E", 2);
            graph.addEdge("4E", "3E", 1);
            graph.addEdge("4E", "4D", 1);
            graph.addEdge("4E", "5E", 4);
            graph.addEdge("4E", "4F", 2);
            graph.addEdge("4F", "3F", 4);
            graph.addEdge("4F", "4E", 1);
            graph.addEdge("4F", "5F", 2);
            graph.addEdge("4F", "4G", 1);
            graph.addEdge("4G", "3G", 1);
            graph.addEdge("4G", "4F", 2);
            graph.addEdge("4G", "4H", 2);
            graph.addEdge("4H", "3H", 1);
            graph.addEdge("4H", "4G", 1);
            graph.addEdge("4H", "4I", 2);
            graph.addEdge("4I", "3I", 1);
            graph.addEdge("4I", "4H", 1);
            graph.addEdge("4I", "4J", 2);
            graph.addEdge("4J", "3J", 1);
            graph.addEdge("4J", "4I", 1);
            graph.addEdge("4J", "5J", 2);
            graph.addEdge("5C", "4C", 3);
            graph.addEdge("5C", "6C", 2);
            graph.addEdge("5C", "5D", 1);
            graph.addEdge("5D", "4D", 1);
            graph.addEdge("5D", "5C", 3);
            graph.addEdge("5D", "6D", 1);
            graph.addEdge("5D", "5E", 4);
            graph.addEdge("5E", "4E", 2);
            graph.addEdge("5E", "5D", 1);
            graph.addEdge("5E", "6E", 2);
            graph.addEdge("5E", "5F", 2);
            graph.addEdge("5F", "4F", 2);
            graph.addEdge("5F", "5E", 4);
            graph.addEdge("5F", "6F", 2);
            graph.addEdge("5J", "4J", 2);
            graph.addEdge("5J", "6J", 2);
            graph.addEdge("6B", "5B", 3);
            graph.addEdge("6B", "6C", 2);
            graph.addEdge("6C", "6B", 3);
            graph.addEdge("6C", "5C", 1);
            graph.addEdge("6C", "6D", 2);
            graph.addEdge("6D", "6C", 2);
            graph.addEdge("6D", "5D", 1);
            graph.addEdge("6D", "6E", 2);
            graph.addEdge("6E", "6D", 1);
            graph.addEdge("6E", "5E", 4);
            graph.addEdge("6E", "6F", 2);
            graph.addEdge("6F", "6D", 2);
            graph.addEdge("6F", "5F", 2);
            graph.addEdge("6H", "6I", 2);
            graph.addEdge("6I", "6J", 2);
            graph.addEdge("6I", "6H", 2);
            graph.addEdge("6J", "5J", 2);
            graph.addEdge("6J", "6I", 2);
            cout << endl << "Graph created." << endl << endl;

        }
        
        else if(command == "PATH"){
            vector<string>* temp = new vector<string>;
            int cost;
            string source;
            string destination;
            cout << "Enter the source vertex name:\t";
            cin >> source;
            cout << endl << "Enter the desination vertex name:\t";
            cin >> destination;
            cout << endl;
            
            cost = graph.dijkstra(source, destination, *temp);
            if(cost != -1){
                cout << "Cost: " << cost << endl;
                for(int i = 0; i < temp->size(); i++){
                    if(i != temp->size()-1){
                        cout << temp->at(i) << " -> ";
                    }
                    else{
                        cout << temp->at(i);
                    }
                }
                cout << endl << endl;
            }
            
        }

        //QUIT - Quit the test program
        else if (command == "QUIT"){
            cout << ">>> EXITING PROGRAM <<<" << endl;
            testing = false;
        }

        //Any non valid command
        else{
            cout << "Not a valid command. Please try again!" << endl << endl;
        }
    }
    return 0;
}