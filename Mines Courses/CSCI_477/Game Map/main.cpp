/*
 * Author: Carson Stevens
 * Date: March 15, 2019
 * Description: Create a Driver program to create and test graph
 *              inputs.
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>

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
        adjList[vertex_name] = new_adj_list;

        cout << endl << "Vertex added to the graph." << endl << endl;
    }

    bool addEdge(string fromVertex, string toVertex, int cost) {
        Edge edge(toVertex, cost);

        if(adjList.count(fromVertex) == 1 && adjList.count(toVertex) == 1){
            adjList[fromVertex].push_back(edge);
            cout << endl << "Edge added to the graph." << endl << endl;
            return true;
        }
        cout << endl << "Couldn't find one of the vertexes." << endl;
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
        return -1;
    }

    bool getAdjacent(string fromVertex, vector<string> &vVertex) {
        if(adjList.count(fromVertex) == 1){
            vVertex = adjList[fromVertex].destination;
            return true;
        }
        return false;
    }

    void printGraph(){
        for(auto vertex : adjList){
            for(Edge edge : vertex.second){
                cout << "(" << vertex.first << ", " << edge.destination << ", " << edge.cost << ")" << endl;
            }
            cout << endl;
        }
    }

};

int main(){

    int testing = true;
    string command;
    Graph graph;

    while(testing){

        cout << "Input Graphing Command:\t";
        cin >> command;

        //INIT - Reset the graph
        if(command == "INIT"){
            graph.makeEmpty();
        }

            //ADDV - Request a string, then add that string as a vertex
        else if(command == "ADDV"){
            string name;
            cout << endl << "Enter a vertex name:\t";
            cin >> name;

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

            graph.addEdge(source, destination, weight);
        }

            //PRINT - Prints each vertex in the grpah with all edges and weights associated
        else if(command == "PRINT"){
            graph.printGraph();
        }

            //TEST1 - Initialize the graph. Then insert the vertices and edges defined for
            //        for the example on slide 6 in lecture 24
        else if(command == "TEST1"){

        }

            //TEST2 - Initializes the graph. Then inserts the vertices and edges defined for
            //        the example on slide 3 in lecture 24
        else if(command == "TEST2"){

        }

            //QUIT - Quit the test program
        else if (command == "QUIT"){
            cout << "EXITING PROGRAM" << endl;
            testing = false;
        }

            //Any non valid command
        else{
            cout << "Not a valid command. Please try again!" << endl << endl;
        }
    }

    return 0;
}