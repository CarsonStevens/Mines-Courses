/*
 * Author:      Carson Stevens
 * Date:        November 26, 2018
 * Description: Creates a new graph to solve the maze problem as seen in
 *              doc/18SpaceWreck.pdf
*/

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <stack>
#include <unordered_map>
// BOOST LIBRARY
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/visitors.hpp>

using namespace std;


//////////////////
//   STRUCTS   //
/////////////////

//Used to recorder the steps taken
struct Room {
    char character;
    int destination;
};

//Represents the corridors from input
struct Corridor{
    int starting_room;
    int ending_room;
    char color;
};


//Vertices of the revised graph
struct Node{
    
    //Variables needed for nodes in revised graph
    int person1_position;
    int person2_position;
    char person1_color;
    char person2_color;
    
    //Overload '==' to compare person positions & colors
    bool operator== (const Node &node) const {
        return person1_position == node.person1_position && person2_position == node.person2_position
            && person1_color    == node.person1_color && person2_color       == node.person2_color;
    }
    
    //Overload '<' for unordered_map hash function
    bool operator<(const Node& node) const {
        return person1_position < node.person1_position || person2_position < node.person2_position
            || person1_color    < node.person1_color || person2_color       < node.person2_color;
    }
};


//Since unordered map was used, recommened hash function was implemented
struct hash_fn {
    
    //Recommended Hash function for C++
    size_t operator()(const Node& k) const{
    using std::size_t;
    using std::hash;
    using std::string;

    return ((hash<int>()(k.person1_position)
            ^ (hash<int>()(k.person2_position) << 1)) >> 1)
            ^ (hash<int>()(k.person1_color) << 1)
            ^ (hash<int>()(k.person2_color) << 1);
    }
};


//Custom Visitor to better handle custom objects
struct Visitor : boost::default_dfs_visitor {
    
    //Variables for custom visitor
    vector<Room> steps;
    unordered_map<Node, Node, hash_fn>& predecessors;
    
    //Constructor for custom visitor
    Visitor (unordered_map<Node, Node, hash_fn>& pred) : steps(std::vector<Room>()),predecessors(pred) {}
      
    template < typename Edge, typename Graph >
    void tree_edge(Edge e, Graph& g) {
        predecessors[g[e.m_target]] = g[e.m_source];
    }
};

//Types needed to create and traverse graph
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, Node, Corridor> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor vertex_t;
typedef boost::graph_traits<Graph>::edge_descriptor edge_t;

//Class that solves maze
class Solver{
    public:
        
        //General graph variables
        int node_num;
        int corridor_num;
        vector<Node> nodes;
        vector<char> node_colors;
        vector<Corridor> corridors;
        
        //Graph and traversal variables
        Graph G;
        unordered_map<Node, Node, hash_fn> predecessors;
        Visitor v;
        stack<string> route;
        vector<Room> steps;
        
        //Constructor
        Solver() : 
            G(Graph()),
            steps(std::vector<Room>()),
            predecessors(std::unordered_map<Node, Node, hash_fn>()),
            v(Visitor(predecessors)) {}
        
        //Creates the new graph
        void create_revised_graph(vector<Node> nodes, vector<char> node_colors, vector<Corridor> corridors){

            //Current Node while building
            Node current_node;
            boost::add_vertex({nodes.at(0).person1_position, nodes.at(0).person2_position, nodes.at(0).person1_color, nodes.at(0).person2_color}, G);
            
            //Create Graph with appropriate ndoes
            for(int i = 0; i < nodes.size(); i++){
                current_node = nodes.at(i);
                
                //Determine the corridors associated with that node
                for(auto corridor : corridors){
                    //Corridor person1 can enter
                    if(corridor.starting_room == current_node.person1_position && corridor.color == current_node.person2_color){
                        Node node;
                        node.person1_position =corridor.ending_room;
                        node.person2_position = current_node.person2_position;
                        node.person1_color = node_colors.at(corridor.ending_room-1);
                        node.person2_color = current_node.person2_color;
                    
                        // Check to see if that node already exists
                        if(find(nodes.begin(), nodes.end(), node) == nodes.end()){
                            nodes.push_back(node);
                            // Add vertex
                            int v = boost::add_vertex({corridor.ending_room, node.person2_position, node_colors.at(corridor.ending_room-1), node.person2_color}, G);
                            // Add edge (CHECK)
                            boost::add_edge(*(boost::vertices(G).first+i), v, corridor, G);
                        }
                    }
                    
                    //Corridor person2 can enter
                    if(corridor.starting_room == current_node.person2_position && corridor.color == current_node.person1_color){
                        Node node;
                        node.person1_position = current_node.person1_position;
                        node.person2_position = corridor.ending_room;
                        node.person1_color = current_node.person1_color;
                        node.person2_color = node_colors.at(corridor.ending_room -1);
                        
                        //Check to see if that node already exists
                        if(std::find(nodes.begin(), nodes.end(), node) == nodes.end()){
                            nodes.push_back(node);
                            // Add vertex
                            int v = boost::add_vertex({node.person1_position, corridor.ending_room, node.person2_color, node_colors.at(corridor.ending_room-1)}, G);
                            // Add edge
                            boost::add_edge(*(boost::vertices(G).first+i), v, corridor, G);
                        }
                    }
                }
            }
            
            //TESTING
            //cout << "Created Graph: Verticies: " << boost::num_vertices(G) << " Edges: " << boost::num_edges(G) << endl;
            
            //Update variable
            this->nodes = nodes;
        }
        
        //Does the traversal
        void solve(){
            
            //Preform DFS
            boost::depth_first_search(G, visitor(v).root_vertex(vertex(0, G)));
            
            //Variables needed for showing route.
            Node predecessor;
            Node start_node = G[0];
            Node current_node = G[num_vertices(G)-1];
            
            //String to hold each step in the route in the needed output
            string step;
            
            //Traverse graph
            while(!(start_node == current_node)){
                predecessor = predecessors.at(current_node);
                
                //Update current values
                int person1_starting = predecessor.person1_position;
                int person2_starting = predecessor.person2_position;
                int person1_ending = current_node.person1_position;
                int person2_ending = current_node.person2_position;
                
                //Find out which person moved
                if(person1_starting != person1_ending){
                    step = "L " + to_string(person1_ending);
                    route.push(step);
                }
                else if(person2_starting != person2_ending){
                    step = "R " + to_string(person2_ending);
                    route.push(step);
                }
                
                //Update current node for the next iteration
                current_node = predecessor;
            }
        }
        
        //Prints the stack created by DFS
        void printRoute(){
            while(!route.empty()){
                cout << route.top() << endl;
                route.pop();
            }
        }
};

//////////////////
//  FUNCTIONS  //
/////////////////

//Reads the input file and returns the revised graph
Solver process_file(string file){
    
    //General variables to read in and proper containers to hold them
    vector<Node> nodes;
    vector<Corridor> corridors;
    vector<char> node_colors;
    int corridor_num;
    int node_num;
    int person1_position;
    int person2_position;
    char color;
    Solver maze;
    
    //Load file
    ifstream maze_input(file);
    if(!maze_input){
        cerr << "Error opening input file:\t" << file << "\nClosing Solver.";
        exit(1);
    }
    
    //Read in node and corridor numbers
    maze_input >> node_num >> corridor_num;
    
    //Read in node colors
    for(int i = 0; i < node_num; i++){
        maze_input >> color;
        node_colors.push_back(color);
    }
    
    //Read in start positions
    maze_input >> person1_position >> person2_position;
    
    //Create root node
    Node root;
    root.person1_position = person1_position;
    root.person2_position = person2_position;
    root.person1_color = node_colors.at(person1_position -1);   // -1 for 0 indexing
    root.person2_color = node_colors.at(person2_position -1);
    nodes.push_back(root);
    
    //Read in rest of nodes
    string line;
    int u, v;
    while(!maze_input.eof()){
        getline(maze_input, line);
        stringstream ss(line);
        ss >> u >> v >> color;
        //Create Corridor
        Corridor corridor;
        corridor.starting_room = u;
        corridor.ending_room = v;
        corridor.color = color;
        corridors.push_back(corridor);
    }
    
    //Close input file
    maze_input.close();
    
    //TESTING
    //cout << "Nodes size: " << nodes.size() << endl << "Node color size: " << node_colors.size() << endl << "Corridors size: " << corridors.size() << endl;
    
    //Create the maze
    maze.create_revised_graph(nodes, node_colors, corridors);
    return maze;
}

///////////////////
//     MAIN     //
//////////////////

int main(int argc, char* argv[]){
    
//Handle Input file checking/setting from flag
    string file;
    if(argc < 2){
        cerr << "Cannot run without maze input file\nClosing Solver.";
        exit(1);
    }
    else{
        file = argv[1];
    }
    
//Use Solver to solve given input
    
    //Reads in file and creates revised graph
    Solver maze = process_file(file);
    
    //Does the traversal and creates the stack from DFS
    maze.solve();
    
    //Prints the stack to provide the correct order
    maze.printRoute();
    
    return 0;
}