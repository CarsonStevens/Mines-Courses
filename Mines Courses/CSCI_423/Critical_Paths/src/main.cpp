/*  Author: Carson Stevens
*   Date: October 27, 2018
*   Description:    Display the probability of the critical
*                   path using a Monte Carlo simulation.
*
*/

#include <iostream>
#include <iomanip>
#include <utility>
#include <sstream>
#include <fstream>
#include <queue>
#include <map>
#include <vector>
#include <algorithm>

using namespace std;

struct NodePair{
    pair<int, int> nodes;
    double weight;
};

struct Random{
    ifstream randomNumbers;
    double random;
    void open_file(string file){
        randomNumbers.open(file);
        if (!randomNumbers){
            cout << "Error Loading Input Random File: " << file << "." << endl;  
            exit(1);
        }
    }
    double getRandom(){
        randomNumbers >> random;
        return random;
    }
};


vector<vector<int>> BuildPaths(vector<vector<int>> N, int currentrow){
    vector<vector<int>> paths;
    vector<vector<int>> finalpaths;
    if(currentrow == N.size()-1){
        vector<int> temp;
        temp.push_back(currentrow);
        finalpaths.push_back(temp);
        return finalpaths;
    } //
    
    for(int i = 0; i < N[currentrow].size(); i++){
        
        if(N[currentrow][i] == 1){
            int row_jump;
            for(int k = 0; k < N.size(); k++){

                if(N[k][i] == -1){
                    row_jump = k;
                    break;
                }
            }

            paths = BuildPaths(N, row_jump);
            for(int j = 0; j < paths.size(); j++){
                paths.at(j).push_back(currentrow);
                finalpaths.push_back(paths.at(j));
            }
        }
    }
    return finalpaths;
}

pair<double, vector<int>> T(vector<vector<int>> N, vector<NodePair> nodePaths, int j){
    pair<double, vector<int>> longest_path;

    if(j == 0){
        longest_path.first = 0;
        longest_path.second = vector<int>(1,0);
        return longest_path;
    }

    int k = 0;
    int l = 0;
    double Tmax = 0;
    int rank = 0;
    for(int q = 0; q < N[j].size(); q++){
        if (N[j][q] == -1){
            rank++;
        }
    }
    // l < number of -1 in row
    while(l < rank){
        if(N[j][k] == -1){
            int i = 0;
            while(N[i][k] != 1){
                i++;
            }
            double weight;
            for(auto x : nodePaths){
                if(x.nodes.first == i + 1 && x.nodes.second == j + 1){
                    weight = x.weight;
                    break;
                }
            }
            auto bigT = T(N, nodePaths, i);
            double t = bigT.first + weight;
            if(t > Tmax){
                Tmax = t;
                longest_path.first = Tmax;
                longest_path.second = bigT.second;
                longest_path.second.push_back(j);
            }
            l++;
        }
        k++;
    }

    return longest_path;
}


int main(int argc, char* argv[]){
    
    // cout << argv[0] << endl << argv[1] << endl << argv[2] << endl << argv[3] << endl;
    int iterations = stoi(argv[2]);
    Random random;
    string line;
    string output = "";
    int path1;
    int path2;
    double weight;
    queue <double> random_numbers;
    vector<NodePair> nodePaths;
    map <int, int> number_of_arcs;
    map <vector<int>, int> finalData; 
    vector<vector<int>> finalpaths;
    vector<string> outputs;
    
    string file = argv[1];
    random.open_file(file);
    // if the file is empty or won't load, it outputs the message

    ifstream paths;
    paths.open(argv[3]);
    if (!paths){
        cout << "Error Loading Input Path File: " << argv[3] << "." << endl;  
        return 1;
    }
    

    int end_node = 0;
    while(paths >> path1 >> path2 >> weight){
        
        //Define parameters for a node arc
        NodePair node_pair;
        pair <double, double> node(path1,path2);
        node_pair.nodes = node;
        
        //For finding size of matrix
        if(path2 > end_node){
            end_node = path2;
        }
        
        //Give arc its weight
        node_pair.weight = weight;

        //Add arc to map
        if(number_of_arcs.count(path1) == 1){
            number_of_arcs[path1]++;
        }
        else{
            number_of_arcs[path1] = 1;
        }
        
        //Add arc to vector of arcs
        nodePaths.push_back(node_pair);
    }
    paths.close();
    
    // Print the # of arcs coming from each node.
    // cout << "Ranks of nodes" << endl;
    // for(auto i : number_of_arcs){
    //     cout << i.first << ":\t" << i.second << endl;
    // }
    
    
    // Create incident matrix using 2d vector
    vector<vector<int>> N(end_node, vector<int>(nodePaths.size(), 0));
    

    for(int j = 0; j < nodePaths.size(); j++){
        N[nodePaths[j].nodes.first - 1][j] = 1;
        N[nodePaths[j].nodes.second - 1][j] = -1;
    }


    finalpaths = BuildPaths(N, 0);
    for(auto path : finalpaths){
        reverse(path.begin(), path.end());
        finalData[path] = 0;
    }
    
    //Printt the finalPaths
    // for(auto i : finalpaths){
    //     for(auto j : i){
    //         cout << j << ", ";
    //     }
    //     cout << endl;
    // }
    
    // Print the matrix
    // for(int i = 0; i < end_node; i++){
    //     for(int j = 0; j < nodePaths.size(); j++){
    //         if(N[i][j] == -1){
    //             cout << "2 ";
    //             continue;
    //         }
    //         cout << N[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    for(int i = 0; i < iterations; i++){
        vector<NodePair> copyNodePaths(nodePaths);
        for(int j = 0; j < copyNodePaths.size(); j++){
            copyNodePaths[j].weight = copyNodePaths[j].weight * random.getRandom();
        }
        auto t = T(N, copyNodePaths, end_node-1);
        finalData.at(t.second)++;
    }

    for(auto x : finalData){
        output = "OUTPUT\t:";
        for(int z = 0; z < x.first.size()-1; z++){
            output += 'a' + to_string(x.first.at(z)+1) + "/" + to_string(x.first.at(z+1)+1) + ",";
        }
        output = output.substr(0, output.size()-1);
        output += ":";
        outputs.push_back(output);
    }

    int counter = 0;
    for(auto x : finalData){
        cout << setw(30) << left << outputs[counter] << setw(10) << right << setprecision(4) << fixed << double(double(x.second) / double(iterations)) << endl;
        counter++;
    }
    return 0;
}