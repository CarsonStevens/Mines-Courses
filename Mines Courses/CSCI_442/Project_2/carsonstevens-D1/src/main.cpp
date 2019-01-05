/**
 * This file contains the main() function that drives the simulation. You'll
 * need to add logic to this file to create a Simulation instance and invoke its
 * run() method.
 */

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <cstdlib>
#include <sstream>
#include "physical_address/physical_address.h"
#include "flag_parser/flag_parser.h"
#include "process/process.h"
#include "page/page.h"
#include "frame/frame.h"


using namespace std;

void file_input_reader(string filename, map<int, string>& process_contents, map<int, vector<string>>& process_virtual_addresses){
  int num_of_processes = 0;
  int current_process = 0;
  int pid = 0;
  string process_image;
  string virtual_address;
  
  ifstream file_input;
  file_input.open(filename);
  
  //Handle if can't open file
  if (!file_input){
        cerr << "Error loading " << filename << ".\n EXITING" << endl;  
        exit (EXIT_FAILURE);
  }
  
  file_input >> num_of_processes;
  
  //Load in processes and images
  while(current_process < num_of_processes){
    file_input >> pid >> process_image;
    process_contents[pid] = process_image;
    current_process++;
  }
  
  while(file_input >> pid >> virtual_address){
    if(process_virtual_addresses.count(pid) == 0){
      vector<string> temp;
      temp.push_back(virtual_address);
      process_virtual_addresses[pid] = temp;
    }
    else{
      process_virtual_addresses[pid].push_back(virtual_address);
    }
  }
  
  //close ifstream
  file_input.close();
}



/**
 * The main entry point to the simulation.
 */
int main(int argc, char** argv) {
  // TODO: implement me
  
  //Flag Handling
  FlagOptions flag_options;
  parse_flags(argc,argv, flag_options);
  
  vector<Process*> processes;
  map<int, string> process_contents;
  map<int, vector<string>> process_virtual_addresses;
  
  file_input_reader(flag_options.filename, process_contents, process_virtual_addresses);
  
  //Instantiates a new Processes from read file
  for(auto process_input : process_contents){
    istringstream ss(process_input.second);
    Process* process = Process::read_from_input(ss);
    processes.push_back(process);
  }
  
  // // print sizes
  vector<int> process_IDs;  //Stores process IDs for printing later
  for(auto i : process_contents){
     process_IDs.push_back(i.first);
  }
  int i = 0;
  for (Process* process : processes) {
    printf("Process: %i\t Size: %zu bytes\n", process_IDs.at(i), process->size());
    i++;
  }
  
  // // print virtual addresses
  cout << endl;
  for (auto i : process_virtual_addresses) {
    cout << "Process ID:\t" << i.first << endl;
    for (auto address : i.second) {
      cout << address << endl;
    }
    cout << "\n" <<std::endl;
  }

  return EXIT_SUCCESS;
}
