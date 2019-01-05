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
#include "simulation/simulation.h"


using namespace std;

void file_input_reader(Simulation& simulation, string filename, map<int, string>& process_contents, map<int, vector<string>>& process_virtual_addresses, vector<int>& pids){
  
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
    pids.push_back(pid);
    process_contents[pid] = process_image;
    current_process++;
  }
  while(file_input >> pid >> virtual_address){
    VirtualAddress virt_address = VirtualAddress::from_string(pid, virtual_address);
    simulation.virtual_address_sequence.push(virt_address);
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
  //Sim object
  Simulation simulation;
  
  //Flag Handling
  FlagOptions flag_options;
  parse_flags(argc,argv, flag_options);
  //Set flags and frames for sim
  simulation.flags = flag_options;
  simulation.max_frames = flag_options.max_frames;
  
  //Containers for storing the process information
  vector<int> pids;
  vector<Process*> processes;
  map<int, string> process_contents;
  map<int, vector<string>> process_virtual_addresses;
  
  file_input_reader(simulation, flag_options.filename, process_contents, process_virtual_addresses, pids);
  
  
  //Instantiates a new Processes from read file
  int j = 0;
  string image = "";
  string temp = "";
  for(auto process_input : process_contents){
    ifstream image_file;
    image_file.open(process_input.second);
    while(!image_file.eof()){
      getline(image_file, temp);
      image += temp;
    }
    istringstream ss(image);
    Process* process = Process::read_from_input(ss);
    process->pid = pids.at(j);
    processes.push_back(process);
    j++;
  }
  simulation.processes = processes;

  // storing ids
  vector<int> process_IDs;  //Stores process IDs for printing later
  for(auto i : process_contents){
     process_IDs.push_back(i.first);
  }
  int i = 0;
  
  //Prints the process with id and their size
  for (Process* process : processes) {
    printf("Process: %i\t Size: %zu bytes\n", process_IDs.at(i), process->size());
    i++;
  }
  cout << endl;
  
  // // print virtual addresses
  // cout << endl;
  // for (auto i : process_virtual_addresses) {
  //   cout << "Process ID:\t" << i.first << endl;
  //   for (auto address : i.second) {
  //     cout << address << endl;
  //   }
  //   cout << "\n" <<std::endl;
  // }
  
  //Actually run the simulation
  simulation.run();
  
  return EXIT_SUCCESS;
}
