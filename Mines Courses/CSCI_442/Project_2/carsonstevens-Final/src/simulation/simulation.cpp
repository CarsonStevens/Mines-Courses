/**
 * This file contains implementations for the methods defined in the Simulation
 * class.
 *
 * You'll probably spend a lot of your time here.
 */

#include "simulation/simulation.h"
#include <iostream>
#include "simulation/simulation.h"
#include "flag_parser/flag_parser.h"
#include "frame/frame.h"
#include "process/process.h"
#include "physical_address/physical_address.h"

using namespace std;

//Function definitions/explanations in simulation.h

int Simulation::findLRU(){
  int index = 0;
  int least_recent_time = 0;
  
  //Find frame referenced last in cache and return index to cache container
  for(int i = 0; i < cache.size(); i++){
    if(cache.at(i)->last_accessed_at < least_recent_time){
      index = i;
      least_recent_time = cache.at(i)->last_accessed_at;
    }
  }
  return index;
}

Process* Simulation::getProcess(size_t pid){
  for(Process* process : processes){
    if(process->pid == pid){
      return process;
    }
  }
  return nullptr;
}

Simulation::Simulation(){
  max_frames = flags.max_frames;
}

void Simulation::run() {
  //Handle flags
  if(flags.strategy == ReplacementStrategy::LRU){
    LRU();
  }
  else{
    FIFO();
  }
  print_stats();
}


char Simulation::perform_memory_access(const VirtualAddress& address) {
  
  //Check to see if it is in memory
  Process* process = getProcess(address.process_id);
  bool in_memory = process->page_table.rows.at(address.page).present;
  process->page_table.rows.at(address.page).last_accessed_at = event_clock;
  process->memory_accesses++;
  
  //If it isn't in memory
  if(!in_memory){
    process->page_faults++;
    handle_page_fault(process, address.page);
    current_fault = true;
  }
  //No fault; continue.
  else{
    current_fault = false;
  }
  
  //return physical_address byte (not important for sim)
  return 0;
}



void Simulation::handle_page_fault(Process* process, size_t page) {
  //Create new frame to be added
  Frame frame;
  frame.set_page(process, page);
  frame.last_accessed_at = event_clock;
  process->page_table.rows.at(page).frame = frame.page_number;
  process->page_table.rows.at(page).loaded_at = event_clock;
  
  //Current Cache isn't full
  if(cache.size() < max_frames){
      process->page_table.rows.at(page).present = true;
      cache.push_back(&frame);
  }
  //Otherwise, full and use designated replacement strategy
  else{
    int replacement_index;
    // LRU replacement
    if(flags.strategy == ReplacementStrategy::LRU){
      //Find the frame to replace
      replacement_index = findLRU();
    }
    //If not LRU, must be FIFO
    else{
      //mod used to create O(1) cyclical reference to vector index
      replacement_index = last_cache_frame_index % max_frames;
      last_cache_frame_index++;
    }
    
    //Get Least recent page_num and process id from frame
    int page_num = cache.at(replacement_index)->page_number;
    int process_num = cache.at(replacement_index)->process->pid;
    //Find process and make it no longer present
    Process* process_to_replace = getProcess(process_num);
    process_to_replace->page_table.rows.at(page_num).present = false;
    
    //Set old frame position to new frame
    cache.at(replacement_index) = &frame;
    process->page_table.rows.at(page).present = true;
  }
  
  
}

//Runs FIFO sim
void Simulation::FIFO(){

  while(!virtual_address_sequence.empty()){
    //Setup for next access
    VirtualAddress virtual_address = virtual_address_sequence.front();
    virtual_address_sequence.pop();
    Process* process = getProcess(virtual_address.process_id);
    
    //Check to make sure a process is associated with the virtual address
    if(process == nullptr){
      cout << "Segfault. EXITING" << endl;
      exit(1);
    }
    
    //Do actually memory access
    perform_memory_access(virtual_address);
    
    //Update simulation counters
    event_clock++;
    print_step(process, virtual_address);
  }
}

//Runs LRU sim
void Simulation::LRU(){
  
  while(!virtual_address_sequence.empty()){
    //Setup for next access
    VirtualAddress virtual_address = virtual_address_sequence.front();
    virtual_address_sequence.pop();
    Process* process = getProcess(virtual_address.process_id);
    
    //Check to make sure a process is associated with the virtual address
    if(process == nullptr){
      cout << "Segfault. EXITING" << endl;
      exit(1);
    }
    
    //Do actually memory access
    perform_memory_access(virtual_address);
    
    //Update simulation counters
    event_clock++;
    print_step(process, virtual_address);
  }
  
}


void Simulation::print_step(Process* process, VirtualAddress virtual_address){
  //For all outputs
  cout << virtual_address << endl;
  PhysicalAddress physical_address = PhysicalAddress(process->page_table.rows.at(virtual_address.page).frame, virtual_address.offset);
  // For verbose output
  if (flags.verbose) {
    Process* process = getProcess(virtual_address.process_id);
    if (current_fault) {
      cout << "  -> PAGE FAULT" << endl;
    }
    else {
      cout << "  -> IN MEMORY" << endl;
    }
    cout << "  -> physical address " << physical_address << endl;
    printf("  -> RSS: %zu\n\n", process->get_rss(max_frames));
  }
}

void Simulation::print_stats(){
  
  cout << endl << "DONE!" << endl << endl;
  
  int total_accesses = 0;
  int total_page_faults = 0;
 
  for (Process* process : processes) {
    std::printf("Process  %lu:  ACCESSES: %lu    FAULTS: %lu    FAULT RATE: %.2f    RSS: %lu\n", 
                 process->pid, process->memory_accesses, process->page_faults, process->get_fault_percent(), process->get_rss(max_frames));
    
    //Summing accesses and faults
    total_accesses    += process->memory_accesses;
    total_page_faults += process->page_faults;
  }
  
  
  
  // Print total memory accesses
  printf("\nTotal memory accesses: \t%d\n", total_accesses);
  
  // Print total page faults
  printf("Total page faults:\t%d\n", total_page_faults);
  
  // Print free frames remaining
  printf("Free frames remaining:\t%lu\n", (NUM_FRAMES - cache.size()));
}

