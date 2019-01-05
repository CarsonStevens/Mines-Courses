/**
 * This file contains the definition of the Simulation class.
 *
 * You're free to modify this class however you see fit. Add new methods to help
 * keep your code modular.
 */


#include "process/process.h"
#include "virtual_address/virtual_address.h"
#include "flag_parser/flag_parser.h"
#include "frame/frame.h"
#include "physical_address/physical_address.h"

#include <cstdlib>
#include <vector>
#include <queue>

#pragma once

using namespace std;
/**
 * Class responsible for running the memory simulation.
 */
class Simulation {
// PUBLIC CONSTANTS
public:

  //Variables used to run sim
  FlagOptions flags;
  queue<VirtualAddress> virtual_address_sequence;
  vector<Frame*> cache;
  size_t event_clock = 0;
  int last_cache_frame_index = 0;
  vector<Process*> processes;
  int max_frames;
  bool current_fault = true;
  
  
  /**
   * The maximum number of frames in the simulated system (512).
   */
  static const size_t NUM_FRAMES = 1 << 9;

// PUBLIC API METHODS
public:
  //Constructor
  Simulation();
  
  //Runs FIFO sim
  void FIFO();
  
  //Runs LRU sim
  void LRU();
  
  //Returns the process from its PID
  Process* getProcess(size_t pid);
  
  //Used to print each step taken
  void print_step(Process* process, VirtualAddress virtual_address);
  
  //Used to find which frame to remove
  int findLRU();
  
  //Prints the sim stats
  void print_stats();
  
  /**
   * Runs the simulation.
   */
  void run();

// PRIVATE METHODS
private:

  /**
   * Performs a memory access for the given virtual address, translating it to
   * a physical address and loading the page into memory if needed. Returns the
   * byte at the given address.
   */
  char perform_memory_access(const VirtualAddress& address);

  /**
   * Handles a page fault, attempting to load the given page for the given
   * process into memory.
   */
  void handle_page_fault(Process* process, size_t page);

// INSTANCE VARIABLES
private:

};
