/**
 * This file contains the definition of the Process class.
 */

#pragma once
#include "page/page.h"
#include "page_table/page_table.h"
#include <vector>
#include <istream>
#include <iostream>
#include <sstream>


/**
 * Represents a process in a simplified form for the memory simulation.
 */
class Process {
// PUBLIC API METHODS
public:

  /**
   * Instantiates a new Process by reading from the given istream.
   */
  static Process* read_from_input(std::istream& in);

  /**
   * Returns the total size of this process, in bytes.
   */
  size_t size() const;
  
  void set_id(size_t id);
  
  /**
   * Returns true if this process contains a page with the provided index, or
   * false otherwise.
   */
  bool is_valid_page(size_t index) const;

  /**
   * Returns the resident set size of this process.
   */
  size_t get_rss() const;
  size_t get_rss(int NUM_FRAMES) const;

  /**
   * Returns the fault rate of this process, as a percentage of all memory
   * accesses.
   */
  double get_fault_percent() const;

// PRIVATE METHODS
private:

  /**
   * Private constructor.
   */
  Process(size_t num_bytes, std::vector<Page*> pages)
      : num_bytes(num_bytes),
        pages(pages),
        page_table(PageTable(pages.size())) {}

// CLASS INSTANCE VARIABLES
public:

  /**
   * The size of this process, in bytes.
   */
  const size_t num_bytes;
  
  size_t pid;

  /**
   * The pages that constitute this process' process image.
   */
  const std::vector<Page*> pages;

  /**
   * The page table for this process.
   */
  PageTable page_table;

  /**
   * The total number of memory accesses this process performed.
   */
  size_t memory_accesses = 0;

  /**
   * The number of page faults this process experienced.
   */
  size_t page_faults = 0;
};
