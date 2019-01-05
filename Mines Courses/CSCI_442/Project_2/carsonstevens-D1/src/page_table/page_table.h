/**
 * This file contains the definition of the PageTable class.
 */

#pragma once
#include <cstdlib>
#include <vector>


/**
 * Represents the page table for a single process.
 */
class PageTable {
// PUBLIC API METHODS
public:

  /**
   * Constructor.
   */
  PageTable(size_t num_pages) : rows(std::vector<Row>(num_pages)) {}

  /**
   * Returns the number of pages that are currently present in memory.
   */
  size_t get_present_page_count() const;

  /**
   * Returns the index of the oldest page that is also present in main memory.
   */
  size_t get_oldest_page() const;

  /**
   * Returns the index of the least recently used page that is also present in
   * main memory.
   */
  size_t get_least_recently_used_page() const;

// CLASS INSTANCE VARIABLES
public:

  /**
   * Represents a single row in the page table.
   */
  struct Row {
    /**
     * True if the page is present in main memory, or false otherwise.
     */
    bool present = false;

    /**
     * The frame containing the corresponding page, if present in memory.
     */
    size_t frame;

    /**
     * The 'virtual time' at which this page was loaded into memory.
     */
    size_t loaded_at = -1;

    /**
     * The 'virtual time' at which this page was last accessed. This is
     * completely infeasible for a real OS to track, but we're not a real OS! =)
     */
    size_t last_accessed_at = -1;
  };

  /**
   * One row for each page in the process. The page number is used as the index.
   */
  std::vector<Row> rows;
};
