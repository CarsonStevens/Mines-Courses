/**
 * This file contains the definition of the Page class.
 */

#pragma once
#include "virtual_address/virtual_address.h"
#include <cstdlib>
#include <istream>
#include <iostream>
#include <vector>


/**
 * Represents a single page of a process.
 */
class Page {
// PUBLIC CONSTANTS
public:

  /**
   * Represents the maximum allowable size of a page, in bytes.
   */
  static const size_t PAGE_SIZE = 1 << VirtualAddress::OFFSET_BITS;

// PUBLIC API METHODS
public:

  /**
   * Consumes from the given istream at most PAGE_SIZE bytes and returns a new
   * Page instance containing those bytes. If the istream is empty, NULL is
   * returned instead.
   */
  static Page* read_from_input(std::istream& in);

  /**
   * Returns the number of bytes present in this page, which should always be a
   * number in the range [0, PAGE_SIZE).
   */
  size_t size() const;

  /**
   * Returns true if the provided offset corresponds to a valid byte in this
   * page, or false otherwise. Should only return true when the page is not
   * empty and the offset is less than the size of the page.
   */
  bool is_valid_offset(size_t offset) const;

  /**
   * Returns a reference to the byte at the given offset. Should only be called
   * if is_valid_offset returns true for the offset.
   */
  char get_byte_at_offset(size_t offset);

// PRIVATE METHODS
private:

  /**
   * Private constructor.
   */
  Page(std::vector<char> bytes) : bytes(bytes) {}

// CLASS INSTANCE VARIABLES
private:

  /**
   * The bytes this page contains.
   */
  std::vector<char> bytes;
};
