/**
 * This file contains the definition of the VirtualAddress class.
 */

#pragma once
#include <cstdlib>
#include <ostream>
#include <string>
#include <iostream>


/**
 * A representation of a virtual address, which includes both the page of a
 * process and the offset into that page.
 */
class VirtualAddress {
// PUBLIC CONSTANTS
public:

  /**
   * The number of bits in the virtual address that are used to represent which
   * the page to be accessed.
   */
  static const size_t PAGE_BITS = 10;

  /**
   * The number of bits in the virtual address that are used to represent the
   * offset into the corresponding page.
   */
  static const size_t OFFSET_BITS = 6;

  /**
   * The total number of bits in a virtual address.
   */
  static const size_t ADDRESS_BITS = PAGE_BITS + OFFSET_BITS;

  /**
   * A bitmask for the offset portion of the address.
   */
  static const size_t OFFSET_BITMASK = (1 << OFFSET_BITS) - 1;

  /**
   * A bitmask for the page portion of the address.
   */
  static const size_t PAGE_BITMASK = ((1 << PAGE_BITS) - 1) << OFFSET_BITS;

// PUBLIC API METHODS
public:

  /**
   * Creates a new VirtualAddress instance for the given process_id and address.
   */
  static VirtualAddress from_string(int process_id, std::string address);

  /**
   * Constructor.
   */
  VirtualAddress(int process_id, int page, int offset)
      : process_id(process_id), page(page), offset(offset) {}

  /**
   * Returns the full address as a binary string (1's and 0's).
   */
  std::string to_string() const;

// INSTANCE VARIABLES
public:

  /**
   * The ID of the process to which this address corresponds.
   */
  const int process_id;

  /**
   * The page number represented by this address.
   */
  const size_t page;

  /**
   * The offset into the specified page represented by this address.
   */
  const size_t offset;
};


/**
 * Output operator to support printing a VirtualAddress in an easy-to-read
 * format.
 */
std::ostream& operator <<(std::ostream& out, const VirtualAddress& address);
