/**
 * This file contains the definition of the PhysicalAddress class.
 */

#pragma once
#include "virtual_address/virtual_address.h"
#include <cstdlib>
#include <ostream>


/**
 * A representation of a physical address, which corresponds directly to a spot
 * in main memory.
 */
class PhysicalAddress {
// PUBLIC CONSTANTS
public:

  /**
   * The number of bits in the physical address that are used to represent the
   * frame in main memory.
   */
  static const size_t FRAME_BITS = 10;

  /**
   * The number of bits in the physical address that are used to represent the
   * offset into the corresponding frame.
   */
  static const size_t OFFSET_BITS = VirtualAddress::OFFSET_BITS;

  /**
   * The total number of bits in a physical address.
   */
  static const size_t ADDRESS_BITS = FRAME_BITS + OFFSET_BITS;

// PUBLIC API METHODS
public:

  /**
   * Constructor.
   */
  PhysicalAddress(int frame, int offset) : frame(frame), offset(offset) {}

  /**
   * Returns the full address as a binary string (1's and 0's).
   */
  std::string to_string() const;

// INSTANCE VARIABLES
public:

  /**
   * The frame represented by this address.
   */
  const int frame;

  /**
   * The offset into the specified frame represented by this address.
   */
  const int offset;
};


/**
 * Output operator to support printing a PhysicalAddress in an easy-to-read
 * format.
 */
std::ostream& operator <<(std::ostream& out, const PhysicalAddress& address);
