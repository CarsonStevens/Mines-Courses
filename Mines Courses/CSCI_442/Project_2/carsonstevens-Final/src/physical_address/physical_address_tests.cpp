/**
 * This file contains tests for the PhysicalAddress class.
 *
 * You need to implement PhysicalAddress so that these tests pass.
 */

#include "physical_address/physical_address.h"
#include "gtest/gtest.h"
#include <bitset>
#include <sstream>

using namespace std;


// The frame and offset used below.
const size_t FRAME = bitset<32>("0010100011").to_ulong();
const size_t OFFSET = bitset<32>("111010").to_ulong();

// The corresponding address.
const string ADDRESS_STRING = "0010100011111010";


TEST(PhysicalAddress, Frame) {
  PhysicalAddress address(FRAME, OFFSET);

  ASSERT_EQ(FRAME, address.frame);
}


TEST(PhysicalAddress, Offset) {
  PhysicalAddress address(FRAME, OFFSET);

  ASSERT_EQ(OFFSET, address.offset);
}


TEST(PhysicalAddress, ToString) {
  PhysicalAddress address(FRAME, OFFSET);

  ASSERT_EQ(ADDRESS_STRING, address.to_string());
}


TEST(PhysicalAddress, OutputOperator) {
  PhysicalAddress address(FRAME, OFFSET);
  stringstream expected_output, output;

  // Form the expected output string.
  expected_output
      << ADDRESS_STRING
      << " [frame: " << FRAME
      << "; offset: " << OFFSET
      << "]";

  output << address;

  ASSERT_EQ(expected_output.str(), output.str());
}

