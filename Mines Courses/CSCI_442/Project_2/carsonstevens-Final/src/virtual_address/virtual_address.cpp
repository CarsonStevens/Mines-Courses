/**
 * This file contains implementations for methods in the VirtualAddress class.
 *
 * You'll need to add code here to make the corresponding tests pass.
 */

#include "virtual_address/virtual_address.h"
#include <bitset>

using namespace std;

//CHECK if 32 or 64
VirtualAddress VirtualAddress::from_string(int process_id, string address) {
  // TODO: implement me
  
  string page_address = address.substr(0,10);  //Get first 10 chars
  string page_offset = address.substr(10);     //Get remaining chars (6)
  
  return VirtualAddress(process_id, bitset<32>(page_address).to_ulong(), bitset<32>(page_offset).to_ulong());
}


string VirtualAddress::to_string() const {
  // TODO: implement me
  string address = bitset<PAGE_BITS>(this->page).to_string() + bitset<OFFSET_BITS>(this->offset).to_string();
  return address;
}

//CHECK FORMATTING
ostream& operator <<(ostream& out, const VirtualAddress& address) {
  // TODO: implement me
  out << "PID " << address.process_id << " @ " << address.to_string() << " [page: " << address.page << "; offset: " << address.offset << "]";
  return out;
}
