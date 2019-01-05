/**
 * This file contains implementations for methods in the Page class.
 *
 * You'll need to add code here to make the corresponding tests pass.
 */

#include "page/page.h"

using namespace std;


// Ensure PAGE_SIZE is initialized.
const size_t Page::PAGE_SIZE;


Page* Page::read_from_input(std::istream& in) {
  
  vector<char> page_contents; //To store page contents
  char current_char;          //To store the current char
  
  //Check if stream is empty
  if(in.rdbuf()->in_avail()){
    //Read in while there are characters and they are less than the total page size
    while(in.get(current_char) && (page_contents.size() < PAGE_SIZE)){
      page_contents.push_back(current_char);
    }
    
    //Create new page with contents read in
    Page* page = new Page(page_contents);
    return page;
  }
  return nullptr;
}


size_t Page::size() const {
  return bytes.size();
}


bool Page::is_valid_offset(size_t offset) const {
  if(size() > 0 && offset < size()){
    return true;
  }
  return false;
}


char Page::get_byte_at_offset(size_t offset) {
  if(is_valid_offset(offset)){
    return bytes.at(offset);
  }
  return 0;
}
