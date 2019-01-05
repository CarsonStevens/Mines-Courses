/**
 * This file contains implementations for methods in the Frame class.
 *
 * You'll need to add code here to make the corresponding tests pass.
 */

#include "frame/frame.h"

using namespace std;


void Frame::set_page(Process* process, size_t page_number) {
  // TODO: implement me
    this->process = process;
    this->page_number = page_number;
    
    if(page_number <= process->pages.size() && page_number >= 0){
        contents = process->pages.at(page_number);
    }
}
