/**
 * This file contains implementations for methods in the Process class.
 *
 * You'll need to add code here to make the corresponding tests pass.
 */

#include "process/process.h"

using namespace std;


Process* Process::read_from_input(std::istream& in) {
  // TODO: implement me
  
  vector<Page*> process_pages;
  string page_contents;
  char current_char;
  
  
  int i = 0;
  string contents;
  
  if(in.rdbuf()->in_avail()){
    while(in.get(current_char)){
      page_contents += current_char;
    }
    while(i < page_contents.length()){
      //Check to make sure addition isn't longer than page size
      if(i + Page::PAGE_SIZE <= page_contents.length()){
        contents = page_contents.substr(i, Page::PAGE_SIZE);
        istringstream ss(contents);
        Page* page = Page::read_from_input(ss);
        process_pages.push_back(page);
        i += Page::PAGE_SIZE;
      }
      else{
        contents = page_contents.substr(i, page_contents.length()-i);
        istringstream ss(contents);
        Page* page = Page::read_from_input(ss);
        process_pages.push_back(page);
        i += Page::PAGE_SIZE;
      }

    }
    Process* process = new Process(page_contents.length(), process_pages);
    
    return process;
  }
  return nullptr;
}


size_t Process::size() const {
  // TODO: implement me
  return num_bytes;
}

void Process::set_id(size_t id){
  pid = id;
}

bool Process::is_valid_page(size_t index) const {
  // TODO: implement me
  if(index < pages.size()){
    return true;
  }
  return false;
}

//For tests
size_t Process::get_rss() const {
  // TODO: implement me
  
  int resident_size = 0;
  for(auto row : page_table.rows){
    if(row.present == true){
      resident_size++;
    }
  }

  return resident_size;
}

//For sim
size_t Process::get_rss(int NUM_FRAMES) const {
  // TODO: implement me
  
  int resident_size = 0;
  for(auto row : page_table.rows){
    if(row.present == true){
      resident_size++;
    }
  }
  if(resident_size >= NUM_FRAMES){
    resident_size = NUM_FRAMES;
  }
  return resident_size;
}

double Process::get_fault_percent() const {
  // TODO: implement me
  
  if(memory_accesses > 0.0){
    return (((double)page_faults / memory_accesses) * 100);
  }
  else{
    return 0.0;
  }
}
