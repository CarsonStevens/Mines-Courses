/**
 * This file contains implementations for methods in the PageTable class.
 *
 * You'll need to add code here to make the corresponding tests pass.
 */

#include "page_table/page_table.h"

using namespace std;


size_t PageTable::get_present_page_count() const {

  int page_count = 0;   //Initialize page counter
  
  //Count pages in table
  for(PageTable::Row row : rows){
    if(row.present == true){
      page_count++;
    }
  }
  
  return page_count;
}

//CHECK
size_t PageTable::get_oldest_page() const {

  for(int i = 0; i < rows.size(); i++){
    if(rows.at(i).present){
      return i;
    }
  }
  return 0;
}

//CHECK
size_t PageTable::get_least_recently_used_page() const {

  int previously_accessed = 0;
  int least_recent_index;
  
  for(int i = 0; i < rows.size(); i++){
    if(rows.at(i).present == true && (rows.at(i).last_accessed_at > previously_accessed)){
      least_recent_index = i;
    }
  }
  return least_recent_index;
}
 