/**
 * This file contains tests for the PageTable class.
 *
 * You need to implement PageTable so that these tests pass.
 */

#include "page_table/page_table.h"
#include "gtest/gtest.h"

using namespace std;


TEST(PageTable, Constructor) {
  PageTable page_table(100);

  ASSERT_EQ(100, page_table.rows.size());

  for (int i = 0; i < 100; i++) {
    ASSERT_EQ(false, page_table.rows[i].present);
    ASSERT_EQ(-1, page_table.rows[i].loaded_at);
    ASSERT_EQ(-1, page_table.rows[i].last_accessed_at);
  }
}


TEST(PageTable, GetPresentPageCount) {
  PageTable page_table(100);

  page_table.rows[10].present = true;
  page_table.rows[42].present = true;
  page_table.rows[99].present = true;

  ASSERT_EQ(3, page_table.get_present_page_count());
}


TEST(PageTable, GetOldestPage) {
  PageTable page_table(100);

  for (size_t i = 0; i < 50; i++) {
    page_table.rows[i].loaded_at = i;
  }

  for (size_t i = 2; i < 5; i++) {
    page_table.rows[i].present = true;
  }

  // Should only include present pages, so page 2 is oldest of those present.
  ASSERT_EQ(2, page_table.get_oldest_page());
}



TEST(PageTable, GetLeastRecentlyUsedPage) {
  PageTable page_table(100);

  for (size_t i = 5; i < 50; i++) {
    page_table.rows[i].last_accessed_at = 100 - i;
  }

  for (size_t i = 10; i < 15; i++) {
    page_table.rows[i].present = true;
  }

  // Should only include present pages, so page 14 is least recently used of
  // those present.
  ASSERT_EQ(14, page_table.get_least_recently_used_page());
}
