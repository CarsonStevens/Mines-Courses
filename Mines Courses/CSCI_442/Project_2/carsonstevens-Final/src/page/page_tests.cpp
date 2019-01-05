/**
 * This file contains tests for the Page class.
 *
 * You need to implement Page so that these tests pass.
 */

#include "page/page.h"
#include "gtest/gtest.h"
#include <sstream>

using namespace std;


TEST(Page, ReadFromInput_EmptyStream) {
  istringstream empty_stream;

  Page* page = Page::read_from_input(empty_stream);

  // Reading from an empty stream should return null, to indicate that the
  // input stream has been fully consumed.
  ASSERT_EQ(nullptr, page);
}


TEST(Page, ReadFromInput_StreamContainsLessThanFullPage) {
  string contents = "AY3SmKknrmqdulbnXYZRtXnuQ5";
  istringstream input_stream(contents);

  Page* page = Page::read_from_input(input_stream);

  // Reading from a stream that contains less than a full page of bytes should
  // consume the entire stream.
  ASSERT_EQ(contents.length(), page->size());
}


TEST(Page, ReadFromInput_StreamContainsMoreThanFullPage) {
  istringstream input_stream(
      "AY3SmKknrmqdulbnXYZRtXnuQ571pA5HXAeB8qh0qR6n6yo303tdkAO9fZYWPLX3"
      "1L_hUt1yI8nL4EkC0NMNm8pSKdVu5m7qDvfbXdHC");

  Page* page = Page::read_from_input(input_stream);

  // Reading from a stream that contains more than a full page of bytes should
  // consume only Page::PAGE_SIZE total bytes from the stream.
  ASSERT_EQ(Page::PAGE_SIZE, page->size());
}


TEST(Page, ReadFromInput_StreamContainsNullCharacters) {
  stringstream input_stream;

  // Populate the stream with some null characters.
  input_stream << '\0' << '1' << '\0' << '2' << '\0';

  Page* page = Page::read_from_input(input_stream);

  // The stream should include every character (byte) present in the stream.
  ASSERT_EQ(5, page->size());
}


TEST(Page, ReadFromInput_StreamContainsWhitespaceCharacters) {
  string contents = "im in ur base killing ur d00dz";
  istringstream input_stream(contents);

  Page* page = Page::read_from_input(input_stream);

  // The page should have all characters from the stream, including whitespace.
  ASSERT_EQ(contents.length(), page->size());
}


TEST(Page, IsValidOffset_ValidValue) {
  string contents = "im in ur base killing ur d00dz";
  istringstream input_stream(contents);

  Page* page = Page::read_from_input(input_stream);

  ASSERT_TRUE(page->is_valid_offset(0));
  ASSERT_TRUE(page->is_valid_offset(contents.length() - 1));
}


TEST(Page, IsValidOffset_InvalidValue) {
  string contents = "im in ur base killing ur d00dz";
  istringstream input_stream(contents);

  Page* page = Page::read_from_input(input_stream);

  ASSERT_FALSE(page->is_valid_offset(contents.length()));
}


TEST(Page, IsValidOffset_InvalidValueFullPage) {
  istringstream input_stream(
      "AY3SmKknrmqdulbnXYZRtXnuQ571pA5HXAeB8qh0qR6n6yo303tdkAO9fZYWPLX3");

  Page* page = Page::read_from_input(input_stream);

  ASSERT_FALSE(page->is_valid_offset(Page::PAGE_SIZE));
}


TEST(Page, GetByteAtOffset) {
  stringstream input_stream;
  input_stream << '\0' << '1' << '\n' << ' ';

  Page* page = Page::read_from_input(input_stream);

  ASSERT_EQ(4, page->size());
  EXPECT_EQ('\0', page->get_byte_at_offset(0));
  EXPECT_EQ('1', page->get_byte_at_offset(1));
  EXPECT_EQ('\n', page->get_byte_at_offset(2));
  EXPECT_EQ(' ', page->get_byte_at_offset(3));
}
