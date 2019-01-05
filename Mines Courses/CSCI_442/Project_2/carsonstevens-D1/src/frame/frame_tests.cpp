/**
 * This file contains tests for the Frame class.
 *
 * You need to implement Frame so that these tests pass.
 */

#include "frame/frame.h"
#include "gtest/gtest.h"
#include <sstream>

using namespace std;


const string PROCESS_IMAGE =
    "AY3SmKknrmqdulbnXYZRtXnuQ571pA5HXAeB8qh0qR6n6yo303tdkAO9fZYWPLX3"
    "1L hUt1yI8nL4EkC0NMNm8pSKdVu";


Process* create_process() {
  istringstream in(PROCESS_IMAGE);

  return Process::read_from_input(in);
}


TEST(Frame, SetPage_PageNumber) {
  Frame frame;

  frame.set_page(create_process(), 42);

  ASSERT_EQ(42, frame.page_number);
}


TEST(Frame, SetPage_Process) {
  Frame frame;
  Process* process = create_process();

  frame.set_page(process, 42);

  ASSERT_EQ(process, frame.process);
}


TEST(Frame, SetPage_Contents) {
  Frame frame;
  Process* process = create_process();

  frame.set_page(process, 1);

  ASSERT_NE(nullptr, process);
  ASSERT_EQ(process->pages[1], frame.contents);
}
