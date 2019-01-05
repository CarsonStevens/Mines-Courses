/**
 * This file contains the definition of the Frame class.
 */

#pragma once
#include "page/page.h"
#include "process/process.h"


/**
 * A convenience for representing a frame of memory.
 */
class Frame {
// PUBLIC API METHODS
public:

  /**
   * Loads the specified page in the given process into this frame.
   */
  void set_page(Process* process, size_t page_number);

// CLASS INSTANCE VARIABLES
public:

  /**
   * A pointer to the page that this frame holds, if any. This is done rather
   * than copying the contents from the page to the frame directly, since this
   * is easier and accomplishes the same thing. =)
   */
  Page* contents = nullptr;

  /**
   * The page number this frame holds (pretend that this is stored in some OS
   * data structure).
   */
  size_t page_number;

  /**
   * The process corresponding to the page this frame holds (pretend that this
   * is stored in some OS data structure).
   */
  Process* process = nullptr;
};
