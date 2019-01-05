# To build the program (which is called mem-sim by default), simply type:
#   make
#
#	To run the tests, type:
#	  make test
#
# To clean up and remove the compiled binary and other generated files, type:
#   make clean
#
# To build AND run the program, type:
#   make run
#

# The name of your binary.
NAME = mem-sim

# Flags passed to the preprocessor.
CPPFLAGS += -Wall -MMD -MP -Isrc -g -std=c++11
TEST_CPPFLAGS = $(CPPFLAGS) -isystem $(GTEST_DIR)/include

# All .cpp files.
SRCS = $(shell find src -name '*.cpp')

# All implementation sources, excluding main.cpp and test files.
IMPL_SRCS = $(shell find src \
	-name '*.cpp' \
	-not -name '*_tests.cpp' \
	-not -name 'main.cpp')

# All test files.
TEST_SRCS = $(shell find src -name '*_tests.cpp')

IMPL_OBJS = $(IMPL_SRCS:src/%.cpp=bin/%.o)
TEST_OBJS = $(TEST_SRCS:src/%.cpp=bin/%.o)
DEPS = $(SRCS:src/%.cpp=bin/%.d)

# Points to the root of Google Test, relative to where this file is.
GTEST_DIR = googletest/googletest

# All Google Test headers and source files.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

TEST_FILTER = '*'

# Build the program.
$(NAME): bin/main.o $(IMPL_OBJS)
	$(CXX) $(CPP_FLAGS) $^ -o $(NAME)

# Build and run the program.
run: $(NAME)
	./$(NAME) -v example_sim

# Build and run the unit tests.
test: bin/all_tests
	./bin/all_tests --gtest_filter=$(TEST_FILTER)

# Remove all generated files.
clean:
	rm -rf $(NAME)* bin/

# Ensure the bin/ directories are created.
$(SRCS): | bin

# Initialize the googletest submodule if needed.
$(GTEST_DIR):
	git submodule update --init

# Ensure googletest is ready to go.
$(GTEST_HEADERS): | $(GTEST_DIR)

# Mirror the directory structure of src/ under bin/.
bin:
	mkdir -p $(shell find src -type d | sed "s/src/bin/")

# Build objects.
bin/%_tests.o: src/%_tests.cpp
	$(CXX) $(TEST_CPPFLAGS) $< -c -o $@

# Build objects.
bin/%.o: src/%.cpp
	$(CXX) $(CPPFLAGS) $< -c -o $@

# Build gtest_main.a.
bin/gtest-all.o: $(GTEST_HEADERS) | bin
	$(CXX) $(TEST_CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc -o $@

bin/gtest_main.o: $(GTEST_HEADERS) | bin
	$(CXX) $(TEST_CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc -o $@

bin/gtest_main.a: bin/gtest-all.o bin/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

# Build the unit tests.
bin/all_tests: bin/gtest_main.a $(IMPL_OBJS) $(TEST_OBJS)
	$(CXX) $(TEST_CPPFLAGS) $(CXXFLAGS) -pthread $^ -o $@

# Auto dependency management.
-include $(DEPS)
