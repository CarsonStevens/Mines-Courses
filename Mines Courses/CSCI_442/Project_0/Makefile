CC = g++
CFLAGS = -c -Wall -DDEBUG -g
LDFLAGS =
COMMON_SOURCES =
TARGET_SOURCES = part1.cpp
TEST_SOURCES = part2.cpp
COMMON_OBJECTS = $(COMMON_SOURCES:.cpp=.o)
TARGET_OBJECTS = $(TARGET_SOURCES:.cpp=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)
EXECUTABLE = part1
TEST_EXECUTABLE = part2
.PHONY: all target tests

all: target tests

target: $(EXECUTABLE)

tests: $(TEST_EXECUTABLE)

$(EXECUTABLE): $(COMMON_OBJECTS) $(TARGET_OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@

$(TEST_EXECUTABLE): $(COMMON_OBJECTS) $(TEST_OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
	
clean:
	rm -f *.o target tests