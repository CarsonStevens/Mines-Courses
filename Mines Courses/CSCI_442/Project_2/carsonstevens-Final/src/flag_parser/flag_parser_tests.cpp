/**
 * This file contains tests for the methods declared in the flag_parser.h file.
 *
 * You need to implement these methods so that the tests in this file pass.
 */

#include "flag_parser/flag_parser.h"
#include "gtest/gtest.h"
#include <vector>
#include <string>
#include <getopt.h>
#include <cassert>

using namespace std;


bool parse_flags(initializer_list<string>, FlagOptions&);


TEST(ParseFlags, Filename) {
  FlagOptions flags;

  ASSERT_TRUE(parse_flags({"file"}, flags));
  ASSERT_EQ("file", flags.filename);
}


TEST(ParseFlags, MultipleFilenames) {
  FlagOptions flags;

  ASSERT_TRUE(parse_flags({"file_1", "file_2"}, flags));
  ASSERT_EQ("file_2", flags.filename);
}


TEST(ParseFlags, NoFilename) {
  FlagOptions flags;

  ASSERT_FALSE(parse_flags({}, flags));
}


TEST(ParseFlags, DefaultVerbose) {
  FlagOptions flags;

  ASSERT_TRUE(parse_flags({"file"}, flags));
  ASSERT_FALSE(flags.verbose);
}


TEST(ParseFlags, Verbose) {
  FlagOptions flags;

  ASSERT_TRUE(parse_flags({"file", "--verbose"}, flags));
  ASSERT_TRUE(flags.verbose);
}


TEST(ParseFlags, VerboseShort) {
  FlagOptions flags;

  ASSERT_TRUE(parse_flags({"file", "-v"}, flags));
  ASSERT_TRUE(flags.verbose);
}


TEST(ParseFlags, DefaultStrategy) {
  FlagOptions flags;

  ASSERT_TRUE(parse_flags({"file"}, flags));
  ASSERT_EQ(ReplacementStrategy::FIFO, flags.strategy);
}


TEST(ParseFlags, StrategyLru) {
  FlagOptions flags;

  ASSERT_TRUE(parse_flags({"file", "--strategy", "LRU"}, flags));
  ASSERT_EQ(ReplacementStrategy::LRU, flags.strategy);
}


TEST(ParseFlags, StrategyLruShort) {
  FlagOptions flags;

  ASSERT_TRUE(parse_flags({"file", "-s", "LRU"}, flags));
  ASSERT_EQ(ReplacementStrategy::LRU, flags.strategy);
}


TEST(ParseFlags, StrategyNoArg) {
  FlagOptions flags;

  ASSERT_FALSE(parse_flags({"file", "--strategy"}, flags));
}


TEST(ParseFlags, InvalidStrategy) {
  FlagOptions flags;

  ASSERT_FALSE(parse_flags({"file", "--strategy", "wut"}, flags));
}


TEST(ParseFlags, DefaultMaxFrames) {
  FlagOptions flags;

  ASSERT_TRUE(parse_flags({"file"}, flags));
  ASSERT_EQ(10, flags.max_frames);
}


TEST(ParseFlags, MaxFrames) {
  FlagOptions flags;

  ASSERT_TRUE(parse_flags({"file", "--max-frames", "42"}, flags));
  ASSERT_EQ(42, flags.max_frames);
}


TEST(ParseFlags, MaxFramesShort) {
  FlagOptions flags;

  ASSERT_TRUE(parse_flags({"file", "-f", "42"}, flags));
  ASSERT_EQ(42, flags.max_frames);
}


TEST(ParseFlags, MaxFramesNoArg) {
  FlagOptions flags;

  ASSERT_FALSE(parse_flags({"file", "--max-frames"}, flags));
}


TEST(ParseFlags, MaxFramesZero) {
  FlagOptions flags;

  ASSERT_FALSE(parse_flags({"file", "--max-frames", "0"}, flags));
}


TEST(ParseFlags, MaxFramesNegative) {
  FlagOptions flags;

  ASSERT_FALSE(parse_flags({"file", "--max-frames", "-42"}, flags));
}


TEST(ParseFlags, MaxFramesNonInteger) {
  FlagOptions flags;

  ASSERT_FALSE(parse_flags({"file", "--max-frames", "forty-two"}, flags));
}


bool parse_flags(initializer_list<string> arg_list, FlagOptions& flags) {
  vector<string> args(arg_list);
  vector<char*> argv;

  args.insert(args.begin(), "binary-name");

  for (size_t i = 0; i < args.size(); i++) {
    argv.push_back((char*) args[i].c_str());
  }

  int argc = argv.size();

  // argv is always null-terminated.
  argv.push_back(nullptr);

  // Reset getopts. Boo for global variables.
  optind = 0;

  testing::internal::CaptureStderr();
  bool result = parse_flags(argc, &argv[0], flags);
  testing::internal::GetCapturedStderr();

  return result;
}
