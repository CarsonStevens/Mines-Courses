/**
 * This file contains implementations for methods in the flag_parser.h file.
 *
 * You'll need to add code here to make the corresponding tests pass.
 */

#include "flag_parser/flag_parser.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <getopt.h>

using namespace std;


void print_usage() {
  cout <<
      "Usage: mem-sim [options] filename\n"
      "\n"
      "Options:\n"
      "  -v, --verbose\n"
      "      Output information about every memory access.\n"
      "\n"
      "  -s, --strategy (FIFO | LRU)\n"
      "      The replacement strategy to use. One of FIFO or LRU.\n"
      "\n"
      "  -f, --max-frames <positive integer>\n"
      "      The maximum number of frames a process may be allocated.\n"
      "\n"
      "  -h --help\n"
      "      Display a help message about these flags and exit\n"
      "\n";
}


bool parse_flags(int argc, char** argv, FlagOptions& flags) {
  // TODO: implement me
  int c;

  while (1){
    static struct option long_options[] =
      {
        /* These options set a flag. */
        {"verbose",     no_argument,        0,  'v'},
        {"strategy",    optional_argument,  0,  's'},
        {"max-frames",  optional_argument,  0,  'f'},
        {"help",        no_argument,        0,  'h'},
        {0,             0,                  0,    0}
      };
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long (argc, argv, "vsfh", long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1)
      break;
    
    //Handles no input file
    if (argc == 1){
      puts("give file");
      return false;
    }
    
    switch (c)
      {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if (long_options[option_index].flag != 0)
          break;
        printf ("option %s", long_options[option_index].name);
        if (optarg)
          printf (" with arg %s", optarg);
        printf ("\n");
        break;

      case 'v':
        flags.verbose = true;
        break;

      case 's':
        if(optarg == NULL && argv[optind] == NULL){
          return false;
        }
        else if(argv[optind] != NULL){
          if(strcmp(argv[optind], "FIFO") == 0){
            flags.strategy = ReplacementStrategy::FIFO;
          }
          else if(strcmp(argv[optind], "LRU") == 0){
            flags.strategy = ReplacementStrategy::LRU;
          }
          else{
            return false; //Flag wasn't a valid input
          }
        }
        break;

      case 'f':
        if (optarg == NULL && argv[optind] == NULL) {
            flags.max_frames = 10;
            return false;
          }
        else if (argv[optind] != NULL){
          if (std::atoi(argv[optind]) > 0) {
            flags.max_frames = std::atoi(argv[optind]);
          }
          else {
            flags.max_frames = 10;
            return false;
          }
        }
        break;

      case 'h':
        print_usage();
        exit(1);
        break;

      case '?':
        /* getopt_long already printed an error message. */
        break;

      default:
        abort ();
      }
  }
  
  //Handles no input file
  if(optind == argc){
    return false;
  }  
  
  for(int file_index = optind; file_index < argc; file_index++){
    flags.filename = argv[file_index];
  }

  return true;
}
