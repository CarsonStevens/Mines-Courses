# Project 3

A simulation of a trival OS memory manager.

Author: Carson Stevens

–   A list of all the files in your submission and what each does.
        flag_parser.cpp:        Takes the command line flags and uses getopt_long to parse and set the flags
                                for what to print and which strategy to use.
                            
        frame.cpp:              Holds the contents of a frame used in the implementing the cache.
        
        page.cpp:               Holds the information need to store a page.
        
        page_table.cpp:         Holds the pages in a table that are accessed by row.
        
        physical_address.cpp:   Holds the physical address (frame bits and offset bits) for a given virtual address
        
        process.cpp:            Holds the page table and methods need to access process attributes.
        
        simulation.cpp:         Holds the data structures needed to preform FIFO and LRU scheduling and stat printing.
        
        main.cpp:               Driver for the program. Reads the file into the needed classes and calls to run it.
    
    
–   Any unusual/interesting features in your programs.
        No added or interesting features.
    
–   Approximate number of hours you spent on the project.
        30 hours
    
–   A couple paragraphs that explain what Belady’s anomaly is and how to use your example input
    file to demonstrate its effects. Be sure to:
    
        * Define Belady’s anomaly:
            Experienced while using FIFO or non-stack based algorithms, increasing the number page 
            frames increases the number of page faults.
            
        * Give the command line invocations that should be used to demonstrate the anomaly with your program
            ./mem-sim --max-frames 10 inputs/sim_1
            ./mem-sim --max-frames 100 inputs/sim_1
            
        * Attempt to explain why the anomaly occurs
            This only occurs during FIFO because it is not a stack based algorithm. Stack based
            algorithms like LRU assign priority to pages while with algorithms like FIFO, recently 
            requested pages can remain at the bottom of the FIFO queue longer. Larger caches 
            cause pages in the cache to be popped off the FIFO queue later than smaller cache sizes.
            This leaves more irrelevant pages in the cache for longer.