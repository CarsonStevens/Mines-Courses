Author: Carson Stevens

– A list of all the files in your submission and what each does:


    main.cpp:   Loads the input file, handles flags, and calls cpu algorithm
    
    CPU.h:      Contains FCFS, RR, PRIORITY, and CUSTOM algorithms. It also handles
                the printing for the "-v" and "-t" and always required stats.
    
    process.h:  Contains the data structures used to execute each algorithm. Mainly
                holds threads.
    
    thread.h:   Contains the data structures used to maintain the thread. Mainly
                manages the bursts.
    
    bursts.h:   Contains the times for IO and CPU for each burst.
    
    event.h:    Contains the information for any event that occurs. Also has a 
                formatted print for when calling the event calendar.
    
    
                
– Any unusual/interesting features in your programs:
    
    To run the program, use executable "./simulator" plus any flags.

– Approximate number of hours you spent on the project.
    
    70 hours
    
– A short essay that explains your custom CPU scheduling algorithm:

    I tried to design an algorithm that prioritized by the priority of the process and
    then using a "shortest thread next scheme". There were 4 queues involved
    (not including the process ready queue). There were 2 dedicated to the cpu portion,
    and 2 dedicated to the blocked/IO portion. First, as processes arrived, their threads were
    added to a priority queue that ordered by the priority of the process. The most 
    important processes are to be popped first. When the disbatcher is invoked, all
    the threads that have the same priority as the first thread in the queue are added
    to the "Shortest Thread Next" (STN) queue. This queue then sorted the threads by 
    their remaining time left (the sum of burst lengths remaining). The blocked queue
    has the same queues implemented. The STN mainly drives the algorithm. It is 
    preemptive. The disbatcher is invoked at the end of each quantum prempting threads
    in processs. 
    
    The quantum starts at 3, but as the algorithm runs, it changes the 
    quantum based off the size of the STN queue. When the size of the queue is small
    (less than 5), the algorithm increase the quantum by 1 each time until it reaches
    7. 7 was chosen because that seemed like the max length of the bursts provided.
    By increasing the quantum when the queue is small, it minimizes context switches
    while not increasing process wait times by that much. When the queue is larger
    than 5, the quantum is decreased by 1 each time until it reaches a quantum of 1,
    decreasing the wait time.
    
    When a process is preempted, it is sent back to the piority queue. 
    This helps to eliminate any starvation by swapping even short processes
    out so that longer ones have a chance to make it to the front of the STN queue.
    Starvation is still possible. Process with low priority fall victim to processes
    with high priority. This, although not fair, was done on purpose to make sure
    that important processes are handled first and if short, handled the fastest way
    possible. Longer processes could also get stuck in the STN queue.
    The hope was that if processes start to build in the STN queue, that by decreasing
    the quantum, more threads would have a chance to be executed. This was in hopes to
    mitigate the forseen starvation. By decreasing starvation, the hope was to 
    increase the fairness of the algorithm. Still noted that there is a bais towards
    shorter threads and threads of higher priority. Worse cases are long processes
    with low priority. They are the most susceptible to starvation and are the 
    case where the algorithm is the least fair.
    