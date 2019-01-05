int BinPacking(vector<int> weights, int currentCapacity){    
    // weights = {w1, w2,w3,...,wn}
    // currentCapactiy = capacity of starting bin
    // Stores the current number of bins in use
    int binCount = 0;
 
    // Holds the list of remaining space in bins
    vector<int> remainingCapacity;
 
    // Add weights in order given
    for (int i = 0; i<weights.size(); i++){
        
 
        // Value of smallest capacity
        int smallest = currentCapacity + 1;
        
        // Index of the current best bin to insert into
        int bestIndex = 0;
        
        // Itterate through current bins to find the best one
        for (int j = 0; j<binCount; j++){
            if ((remainingCapacity.at(j) >= weights.at(i)) && (remainingCapacity.at(j) - weights.at(i) < smallest)){
                bestIndex = j;
                smallest = remainingCapacity.at(j) - weights.at(i);
            }
        }
 
        // If no bin can fit the newest weight, create a new bin
        if (smallest == currentCapacity + 1){
            remainingCapacity.push_back(currentCapacity - weights.at(i));
            binCount++;
        }
        else{
            // Remove new weight from Capacity of best bin
            remainingCapacity.at(bestIndex) -= weights.at(i);
        } 
    }
    return binCount;
}