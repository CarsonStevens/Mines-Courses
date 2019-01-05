#Author: Carson Stevens
#Date: 2/1/2018
#Description: To set up the analog to Digital COnverter chip and see how to use
#             sensors and gather data

import reader
import numpy as np
import time
from collections import OrderedDict
import json
import matplotlib.pyplot as plt

#creates the array to store the values
myValues = np.zeros(500)


#creates dictionary
data = {}

#dictionary for alternative method
_data = {}

#Defines x and y points for graph later
graphX = []
graphY = []


#for loop to get 500 values
for x in range(0, 500):

    #reads in the input and stores it into the variable value
    value = (reader.read(0))
    value = round(value, 2)
    value = value * 100
    
    #prints the value
    #print(value)

    #stores the value into an array
    myValues[x] = value

    #adds the points to the graph arrays
    graphX.append(x)
    graphY.append(value)
    
    #add values to dictionary
    data[("Reading " + str(x+1))] = str(value) + "%"

    #adds values to dictionary if key is int
    _data[(x+1)] = str(value) + "%"
    
    #makes a 0.05s delay between inputs
    time.sleep(0.05)


#Prints the data dictionary, but not numerically because key is string
print("Non-numerical dump because key is string value")
print(json.dumps(data, indent=4))
print("Numerical dump, but lacking the word reading before each entry")
print(json.dumps(_data, indent=4))


#Used to sort the dictionary if data by key for ints
sortedList = sorted(_data.items(), key=lambda x: x[0])
_data = OrderedDict(sortedList)

#Used to print the int dictionary in order and good formatting if key is int
print("Method to print values not using json dumps, but for loop")
for i in sorted(_data):
    print ("Reading: ", repr(i).ljust(4) , "Value: " , repr(_data[i]).rjust(5))


#Creates graph
plt.plot(graphX,graphY)

#sets domain for x and y on graph
plt.xlim(0,500)
plt.ylim(0,100)

#gives the graph a title
plt.title("Light Values")

#names the axises
plt.xlabel("Reading #")
plt.ylabel("Light Value")

#prints the graph
plt.show()

