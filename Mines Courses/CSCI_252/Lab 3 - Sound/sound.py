#Author: Carson Stevens
#Date: 2/8/2018
#Description: To experiment with the sound buzzer

#Stop...Think
#a1.    Transmitted with 1, rising
#a2.    Transmitted with 0, falling
#a3.    2 bytes
#c1.    1023

import numpy as np
import time
import spidev
import RPi.GPIO as GPIO



#ADC reader code ... inline vs include library
spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 10000000

#Buzz function is provided (uses half period method)
#This code is derived from basics physics of how sound works but to save time 
#we googled those calculations.

buzzerPin = 27
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzerPin, GPIO.OUT)

def buzz(pitch, duration):
    period = 1.0 / pitch
    delay = period / 2
    cycles = int(duration * pitch)

    for i in range(cycles):
        GPIO.output(buzzerPin, True)
        time.sleep(delay)
        GPIO.output(buzzerPin, False)
        time.sleep(delay)

    time.sleep(duration * 0.3)


def readAdc(channel):
    #Read the raw data for channel 0 using the xfer2 method, which sends AND
    #recieves depending on the clock rise/fall
    r = spi.xfer2([int('01100000',2), 15])

    #get data
    #gett 10 bit bitstring from r[0]
    s = bin(r[0])[2:].zfill(10)
    #append 8 0's to last 2 bits from r[0]
    data = int(s[8:] + '0'*8, 2) +r[1]
    return data

#loads song txt file to 2d array
song = np.loadtxt("song.txt")

#prints the song array of pitches and durations
#print(song)

while(True):
    #print(readAdc(0))
    GPIO.output(buzzerPin,True)
    #buzz(500, 0.02)
    
    for x, y in song:
        pitch= int(x)
        buzz(pitch, y)
    

    #for pitch, duration in song(x,y):
        #buzz(pitch, duration)

GPIO.cleanup()
        


 
