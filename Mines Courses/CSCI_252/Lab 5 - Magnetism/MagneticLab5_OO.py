#Author: Carson Stevens
#Date: March 8, 2018
#Description: Use a reed switch to detect a magnetic field.

#import the GPIO and time libraries
import RPi.GPIO as GPIO
import time
import math

#############################################################
#                       GPIO Setup                          #
#############################################################


#setup two variables to hold the values for the pin numbers
#(one for LED and one for reed switch)

reedPin = 27
LEDPin = 26

#setup the pin mode for the GPIO 
GPIO.setmode(GPIO.BCM)

#turn off the warnings, this is optional
GPIO.setwarnings(False)

#setup the reed switch as an input pin
#we need to add as a third argument, GPIO.PUD_UP for pull-up resistance
GPIO.setup(reedPin, GPIO.IN, GPIO.PUD_DOWN)

#setup the LED as an output pin
GPIO.setup(LEDPin, GPIO.OUT)


#############################################################
#                       Class Setup                         #
#############################################################


class Speedometer:
   def __init__(self, elapsedTime = 0, startTime = 0, radius_cm = 0, speedMPS = 0, totalDistance = 0, pulseCount = 0):
      self.elapsedTime = elapsedTime
      self.startTime = startTime
      self.radius_cm = radius_cm
      self.speedMPS = self.calculateSpeed()
      self.pulseCount = pulseCount
      self.totalDistance = pulseCount * radius_cm
      
   def __call__(self,channel):
      self.pulseCount = self.pulseCount + 1
      self.elapsedTime = time.time() - self.startTime
      self.startTime = time.time()
      self.totalDistance = self.totalDistance + math.pi * 2 * self.radius_cm
      
   def calculateSpeed(self):
      pi = math.pi
      t = float(self.elapsedTime)
      #print(t)
      r = int(self.radius_cm)
      self.speedMPS = float(2 * pi * r)/((t)*100)
      
      return self.speedMPS
      
   def printData(self):
      print("Speed (in m/s): ", round(self.speedMPS, 3), "Total Distance(cm):", round(self.totalDistance, 3), "Pulse Count: ", self.pulseCount)
      
   def printElapsed(self):
      print(self.elapsedTime)

#############################################################
#                       Main Setup                          #
#############################################################

radius = int(input("What is the radius for your speedometer?  "))

speedometer = Speedometer(time.time(), time.time(), radius, 0, 0, 0)


#area for the logic to detect high/low from reed switch and light LED
try:
   GPIO.add_event_detect(reedPin, GPIO.FALLING, callback = speedometer, bouncetime = 25)
   while True:
      
      speedometer.calculateSpeed()
      #speedometer.printElapsed()
      speedometer.printData()
      time.sleep(1)
      

#capture the control c and exit cleanly
except(KeyboardInterrupt, SystemExit): 
   print("User requested exit... bye!")
   GPIO.cleanup()


