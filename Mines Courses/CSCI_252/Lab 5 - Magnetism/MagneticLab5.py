#Author: Carson Stevens
#Date: March 8, 2018
#Description: Use a reed switch to detect a magnetic field.

#import the GPIO and time libraries
import RPi.GPIO as GPIO
import time


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


   
data = []

#area for the logic to detect high/low from reed switch and light LED
try:
   while True:
      #capture and print the input from the reed switch using GPIO.input
      output = GPIO.input(reedPin)
      data.append(output)
      
      #if the captured input is zero, then pull a LED high (True)
      if (output == 1):
         GPIO.output(LEDPin, True)


      #otherwise, pull a LED low (False)
      else:
         GPIO.output(LEDPin, False)

      #sleep for a bit, just to slow things down, how long is up to you
      time.sleep(0.25)



#capture the control c and exit cleanly
except(KeyboardInterrupt, SystemExit): 
   print("User requested exit... bye!")
   GPIO.cleanup()

for i in data:
   print(i)

