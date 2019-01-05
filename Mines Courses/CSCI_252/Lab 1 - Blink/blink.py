#Author: Carson Stevens
#Date 1/25/2018
#Description: Make an LED blink; try out different combinations

import RPi.GPIO as GPIO
import time

#asks user for options on what to do.
toDo = (input("What would you like to do? \r\n Input (1) to turn on a solid LED.\r\n Input (2) to make an LED blink. \r\n Input (3) to use a button to turn on an LED.\r\n Input (4) to turn on multiple LEDs \r\n (q) to quit. ")

#Sets the GPIO mode
GPIO.setmode(GPIO.BCM)

#Taken out and code modified
#pin = int(input("What pin would you like to use"))


while(toDo != 'q'):

    if(toDo == 1):
        pin = 24
        GPIO.setup(pin, GPIO.OUT)
        while True:
            GPIO.output(24, True)

    elif (toDo == 2):
        pin = 25
        GPIO.setup(pin, GPIO.OUT)
        #amount of time between blinks
        toSleep = float(input("Input time to sleep:"))
        #amount of blinks
        blinks = int(input("How many times should it blink:"))

        counter = 0

        #blinks the chosen LED for the amount of blinks set
        while (counter < blinks):

            #code for the blinking portion
            GPIO.output(pin, True)
            time.sleep(toSleep)
            GPIO.output(pin, False)
            time.sleep(toSleep)

            #iterates the counter for the amount of total blinks
            counter = counter + 1

    elif (toDo == 3):
        pin = 26
        GPIO.setup(pin, GPIO.OUT)

        while True:
            #Causes the button to work. When pressed, it causes the LED in pin 26
            #to turn on
            if True:
                GPIO.output(26, True)

    elif (toDo == 4)
        pin = 24
        pin2 = 25
        GPIO.setup(pin, GPIO.OUT)
        GPIO.setup(pin2, GPIO.OUT)

        #Turns on the two ports with LEDs
        while True:
            GPIO.output(pin, True)
            GPIO.output(pin2, True)

    GPIO.cleanup()
    toDo = input("What would you like to do? \r\n Input (1) to turn on a solid LED.\r\n Input (2) to make an LED blink. \r\n Input (3) to use a button to turn on an LED.\r\n Input (4) to turn on multiple LEDs\r\n (q) to quit. ")


#added again incase the 'q' command is hit in the middle of execution and thus doesn't reach the
#clean up in the loop.
GPIO.cleanup()