import RPi.GPIO as GPIO
import time
from gpiozero import AngularServo

GPIO.setmode(GPIO.BCM)
GPIO.setup(16, GPIO.OUT)
p = AngularServo(16, min_angle = 0, max_angle = 360)
#p.start(5)
try:
    for x in range (0,360):
        p.angle = x
        p.angle = 0

except(KeyboardInterrupt, SystemExit): 
    print("User requested exit... bye!")
    GPIO.cleanup()
