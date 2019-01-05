#Fun with Servos

import RPi.GPIO as GP
import time as t

GP.setmode(GP.BCM)
GP.setup(17,GP.OUT)
GP.setup(16,GP.OUT)

#xServo = GP.PWM(16,50)
yServo = GP.PWM(17,50)
def SetAngle(motor,angle):
    duty = angle / 18 + 2
    GP.output(17,True)
    motor.ChangeDutyCycle(duty)
    t.sleep(0.3)
    GP.output(17,False)
    motor.ChangeDutyCycle(0)
def SetAngleX(motor, angle):
    duty = angle / 180000 + 2
    GP.output(16,True)
    motor.ChangeDutyCycle(duty)
    t.sleep(0.3)
    GP.output(16,False)
    motor.ChangeDutyCycle(0)
try:
    yServo.start(0)
    #xServo.start(0)
    for y in range(60,155):
       # for x in range(0,360):
          #  SetAngleX(xServo,x)
          #  SetAngleX(xServo,0)
          #  print(x)
         #   t.sleep(0.125)
        SetAngle(yServo,y)
    
        print(y)
        t.sleep(0.01) 
    yServo.stop()
    #xServo.stop()
    GP.cleanup() 
except(KeyboardInterrupt, SystemExit): 
    print("User requested exit... bye!")
    yServo.stop()
  #  xServo.stop()
    GP.cleanup()
