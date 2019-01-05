#Data Input Testing
import csv
import math as m
import RPi.GPIO as GP
import time as t

#servo Setup
GP.setmode(GP.BCM)
GP.setup(17,GP.OUT)
##GP.setup(16,GP.OUT)
##xServo = GP.PWM(16,50)
yServo = GP.PWM(17,50)

cave = input("What is the name of the Cave? \n This is the file where the data will be stored: ")
#changes user input to new CSV file name
data_title = cave + ".csv"

#names the data types (# of rows) to be stored
with open(data_title,'w') as newFile:
   newFileWriter = csv.writer(newFile)
   newFileWriter.writerow(['X_Coordinate', 'Y_Coordinate', 'Z_Coordinate'])

data = []
def SetAngle(motor,angle):
    duty = angle / 18 + 2
    GP.output(17,True)
    motor.ChangeDutyCycle(duty)
    t.sleep(0.3)
    GP.output(17,False)
    motor.ChangeDutyCycle(0)
    
try:
    yServo.start(0)
   # #xServo.start(0)
    for y in range(0,180,10):
        for x in range(0,360):
         #  # SetAngleX(xServo,x)
         #  # SetAngleX(xServo,0)
        #    #print(x)
        #    #t.sleep(0.125)
            phi = y*m.pi/180 #vertical angle
            theta = x*m.pi/180 #horizontal angle #from x axis
            rho = 10 #distance
            xVal = rho * m.cos(phi) * m.cos(theta)
            yVal = rho * m.cos(phi) * m.sin(theta)
            zVal = rho * m.sin(phi)
            coordinateSet = [xVal,yVal,zVal]
            print(coordinateSet)
            data.append(coordinateSet)
        SetAngle(yServo,y)
        print(y)
        t.sleep(0.01) 
    yServo.stop()
    ##xServo.stop()
    GP.cleanup() 
except(KeyboardInterrupt, SystemExit): 
    print("User requested exit... bye!")
    yServo.stop()
    ##xServo.stop()
    GP.cleanup()
    


#Write data to a CSV file path with values stored in vairable data
with open(data_title, "a", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for line in data:
        writer.writerow(line)
