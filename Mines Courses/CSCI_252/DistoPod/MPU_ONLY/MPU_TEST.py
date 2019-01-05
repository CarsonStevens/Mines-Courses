################################################################################
# Tasks still left to do are marked by: TODO                                   #
################################################################################

# Import libraries/.py files
import csv
import math
import RPi.GPIO as GPIO
import time
import smbus
import statistics
import warnings
import numpy as np
from numpy.linalg import norm
from quaternion import Quaternion
from madgwickahrs import MadgwickAHRS
from mpu9250 import MPU


################################################################################
# TODO:
#   Define sensor ports on pi wedge
################################################################################

# Servo Setup
GPIO.setmode(GPIO.BCM)

    
# Define containers for Calibration
accelX_cal = []
accelY_cal = []
accelZ_cal = []
gyroX_cal = []
gyroY_cal = []
gyroZ_cal = []
    

# Define x,y,z,theta,phi
x_coordinate = 0
y_coordinate = 0
z_coordinate = 0
theta = 0
phi = 0

# Define container where Coordinate data is stored
data = []


################################################################################
# Function Definitions                                                         #
################################################################################


# Sums values in a list
def Sum(data):
    offset = 0
    for i in data:
        offset = offset + i
    return offset
    
#Check the domain of the Euler inputs
def Domain(angle, domain):
    while (angle >= domain):
        angle -= domain
    return angle


# Function to get all offsets
def Offset(total_time, offset_data):
    
    # Heigharchy of offset_data is offset_data -> [a,g,m] -> [x,y,z]
    ax = []
    ay = []
    az = []
    gx = []
    gy = []
    gz = []
    
    for i in offset_data:
        # Acceleration Data
        ax.append(i[0][0])
        ay.append(i[0][1])
        az.append(i[0][2])
        # Gyro Data
        gx.append(i[1][0])
        gy.append(i[1][1])
        gz.append(i[1][2])
    
    #Integrate once for velocity
    print("Acceleration to velocity x:")
    for i in ax:
        print(i)
    print("Accerlation to velcoity y:")
    for i in ay:
        print(i)
    #Integrate once for velocity
    vx_offset = MidPointReimann(total_time, ax)
    vy_offset = MidPointReimann(total_time, ay)
    vz_offset = MidPointReimann(total_time, az)
    
    # Integrate twice for displacement    
    ax_offset = MidPointReimann(total_time, vx_offset)
    ay_offset = MidPointReimann(total_time, vy_offset)
    az_offset = MidPointReimann(total_time, vz_offset)
    
    #Integrate deg/s to get deg at each time
    gx_offset = MidPointReimann(total_time, gx)
    gy_offset = MidPointReimann(total_time, gy)
    gz_offset = MidPointReimann(total_time, gz)

    x = Sum(ax_offset)
    y = Sum(ay_offset)
    z = Sum(az_offset)
    roll = Sum(gx_offset)
    pitch = Sum(gy_offset)
    yaw = Sum(gz_offset)
    
    return [x,y,z,roll,pitch,yaw]
    
    
def AngleOffset(roll, pitch, yaw):
    
    #Euler Angle conversion to Cartesian 
    x = math.cos(yaw)*math.cos(pitch)
    y = math.sin(yaw)*math.cos(pitch)
    z = math.cos(pitch) *math.sin(roll)
    
    #Direction of cosines of a vector to find theta and phi
    theta = math.acos(x/(math.sqrt(math.pow(x,2) + math.pow(y,2) + math.pow(z,2))))
    phi = math.acos(y/(math.sqrt(math.pow(x,2) + math.pow(y,2) + math.pow(z,2))))
    
    return ([theta,phi])
    

# Function to Set Servo to correct angle
def SetAngle(motor,angle,pin):
    duty = angle / 18 + 2
    GPIO.output(pin,True)
    motor.ChangeDutyCycle(duty)
    time.sleep(0.2)


#Function to get one offset
def MidPointReimann(total_time, offset_data):
    
    # The collection of distance/angle offset
    offset = []
        
    # Gives number of data points in offset_data
    num_data = len(offset_data)
    
    # Should probably change to imu sampling rate*****
    # Gives the width of each rectangle for the reimann sum
    width = total_time/num_data
    
    index = 0
    while (index < len(offset_data)-1):
        left_height = a[index]
        right_height = a[index + 1]
        height = (left_height + right_height)/2
        #print("Trying to set offset[",index,"] to ", height*width)
        offset.append(height * width)
        index = index + 1
    return offset
    
def CalibrationCalc(offset_data):
    accel_num = len(offset_data[0])
    gyro_num = len(offset_data[1])
    
    displacement_X = Sum(offset_data[0][0])/accel_num
    displacement_Y = Sum(offset_data[0][1])/accel_num
    displacement_Z = Sum(offset_data[0][2])/accel_num
    
    displacement_roll = Sum(offset_data[1][0])/gyro_num
    displacement_pitch = Sum(offset_data[1][1])/gyro_num
    displacement_yaw = Sum(offset_data[1][2])/gyro_num
    
    angle_displacements = AngleOffset(displacement_roll, displacement_pitch, displacement_yaw)
    
    # Convert from rad from filter back to deg
    angle_displacements[0] = angle_displacements[0] * 180/math.pi
    angle_displacements[1] = angle_displacements[1] * 180/math.pi
    
    
    return([displacement_X, displacement_Y, displacement_Z, angle_displacements[0], angle_displacements[1]])


################################################################################


#                               Start of Calibration                           #


################################################################################


# Starts Calibration
Calibration_MPU = MPU()
Calibration_AHRS_Filter = MadgwickAHRS()
calibration_time = 10
Calibration_Timer = time.time() + calibration_time
calibration_offset_data = []
calibration_accel = []
calibration_gyro = []

while(time.time() <= Calibration_Timer):
    a = Calibration_MPU.accel
    g = Calibration_MPU.gyro
    m = Calibration_MPU.mag

    # Convert from deg to rad for filter
    g[0] = g[0] * math.pi/180
    g[1] = g[1] * math.pi/180
    g[2] = g[2] * math.pi/180
    
    # Apply AHRS Filter
    g = Calibration_AHRS_Filter.update(g,a,m)
    
    
    calibration_accel.append([a[0],a[1],a[2]])
    calibration_gyro.append([g[0],g[1],g[2]])

calibration_offset_data.append(calibration_accel)
calibration_offset_data.append(calibration_gyro)

#Method 1
#calibration_offset = Offset(calibration_time,calibration_offset_data)

#Method 2
#Converts from rads back to degs
roll_calibration_offset = Sum(calibration_offset_data[1][0])/len(calibration_offset_data[1][0]) * 180 / math.pi
pitch_calibration_offset = Sum(calibration_offset_data[1][1])/len(calibration_offset_data[1][1]) * 180 / math.pi
yaw_calibration_offset = Sum(calibration_offset_data[1][2])/len(calibration_offset_data[1][2]) * 180 /math.pi


calibration_offset = CalibrationCalc(calibration_offset_data)

x_calibration_offset = calibration_offset[0]
y_calibration_offset = calibration_offset[1]
z_calibration_offset = calibration_offset[2] 
theta_calibration_offset = calibration_offset[3]
phi_calibration_offset = calibration_offset[4]

print("X Accel Calibration Offset:\t", x_calibration_offset)
print("Y Accel Calibration Offset:\t", y_calibration_offset)
print("Z Accel Calibration Offset:\t", z_calibration_offset)
print("Theta Calibration Offset:\t", theta_calibration_offset)
print("Phi Calibration Offset:\t", phi_calibration_offset)


################################################################################


#                               End of Calibration                             #


################################################################################

# Define main's local offset variables
x_offset = 0
y_offset = 0
z_offset = 0
theta_offset = 0
phi_offset = 0

    
################################################################################


#                               Start of Program                               #

######################## ANY USE OF MUST FUNCTIONS MUST BE IN RAD###############
################################################################################

# Start Cave Scan(s)
scan = True;
while(scan == True):

    more_scans = input("Would you like to continue scanning more? \n If so, type 'yes':\t")
    if(more_scans != "yes"):
        scan = False
    
################################################################################

#                         Start Sensor Move                                    #

################################################################################

    # Create MPU Object and define move function
    mpu = MPU()
    AHRS_Filter = MadgwickAHRS() 
    # Store mpu data
    offset_data = []

    # Start time of moving the sensor
    move_time = time.time()

    try:
        while True:
            # Storage of a,g,m is [x,y,z]
            a = mpu.accel
            a[0] = a[0] - x_calibration_offset
            a[1] = a[1] - y_calibration_offset
            a[2] = a[2] - z_calibration_offset
            print("Accel: {:.3f} {:.3f} {:.3f} mg".format(*a))
            g = mpu.gyro
            g[0] = g[0] - roll_calibration_offset
            g[1] = g[1] - pitch_calibration_offset
            g[2] = g[2] - pitch_calibration_offset
            print("Gyro: {:.3f} {:.3f} {:.3f} dps".format(*g))
            m = mpu.mag
            #print("Magnet: {:.3f} {:.3f} {:.3f} mT".format(*m))
            # t = mpu.temp
            # print 'Temperature: {:.3f} C'.format(t)
            #time.sleep(0.5)

            # Convert from deg to rad for filter
            g[0] = g[0] * math.pi/180
            g[1] = g[1] * math.pi/180
            g[2] = g[2] * math.pi/180

            
            # Apply AHRS Filter
            g = AHRS_Filter.update(g,a,m)

            # Convert from rad from filter back to degree
            g[0] = g[0] * 180/math.pi
            g[1] = g[1] * 180/math.pi
            g[2] = g[2] * 180/math.pi

            # Append data to contain used in Offset Function
            offset_data.append([a,g,m])


    except KeyboardInterrupt:
        print ("\nDone moving!\n")
        

    # End time of moving the sensor
    move_time = time.time() - move_time
    print("Total move time:\t", move_time)
    
    # Returns [ax_offset, ay_offset, az_offset, gx_offset, gy_offset, gz_offset]
    all_offsets = Offset(move_time, offset_data)
    
    # Update all the offsets
    x_offset = x_offset + all_offsets[0]
    y_offset = y_offset + all_offsets[1]
    z_offset = z_offset + all_offsets[2]
    print("(x,y,z) displacement:\t (", x_offset, ",", y_offset, ",", z_offset, ")")
    
    
    # Turn x,y,z gyroscopic offsets to phi and theta offsets
    angle_offsets = AngleOffset(all_offsets[3], all_offsets[4], all_offsets[5])
    theta_offset = theta_offset + angle_offsets[0]
    phi_offset = phi_offset + angle_offsets[1]
    print("(theta,phi) displacement:\t (", theta_offset, ",", phi_offset, ")")
        
################################################################################
#                                                                              #
#                           End Sensor Move                                    #
#               Returns to top of loop to scan again                           #
################################################################################
 
        
# Cleanup GPIO
GPIO.cleanup() 
