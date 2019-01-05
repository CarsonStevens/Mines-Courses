# Authors: Blue O'Brennan and Carson Stevens
# Date: May 3, 2018
# Description:  The DistoPod is a collection of sensors used to map an environment.
#               The data is then stored into a .csv file that is exported to a
#               C# Unity program that allows the user to explore the environment
#               in a 3D world.

# Import libraries/.py files
import csv
import math
import RPi.GPIO as GPIO
import time
import smbus
import statistics
from quaternion import Quaternion
from numpy.linalg import norm
from madgwickahrs import MadgwickAHRS
from mpu9250 import MPU
from Coordinate import Coordinate
from lidar_lite import Lidar_Lite


# Servo Setup
GPIO.setmode(GPIO.BCM)
x_servo_pin = 16
y_servo_pin = 18
GPIO.setup(y_servo_pin,GPIO.OUT)
GPIO.setup(x_servo_pin,GPIO.OUT)
xServo = GPIO.PWM(x_servo_pin,50)
yServo = GPIO.PWM(y_servo_pin,50)

# Laser Setup
laser_pin = 17
GPIO.setup(laser_pin,GPIO.OUT)
GPIO.output(laser_pin,False)

# Lidar Setup
lidar_bus = smbus.SMBus(1)
lidar_address = 0x62
lidar = Lidar_Lite()
connected = lidar.connect(1)
if connected < -1:
    print("Lidar not connected")
    
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
    
    # DEBUG
    # print("Acceleration to velocity x:")
    # for i in ax:
    #     print(i)
    # print("Accerlation to velcoity y:")
    # for i in ay:
    #     print(i)
    
    
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
    
    
# Returns Euler angles -> theta and phi offsets in degrees
#   Roll, Pitch, and Yaw must be pasted in rad
def AngleOffset(roll, pitch, yaw):
    
    # Euler Angle conversion to Cartesian 
    x = math.cos(yaw)*math.cos(pitch)
    y = math.sin(yaw)*math.cos(pitch)
    z = math.cos(pitch) *math.sin(roll)
    
    # Direction of cosines of a vector to find theta and phi (in degrees)
    theta = math.acos(x/(math.sqrt(math.pow(x,2) + math.pow(y,2) + math.pow(z,2))))*180/math.pi
    phi = math.acos(y/(math.sqrt(math.pow(x,2) + math.pow(y,2) + math.pow(z,2))))*180/math.pi
    
    # Returns in Degrees
    return ([theta,phi])
    

# Function to Set Servo to correct angle
def SetAngle(motor,angle,pin):
    duty = angle / 18 + 2
    GPIO.output(pin,True)
    motor.ChangeDutyCycle(duty)
    time.sleep(0.2)


# Function to get one offset
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

# Function used only during calibration to find the calibration offsets    
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
    angle_displacements[0] = angle_displacements[0]
    angle_displacements[1] = 90 - (angle_displacements[1])
    
    
    return([displacement_X, displacement_Y, displacement_Z, angle_displacements[0], angle_displacements[1]])


################################################################################


#                               Start of Calibration                           #


################################################################################

print("\n\nStarting Calibration.....\n\n")

# Starts Calibration
Calibration_MPU = MPU()
Calibration_AHRS_Filter = MadgwickAHRS()
calibration_time = 5
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

#Offsets used for individual gyro sensor-axis (rad -> deg)
roll_calibration_offset = Sum(calibration_offset_data[1][0])/len(calibration_offset_data[1][0])* 180/math.pi
pitch_calibration_offset = Sum(calibration_offset_data[1][1])/len(calibration_offset_data[1][1])* 180/math.pi
yaw_calibration_offset = Sum(calibration_offset_data[1][2])/len(calibration_offset_data[1][2])* 180/math.pi

#Calibrates all offsets (Converts theta and phi into degs)
calibration_offset = CalibrationCalc(calibration_offset_data)

#Container assignment for main
x_calibration_offset = calibration_offset[0]
y_calibration_offset = calibration_offset[1]
z_calibration_offset = calibration_offset[2] 
theta_calibration_offset = calibration_offset[3]
phi_calibration_offset = calibration_offset[4]

#Prints the Calibration Data to the user
print("\nCalibration Data:\n")
print("X Accel Calibration Offset:\t", x_calibration_offset, "meters")
print("Y Accel Calibration Offset:\t", y_calibration_offset, "meters")
print("Z Accel Calibration Offset:\t", z_calibration_offset, "meters")
print("Theta Calibration Offset:\t", theta_calibration_offset, "degrees")
print("Phi Calibration Offset:\t", phi_calibration_offset, "degrees")
print("\nEnd of Calibration Phase\n\n")

################################################################################


#                               End of Calibration                             #


################################################################################

# Define main's local offset variables (The origin for the Coordinate System)
x_offset = 0
y_offset = 0
z_offset = 0
theta_offset = 0
phi_offset = 0

    
################################################################################


#                               Start of Program                               #


################################################################################

# Get user input for name of cave/file
cave = input("What is the name of the Cave? (Name must be entered with no spaces)\nThis is the file where the data will be stored: ")
# Changes user input to new CSV file name
data_title = cave + ".csv"


# Gets the users resolution input for the scan (should be between 1-180).
print("\n")
# Theta direction
x_resolution = int(input("What is the resolution (in degrees) you want for the pan direction?\t"))
# Phi direction
y_resolution = int(input("What is the resolution (in degrees) you want for the tilt direction?\t"))
print("\n")

# Start Cave Scan(s)
scan = True;
while(scan == True):
    
    # Start Servos
    yServo.start(0)
    xServo.start(0)
    
    # Turn on laser
    GPIO.output(laser_pin,True)
    
    # Starts timer for scan execution
    start = time.time()
    
    for x in range(0,180,x_resolution):
        
        # xServo.position(0,x)      # Going to be used with now Servo Controller
        
        # Time for servo to move
        time.sleep(0.2)
        
        angle = x/1.8 + 35
        # Set pan (x) servo angle
        SetAngle(xServo,angle, x_servo_pin)
        
        for y in range(0,180,y_resolution):
            # yServo.position(1,y)      # Going to be used with new Servo Controller
            
            # Conversion for PWM duty cycle
            angle_y = y/1.8 + 35
            # Set tilt (y) servo angle
            SetAngle(yServo,angle_y, y_servo_pin)
            
            # Set rho equal to lidar distance
            rho = lidar.getDistance()/100
            
            # Convert phi and theta into rads
            # Vertical angle
            phi = y*math.pi/180
            # Horizontal angle from x-axis
            theta = x*math.pi/180 
            
            # Create Coordinate Object to store data
            xyz_coordinate = Coordinate(rho, theta, phi, x_offset, y_offset, z_offset, theta_offset, phi_offset)
            
            # Convert data into (x,y,z)
            xyz_coordinate.parameterize()
            
            # Add new Coordinate to data[]
            data.append(xyz_coordinate.get_xyz())
    
    # End time of Scan
    end = time.time()
    
    # Prints total time of a scan
    print(start - end)
    
    # Turn off laser
    GPIO.output(laser_pin,False)
    
    # Stop Servos
    yServo.stop()
    xServo.stop()
    
    
################################################################################

#                         End of Single Scan                                   #

################################################################################
    
    
    more_scans = input("Would you like to move the sensor and continue scanning more? \n If so, type 'yes'.\n Keyboard Interrupt to stop moving:\t")
    if(more_scans != "yes"):
        scan = False
        break
    
################################################################################

#                         Start Sensor Move                                    #

################################################################################

    # Create MPU Object and define move function
    mpu = MPU()
    
    # Defines the Filter object to filter the data
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
            g[2] = g[2] - yaw_calibration_offset
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

            # Append data to contain used in Offset Function
            offset_data.append([a,g,m])

    # Interrupt to stop moving the sensor
    except KeyboardInterrupt:
        print ("\nDone moving!\n")
        

    # End time of moving the sensor
    move_time = time.time() - move_time
    print("Total move time:\t", move_time)
    
    # Returns [ax_offset, ay_offset, az_offset, gx_offset, gy_offset, gz_offset]
    # Note that the gryo offsets are in rads here
    all_offsets = Offset(move_time, offset_data)
    
    # Update all the offsets
    x_offset = x_offset + all_offsets[0]
    y_offset = y_offset + all_offsets[1]
    z_offset = z_offset + all_offsets[2]
    print("(x,y,z) displacement:\t (", x_offset, ",", y_offset, ",", z_offset, ")")
    
    
    # Turn x,y,z gyroscopic offsets to phi and theta offsets
    # Angle Offset Math (rad -> degrees)
    angle_offsets = AngleOffset(all_offsets[3], all_offsets[4], all_offsets[5])
    theta_offset = theta_offset + angle_offsets[0]
    phi_offset = phi_offset + (90-angle_offsets[1])
    print("(theta,phi) displacement:\t (", theta_offset, ",", phi_offset, ")\n\n")
        
################################################################################
#                                                                              #
#                           End Sensor Move                                    #
#               Returns to top of loop to scan again                           #
################################################################################
        
# Cleanup GPIO
GPIO.cleanup() 


################################################################################

#                          Write Data to File                                  #

################################################################################

print("\nWriting Data to", data_title, "....")
    
# Names the data types (# of rows) to be stored
with open(data_title,'w') as newFile:
   newFileWriter = csv.writer(newFile)
   # Defines the titles of the columns
   newFileWriter.writerow(['X_Coordinate', 'Y_Coordinate', 'Z_Coordinate'])
   
# Write data to a CSV file path with values stored in vairable data
with open(data_title, "a", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for line in data:
        # Writes list to row in file
        writer.writerow(line)
        
print("\nDone writing file. Exiting program")