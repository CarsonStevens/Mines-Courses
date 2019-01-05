#Carson Stevens
#Lab4: Acceleration
#Description: to collect data from an accelerometer
#2/22/2018

#Description: This sample file is for Lab4 - and intended as a bare-bones structure to
#       work with the I2C and the accelerometer. Once the device is wired and RPi
#       running, this code should work as is. Students are tasked to modify this
#       code to an object-oriented solution for credit.
import smbus 
import time





class Accelerometer:
    def __init__(self, xAccel = 0, yAccel = 0, zAccel = 0):
        self.xAccel = xAccel
        self.yAccel = yAccel
        self.zAccel = zAccel

        
    def printData(self):
        print("Acceleration in x is ", self.xAccel)
        print("Acceleration in y is ", self.yAccel)
        print("Acceleration in z is ", self.zAccel)

    def printCoord(self):
        print( "(" + str(self.xAccel) + ", " + str(self.yAccel) + ", " + str(self.zAccel) +")")




# Get I2C bus - initial bus to channel 1
bus = smbus.SMBus(1) 





aData = []

for i in range (0, 10):
    #Parameters for write_byte_data
        #1. Address of the device
        #2. Communication data - active mode control register
        #3. Our data - 0 (standby mode) or 1 (active)
        bus.write_byte_data(0x1D, 0x2A, 1) 
        #time.sleep(0.5)

        #Read from the status register, real-time status register 0x00
        #Data returned will be an array
        #Contents of 7 bytes read and stored in data array represent:
        #status (ignore), MSBx, LSBx, MSBy, LSBy, MSBz, LSBz
        data = bus.read_i2c_block_data(0x1D, 0x00, 7)

        MSB_x = data[1]
        LSB_x = data[2]
        MSB_y = data[3]
        LSB_y = data[4]
        MSB_z = data[5]
        LSB_z = data[6]

        numberOfBits = 16

        xAccel =(MSB_x * 256 + LSB_x) / numberOfBits
        yAccel =(MSB_y * 256 + LSB_y) / numberOfBits
        zAccel =(MSB_z * 256 + LSB_z) / numberOfBits

        if xAccel > 2047:
            xAccel -= 4096
        if yAccel > 2047:
            yAccel -= 4096
        if zAccel > 2047:
            zAccel -= 4096

        x = Accelerometer(xAccel, yAccel, zAccel)
        aData.append(x)
        #x.printData()
        
        #put register in standbye mode
        bus.write_byte_data(0x1D, 0x2A, 0) 
        time.sleep(0.05)

        

for x in aData:
    x.printCoord()
    x.printData()
    print("")
    time.sleep(0.15)


#used in while loop collection
accelData=[]



#used to collect the data in a while loop
try:
    while True:
        #Parameters for write_byte_data
        #1. Address of the device
        #2. Communication data - active mode control register
        #3. Our data - 0 (standby mode) or 1 (active)
        bus.write_byte_data(0x1D, 0x2A, 1) 
        time.sleep(0.5)

        #Read from the status register, real-time status register 0x00
        #Data returned will be an array
        #Contents of 7 bytes read and stored in data array represent:
        #status (ignore), MSBx, LSBx, MSBy, LSBy, MSBz, LSBz
        data = bus.read_i2c_block_data(0x1D, 0x00, 7)

        MSB_x = data[1]
        LSB_x = data[2]
        MSB_y = data[3]
        LSB_y = data[4]
        MSB_z = data[5]
        LSB_z = data[6]

        numberOfBits = 16

        xAccel =(MSB_x * 256 + LSB_x) / numberOfBits
        yAccel =(MSB_y * 256 + LSB_y) / numberOfBits
        zAccel =(MSB_z * 256 + LSB_z) / numberOfBits

        if xAccel > 2047:
            xAccel -= 4096
        if yAccel > 2047:
            yAccel -= 4096
        if zAccel > 2047:
            zAccel -= 4096

        #put register in standbye mode
        bus.write_byte_data(0x1D, 0x2A, 0) 
        time.sleep(0.05)

        accelData.append([xAccel, yAccel, zAccel])
         
        #print(xAccel, yAccel, zAccel)

#capture the control c and exit cleanly
except(KeyboardInterrupt, SystemExit): 
    print("User requested exit... bye!")

print(accelData)


    
#Sample modified from https://www.controleverything.com/content/Accelorometer?sku=MMA8452Q_I2CS#tabs-0-product_tabset-2
