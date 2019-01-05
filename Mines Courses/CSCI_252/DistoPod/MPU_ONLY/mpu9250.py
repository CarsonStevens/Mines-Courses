# try:
# 	import smbus2 as smbus
# except ImportError:
# 	print('WARNING: Using fake hardware')
# 	from smbusHelp import smbus
# 	# from fake_rpi import smbus
import smbus
from time import sleep
import struct
import math

# Todo:
# - replace all read* with the block read?

################################
# MPU9250
################################
MPU9250_ADDRESS = 0x68
AK8963_ADDRESS  = 0x0C
DEVICE_ID       = 0x71
WHO_AM_I        = 0x75
PWR_MGMT_1      = 0x6B
INT_PIN_CFG     = 0x37
INT_ENABLE      = 0x38
# --- Accel ------------------
ACCEL_DATA    = 0x3B
ACCEL_CONFIG  = 0x1C
ACCEL_CONFIG2 = 0x1D
ACCEL_2G      = 0x00
ACCEL_4G      = (0x01 << 3)
ACCEL_8G      = (0x02 << 3)
ACCEL_16G     = (0x03 << 3)
# --- Temp --------------------
TEMP_DATA = 0x41
# --- Gyro --------------------
GYRO_DATA    = 0x43
GYRO_CONFIG  = 0x1B
GYRO_250DPS  = 0x00
GYRO_500DPS  = (0x01 << 3)
GYRO_1000DPS = (0x02 << 3)
GYRO_2000DPS = (0x03 << 3)

# --- AK8963 ------------------
MAGNET_DATA  = 0x03
AK_DEVICE_ID = 0x48
AK_WHO_AM_I  = 0x00
AK8963_8HZ   = 0x02
AK8963_100HZ = 0x06
AK8963_14BIT = 0x00
AK8963_16BIT = (0x01 << 4)
AK8963_CNTL1 = 0x0A
AK8963_CNTL2 = 0x0B
AK8963_ASAX  = 0x10
AK8963_ST1   = 0x02

SAMPLE_RATE  = 0x25

class MPU(object):
	def __init__(self, bus=1):
	
		# Setup the IMU
		# reg 0x25: SAMPLE_RATE= Internal_Sample_Rate / (1 + SMPLRT_DIV)
		# reg 0x29: [2:0] A_DLPFCFG Accelerometer low pass filter setting
		# 	ACCEL_FCHOICE 1
		# 	A_DLPF_CFG 4
		# 	gives BW of 20 Hz
		# reg 0x35: FIFO disabled default - not sure i want this ... just give me current reading
		# might include an interface where you can change these with a dictionary:
		# 	setup = {
		# 		ACCEL_CONFIG: ACCEL_4G,
		# 		GYRO_CONFIG: AK8963_14BIT | AK8963_100HZ
		# 	}
		
		self.bus = smbus.SMBus(bus)

		# let's double check we have the correct device address
		ret = self.read8(MPU9250_ADDRESS, WHO_AM_I)
		if ret is not DEVICE_ID:
			raise Exception('MPU9250: init failed to find device')

		self.write(MPU9250_ADDRESS, PWR_MGMT_1, 0x00)  # turn sleep mode off
		sleep(0.2)
		self.bus.write_byte_data(MPU9250_ADDRESS, PWR_MGMT_1, 0x01)  # auto select clock source
		self.write(MPU9250_ADDRESS, ACCEL_CONFIG, ACCEL_2G)
		self.write(MPU9250_ADDRESS, GYRO_CONFIG, GYRO_250DPS)

		# You have to enable the other chips to join the I2C network
		# then you should see 0x68 and 0x0c from:
		# sudo i2cdetect -y 1
		self.write(MPU9250_ADDRESS, INT_PIN_CFG, 0x22)
		self.write(MPU9250_ADDRESS, INT_ENABLE, 0x01)
		sleep(0.1)

		ret = self.read8(AK8963_ADDRESS, AK_WHO_AM_I)
		if ret is not AK_DEVICE_ID:
			raise Exception('AK8963: init failed to find device')
		self.write(AK8963_ADDRESS, AK8963_CNTL1, (AK8963_16BIT | AK8963_8HZ))

		# all 3 are set to 16b or 14b readings, we have take half, so one bit is
		# removed 16 -> 15 or 13 -> 14
		self.alsb = 2 / 2**15
		self.glsb = 250 / 2**15
		self.mlsb = 4800 / 2**15

		# i think i can do this???
		# self.convv = struct.Struct('<hhh')

	def __del__(self):
		self.bus.close()

	def write(self, address, register, value):
		self.bus.write_byte_data(address, register, value)

	def read8(self, address, register):
		data = self.bus.read_byte_data(address, register)
		return data

	def read16(self, address, register):
		data = self.bus.read_i2c_block_data(address, register, 2)
		return self.conv(data[0], data[1])

	def read_xyz(self, address, register, lsb):
		
		# Reads x, y, and z axes at once and turns them into a tuple.
		
		# data is MSB, LSB, MSB, LSB ...
		data = self.bus.read_i2c_block_data(address, register, 6)

		# data = []
		# for i in range(6):
		# 	data.append(self.read8(address, register + i))

		x = self.conv(data[0], data[1]) * lsb
		y = self.conv(data[2], data[3]) * lsb
		z = self.conv(data[4], data[5]) * lsb

        # print('>> data', data)
		# ans = self.convv.unpack(*data)
		# ans = struct.unpack('<hhh', data)[0]
		# print('func', x, y, z)
		# print('struct', ans)

		return [x, y, z]

	def conv(self, msb, lsb):
		
		# http://stackoverflow.com/questions/26641664/twos-complement-of-hex-number-in-python
		
		value = lsb | (msb << 8)
		# if value >= (1 << 15):
		# 	value -= (1 << 15)
		# print(lsb, msb, value)
		return value
		
# 	def Update(self,a,g,m):
		
# import warnings
# import numpy as np
# from numpy.linalg import norm

#     def update(self, gyroscope, accelerometer, magnetometer):
#         """
#         Perform one update step with data from a AHRS sensor array
#         :param gyroscope: A three-element array containing the gyroscope data in radians per second.
#         :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
#         :param magnetometer: A three-element array containing the magnetometer data. Can be any unit since a normalized value is used.
#         :return:
#         """
        
#         #Define Quarternion
#     	Quaternion = []
#     	q1 = math.cos(gx/2) * math.cos(gy/2) * math.cos(gz/2) * + math.sin(gx/2) * math.sin(gy/2) * math.sin(gz/2)
#     	Quaternion.append(q1)
#     	q2 = math.sin(gx/2) * math.cos(gy/2) * math.cos(gz/2) * - math.cos(gx/2) * math.sin(gy/2) * math.sin(gz/2)
#     	Quaternion.append(q2)
#     	q3 = math.cos(gx/2) * math.sin(gy/2) * math.cos(gz/2) * + math.sin(gx/2) * math.cos(gy/2) * math.sin(gz/2)
#     	Quaternion.append(q3)
#     	q4 = math.cos(gx/2) * math.cos(gy/2) * math.sin(gz/2) * - math.sin(gx/2) * math.sin(gy/2) * math.cos(gz/2)
#     	Quaternion.append(q4)
        
        

#         gyroscope = np.array(gyroscope, dtype=float).flatten()
#         accelerometer = np.array(accelerometer, dtype=float).flatten()
#         magnetometer = np.array(magnetometer, dtype=float).flatten()

#         # Normalise accelerometer measurement
#         if norm(accelerometer) is 0:
#             warnings.warn("accelerometer is zero")
#             return
#         accelerometer /= norm(accelerometer)

#         # Normalise magnetometer measurement
#         if norm(magnetometer) is 0:
#             warnings.warn("magnetometer is zero")
#             return
#         magnetometer /= norm(magnetometer)

#         h = q * (Quaternion(0, magnetometer[0], magnetometer[1], magnetometer[2]) * q.conj())
#         b = np.array([0, norm(h[1:3]), 0, h[3]])

#         # Gradient descent algorithm corrective step
#         f = np.array([
#             2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
#             2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
#             2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2],
#             2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]) - magnetometer[0],
#             2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - magnetometer[1],
#             2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - magnetometer[2]
#         ])
#         j = np.array([
#             [-2*q[2],                  2*q[3],                  -2*q[0],                  2*q[1]],
#             [2*q[1],                   2*q[0],                  2*q[3],                   2*q[2]],
#             [0,                        -4*q[1],                 -4*q[2],                  0],
#             [-2*b[3]*q[2],             2*b[3]*q[3],             -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
#             [-2*b[1]*q[3]+2*b[3]*q[1], 2*b[1]*q[2]+2*b[3]*q[0], 2*b[1]*q[1]+2*b[3]*q[3],  -2*b[1]*q[0]+2*b[3]*q[2]],
#             [2*b[1]*q[2],              2*b[1]*q[3]-4*b[3]*q[1], 2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]]
#         ])
#         step = j.T.dot(f)
#         step /= norm(step)  # normalise step magnitude

#         # Compute rate of change of quaternion
#         qdot = (q * Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5 - self.beta * step.T

#         # Integrate to yield quaternion
#         q += qdot * self.samplePeriod
#         self.quaternion = Quaternion(q / norm(q))  # normalise quaternion

#     def update_imu(self, gyroscope, accelerometer):
#         """
#         Perform one update step with data from a IMU sensor array
#         :param gyroscope: A three-element array containing the gyroscope data in radians per second.
#         :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
#         """
#         q = self.quaternion

#         gyroscope = np.array(gyroscope, dtype=float).flatten()
#         accelerometer = np.array(accelerometer, dtype=float).flatten()

#         # Normalise accelerometer measurement
#         if norm(accelerometer) is 0:
#             warnings.warn("accelerometer is zero")
#             return
#         accelerometer /= norm(accelerometer)

#         # Gradient descent algorithm corrective step
#         f = np.array([
#             2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
#             2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
#             2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2]
#         ])
#         j = np.array([
#             [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
#             [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
#             [0, -4*q[1], -4*q[2], 0]
#         ])
#         step = j.T.dot(f)
#         step /= norm(step)  # normalise step magnitude

#         # Compute rate of change of quaternion
#         qdot = (q * Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5 - self.beta * step.T

#         # Integrate to yield quaternion
#         q += qdot * self.samplePeriod
#         self.quaternion = Quaternion(q / norm(q))  # normalise quaternion
# 	    # Need to define Sample Period, Beta, Quarterion
# 	    SamplePeriod = SAMPLE_RATE
#     	Beta = 1
    	
#     	# Need to make these self.(accel, gyro, mag)
#     	# Needs to be in rad/s
#     	gx = g[0]*math.pi/180
#     	gy = g[1]*math.pi/180
#     	gz = g[2]*math.pi/180
#     	ax = a[0]
#     	ay = a[1]
#     	az = a[2]
#     	mx = m[0]
#     	my = m[1]
#     	mz = m[2]
    	
#     	#Define Quarternion
#     	Quaternion = []
#     	q1 = math.cos(gx/2) * math.cos(gy/2) * math.cos(gz/2) * + math.sin(gx/2) * math.sin(gy/2) * math.sin(gz/2)
#     	Quaternion.append(q1)
#     	q2 = math.sin(gx/2) * math.cos(gy/2) * math.cos(gz/2) * - math.cos(gx/2) * math.sin(gy/2) * math.sin(gz/2)
#     	Quaternion.append(q2)
#     	q3 = math.cos(gx/2) * math.sin(gy/2) * math.cos(gz/2) * + math.sin(gx/2) * math.cos(gy/2) * math.sin(gz/2)
#     	Quaternion.append(q3)
#     	q4 = math.cos(gx/2) * math.cos(gy/2) * math.sin(gz/2) * - math.sin(gx/2) * math.sin(gy/2) * math.cos(gz/2)
#     	Quaternion.append(q4)
    	
#         # q1 = Quaternion[0]
#         # q2 = Quaternion[1]
#         # q3 = Quaternion[2]
#         # q4 = Quaternion[3]
#         #Auxiliary variables to avoid repeated arithmetic

#         2q1 = 2f * q1
#         float _2q2 = 2f * q2
#         float _2q3 = 2f * q3
#         float _2q4 = 2f * q4
#         float _2q1q3 = 2f * q1 * q3
#         float _2q3q4 = 2f * q3 * q4
#         float q1q1 = q1 * q1
#         float q1q2 = q1 * q2
#         float q1q3 = q1 * q3
#         float q1q4 = q1 * q4
#         float q2q2 = q2 * q2
#         float q2q3 = q2 * q3
#         float q2q4 = q2 * q4
#         float q3q3 = q3 * q3
#         float q3q4 = q3 * q4
#         float q4q4 = q4 * q4

#         # Normalise accelerometer measurement
#         norm = (float)Math.Sqrt(ax * ax + ay * ay + az * az)
#         if (norm == 0f) return  #handle NaN
#         norm = 1 / norm    #use reciprocal for division
#         ax *= norm
#         ay *= norm
#         az *= norm

#         # Normalise magnetometer measurement
#         norm = (float)math.sqrt(mx * mx + my * my + mz * mz)
#         if (norm == 0f) return; # handle NaN
#         norm = 1 / norm    # use reciprocal for division
#         mx *= norm
#         my *= norm
#         mz *= norm

#         # Reference direction of Earth's magnetic field
#         _2q1mx = 2f * q1 * mx
#         _2q1my = 2f * q1 * my
#         _2q1mz = 2f * q1 * mz
#         _2q2mx = 2f * q2 * mx
#         hx = mx * q1q1 - _2q1my * q4 + _2q1mz * q3 + mx * q2q2 + _2q2 * my * q3 + _2q2 * mz * q4 - mx * q3q3 - mx * q4q4
#         hy = _2q1mx * q4 + my * q1q1 - _2q1mz * q2 + _2q2mx * q3 - my * q2q2 + my * q3q3 + _2q3 * mz * q4 - my * q4q4
#         _2bx = (float)math.sqrt(hx * hx + hy * hy)
#         _2bz = -_2q1mx * q3 + _2q1my * q2 + mz * q1q1 + _2q2mx * q4 - mz * q2q2 + _2q3 * my * q4 - mz * q3q3 + mz * q4q4
#         _4bx = 2f * _2bx
#         _4bz = 2f * _2bz

#         # Gradient decent algorithm corrective step
#         s1 = -_2q3 * (2f * q2q4 - _2q1q3 - ax) + _2q2 * (2f * q1q2 + _2q3q4 - ay) - _2bz * q3 * (_2bx * (0.5f - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q4 + _2bz * q2) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + _2bx * q3 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5f - q2q2 - q3q3) - mz)
#         s2 = _2q4 * (2f * q2q4 - _2q1q3 - ax) + _2q1 * (2f * q1q2 + _2q3q4 - ay) - 4f * q2 * (1 - 2f * q2q2 - 2f * q3q3 - az) + _2bz * q4 * (_2bx * (0.5f - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (_2bx * q3 + _2bz * q1) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + (_2bx * q4 - _4bz * q2) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5f - q2q2 - q3q3) - mz);
#         s3 = -_2q1 * (2f * q2q4 - _2q1q3 - ax) + _2q4 * (2f * q1q2 + _2q3q4 - ay) - 4f * q3 * (1 - 2f * q2q2 - 2f * q3q3 - az) + (-_4bx * q3 - _2bz * q1) * (_2bx * (0.5f - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (_2bx * q2 + _2bz * q4) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + (_2bx * q1 - _4bz * q3) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5f - q2q2 - q3q3) - mz)
#         s4 = _2q2 * (2f * q2q4 - _2q1q3 - ax) + _2q3 * (2f * q1q2 + _2q3q4 - ay) + (-_4bx * q4 + _2bz * q2) * (_2bx * (0.5f - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + (-_2bx * q1 + _2bz * q3) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + _2bx * q2 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5f - q2q2 - q3q3) - mz)
#         norm = 1f / (float)math.sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)    #normalise step magnitude
#         s1 *= norm
#         s2 *= norm
#         s3 *= norm
#         s4 *= norm

#         # Compute rate of change of quaternion
#         qDot1 = 0.5f * (-q2 * gx - q3 * gy - q4 * gz) - Beta * s1
#         qDot2 = 0.5f * (q1 * gx + q3 * gz - q4 * gy) - Beta * s2
#         qDot3 = 0.5f * (q1 * gy - q2 * gz + q4 * gx) - Beta * s3
#         qDot4 = 0.5f * (q1 * gz + q2 * gy - q3 * gx) - Beta * s4

#         # Integrate to yield quaternion
#         q1 += qDot1 * SamplePeriod
#         q2 += qDot2 * SamplePeriod
#         q3 += qDot3 * SamplePeriod
#         q4 += qDot4 * SamplePeriod
#         norm = 1f / (float)math.sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4);    #normalise quaternion
#         Quaternion[0] = q1 * norm
#         Quaternion[1] = q2 * norm
#         Quaternion[2] = q3 * norm
#         Quaternion[3] = q4 * norm
        
#         #Variable reassignment for readability for Euler Angle Conversion
#         q1 = Quaternion[0]
#         q2 = Quaternion[1]
#         q3 = Quaternion[2]
#         q4 = Quaternion[3]
        
#         #Convert back to Euler Angles
#         gx = math.atan((2*(q1*q2+q3*q4))/(1-2*(math.pow(q2,2)+math.pow(q3,2))))
#         gy = math.asin(2*(q1*q3 + q4*q2))
#         gz = math.atan((2*(q1*q4+q2*q3))/(1-2*(math.pow(q3,2)+math.pow(q4,2))))
        
#         return ([gx,gy,gz])
    
    
	@property
	def accel(self):
		return self.read_xyz(MPU9250_ADDRESS, ACCEL_DATA, self.alsb)

	@property
	def gyro(self):
		return self.read_xyz(MPU9250_ADDRESS, GYRO_DATA, self.glsb)

	@property
	def temp(self):
		"""
		Returns chip temperature in C
		pg 33 datasheet:
		Temp_degC = ((Temp_out - Temp_room)/Temp_Sensitivity) + 21 degC
		"""
		temp_out = self.read16(MPU9250_ADDRESS, TEMP_DATA)
		temp = temp_out / 333.87 + 21.0  # these are from the datasheets
		return temp

	@property
	def mag(self):
		return self.read_xyz(AK8963_ADDRESS, MAGNET_DATA, self.mlsb)