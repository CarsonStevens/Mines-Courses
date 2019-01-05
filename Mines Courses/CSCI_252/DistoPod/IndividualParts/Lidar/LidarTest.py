#Lidar Test
from lidar_lite import Lidar_Lite
import time as t
lidar = Lidar_Lite()
connected = lidar.connect(1)
if connected < -1:
    print("Nope")
for x in range (0,100):
    print(lidar.getDistance())
    t.sleep(0.5)

####################################################

import smbus
from time import time
import statistics
bus = smbus.SMBus(1)
address = 0x62

def read():
    bus.write_byte_data(address, 0x00,0x04)
    while bus.read_byte_data(address,0x01) % 2 == 1:
        pass
    high = bus.read_byte_data(address,0x0f)
    low = bus.read_byte_data(address,0x10)
    distance = (high << 8) + low
    return distance
