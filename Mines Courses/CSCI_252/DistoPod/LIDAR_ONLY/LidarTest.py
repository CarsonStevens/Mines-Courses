#Lidar Test
from lidar_lite import Lidar_Lite
import time as t

####################################################

lidar = Lidar_Lite()
connected = lidar.connect(1)
if connected < -1:
    print("Nope")
for x in range (0,100):
    print(lidar.getDistance())
    t.sleep(0.5)


