def CalibrationCalc(offset_data):
    offset = 0
    for i in offset_data:
        offset = offset + offset_data[i]
    return offset/len(offset_data)

accel_cal = []
gyro_cal = []
Calibration_MPU = MPU()
Calibration_Timer = time.time() + 10

while(time.time() <= Calibration_Timer):
    a = Calibration_MPU.accel
    g = Calibration_MPU.gyro
    accel_cal.append(a)
    gyro_cal.append(g)

accelX_cal = accel_cal[0]
accelY_cal = accel_cal[1]
accelZ_cal = accel_cal[2]
gyroX_cal = gyro_cal[0]
gyroY_cal = gyro_cal[1]
gyroZ_cal = gyro_cal[2]


    
accelX_calibration_offset = CalibrationCalc(accelX_cal)
gyroX_calibration_offset = CalibrationCalc(accelY_cal)
accelY_calibration_offset = CalibrationCalc(accelZ_cal)
gyroY_calibration_offset = CalibrationCalc(gyroX_cal)
accelZ_calibration_offset = CalibrationCalc(gyroY_cal)
gyroZ_calibration_offset = CalibrationCalc(gyroZ_cal)

print(accelX_calibration_offset, gyroX_calibration_offset)
print(accelY_calibration_offset, gyroY_calibration_offset)
print(accelZ_calibration_offset, gyroZ_calibration_offset)
