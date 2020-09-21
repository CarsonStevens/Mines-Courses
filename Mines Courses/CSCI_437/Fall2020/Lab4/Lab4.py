#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import time
import PIL.Image
from io import BytesIO
import IPython.display
import numpy as np

#Use 'jpeg' instead of 'png' (~5 times faster)
def array_to_image(a, fmt='jpeg'):
    #Create binary stream object
    f = BytesIO()
    #Convert array to binary stream object
    PIL.Image.fromarray(a).save(f, fmt)
    return IPython.display.Image(data=f.getvalue())

def nothing(x):pass



# binary_img = cv2.adaptiveThreshold(src=gray_img,maxValue=255,  # output value where condition met
#                                    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
#                                    thresholdType=cv2.THRESH_BINARY,  # threshold_type
#                                    blockSize=51,  # neighborhood size (a large odd number)
#                                    C=-10)  # a constant to subtract from mean


# In[12]:


d1 = IPython.display.display("", display_id=1)
d2 = IPython.display.display("", display_id=2)
d3 = IPython.display.display("", display_id=3)
low_thresholds = [134, 70, 50]
high_thresholds = [255, 255, 255]

stop0 = cv2.imread("stop0.jpg")
stop1 = cv2.imread("stop1.jpg")
stop2 = cv2.imread("stop2.jpg")
stop3 = cv2.imread("stop3.jpg")
stop4 = cv2.imread("stop4.jpg")

stops = [stop0,stop1,stop2,stop3,stop4]
bgr_img = stop4
image_height = bgr_img.shape[0]    
image_width = bgr_img.shape[1]# Convert BGR to HSV.
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)# Split into the different bands.
planes = cv2.split(hsv_img)    
windowNames = ["Hue image", "Saturation image", "Gray image"]
try:
#     for i in range(3):        
#         cv2.namedWindow(windowNames[i])# Create trackbars.
#     for i in range(3):        
#         cv2.createTrackbar("Low", windowNames[i], low_thresholds[i], 255, nothing)        
#         cv2.createTrackbar("High", windowNames[i], high_thresholds[i], 255, nothing)
    while True:# Create output thresholded image.
        thresh_img = np.full((image_height, image_width), 255, dtype=np.uint8)
        for i in range(3):            
#             low_val = cv2.getTrackbarPos("Low", windowNames[i])            
#             high_val = cv2.getTrackbarPos("High", windowNames[i]) 
            low_val = low_thresholds[i]
            high_val = high_thresholds[i]
            _,low_img = cv2.threshold(planes[i], low_val, 255, cv2.THRESH_BINARY)            
            _,high_img = cv2.threshold(planes[i], high_val, 255, cv2.THRESH_BINARY_INV)            
            thresh_band_img = cv2.bitwise_and(low_img, high_img)            
            #cv2.imshow(windowNames[i], thresh_band_img)# AND with output thresholded image.
            thresh_img = cv2.bitwise_and(thresh_img, thresh_band_img)        


        kernelClose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filtered_img_close = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernelClose)
        filtered_img_open = cv2.morphologyEx(filtered_img_close, cv2.MORPH_OPEN, kernelOpen)
        d1.update(array_to_image(filtered_img_close))
        d2.update(array_to_image(filtered_img_open))
        cv2.imwrite("stop_output4.jpg", filtered_img_open)

except:
    pass
finally:
    cv2.destroyAllWindows()
    IPython.display.clear_output()
    


# 
# <div style="display:flex; flex-direction:row; justify-contents:space-around;font-size:3rem;width: 100%;color:rgba(255,255,255,1);padding:2rem 5rem;">
#     <div style="background-color: rgba(0,0,0,0.3);padding:1rem;">
#         <h2>Hue</h2>
#         <h4 style="margin-bottom: 1rem;">Low</h4>
#         134
#         <h4 style="margin-bottom: 1rem;">High</h4>
#         255
#     </div>
#     <div style="background-color: rgba(0,0,0,0.6);padding:1rem;">
#        <h2>Saturation</h2>
#         <h4 style="margin-bottom: 1rem;">Low</h4>
#         50
#         <h4 style="margin-bottom: 1rem;">High</h4>
#         255
#     </div>
#     <div style="background-color: rgba(0,0,0,0.9);padding:1rem;">
#         <h2>Grey</h2>
#         <h4 style="margin-bottom: 1rem;">Low</h4>
#         70
#         <h4 style="margin-bottom: 1rem;">High</h4>
#         255
#     </div>
# </div>
# 
# <div style="display:flex; flex-direction:row; justify-contents:space-around;font-size:3rem;width: 100%;color:rgba(0,0,0,1);padding:2rem 5rem;">
#     <div style="margin:1rem">
#         <h4>Stop0</h4>
#         <img src="stop_output0.jpg" width="200px">
#     </div>
#        <div style="margin:1rem">
#         <h4>Stop1</h4>
#         <img src="stop_output1.jpg" width="200px">
#     </div>
#         <div style="margin:1rem">
#         <h4>Stop2</h4>
#         <img src="stop_output2.jpg" width="200px">
#     </div>
#         <div style="margin:1rem">
#         <h4>Stop3</h4>
#         <img src="stop_output3.jpg" width="200px">
#     </div>
#         <div style="margin:1rem">
#         <h4>Stop4</h4>
#         <img src="stop_output4.jpg" width="200px">
#     </div>
# </div>
# 

# In[ ]:




