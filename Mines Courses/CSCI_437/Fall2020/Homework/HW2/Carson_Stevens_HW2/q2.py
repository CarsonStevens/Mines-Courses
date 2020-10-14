#!/usr/bin/env python
# coding: utf-8

# <h2>Question 2</h2>
# Using the method of normalized cross-correlation, find all instances of the letter “a” in the image “textsample.tif”. To do this, first extract a template subimage w(x,y) of an example of the letter “a”. Then match this template to the image (you can use OpenCV’s “matchTemplate” function). Find local peaks in the correlation scores image. Then threshold these peaks (you will have to experiment with the threshold) so that you get 1’s where there are “a”s and nowhere else. You can use OpenCV’s “connectedComponentsWithStats” function to extract the centroids of the peaks. Take the locations found and draw a box (or some type of marker) overlay on the original image showing the locations of the “a”s. Your program should also count the number of detected “a”s.
# 
# <h3>Solution</h3>
# I first read in the textsample.tif and converted it to a binary image. I then tried to separate some of the letters and normalize the shapes of the "a"s with morphological operations. I then let the user choose a template or read in a template from file. A scores image is then produced. After this, I counted the "a"s on the page and decreased my threshold until I had the all of the "a"s and nothing else selected. This worked for the most part. Some of the "a"s are boxed multiple times. When I did the final drawing on the image, I made sure that the current matches weren't on top of each other and decreased my count to correct for the duplicates.
# 
# <h4>"a"s Count: 50</h4>
# 

# In[8]:


import cv2
import numpy as np

def getUserTemplate(img):
    try:
        r = cv2.selectROI("ROI Template Selection", img)
        x, y, w, h = r
        # Crop image
        imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        original_with_ROI=cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
    finally:
        cv2.destroyAllWindows()
        return original_with_ROI, imCrop, w, h

def thinning(img1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # Create an empty output image to hold values
    thin = np.zeros(thresh_img.shape,dtype='uint8')
    img1 = thresh_img
    # Loop until erosion leads to an empty set
    while (cv2.countNonZero(img1)!=0):
        # Erosion
        erode = cv2.erode(img1,kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset,thin)
        # Set the eroded image for next iteration
        img1 = erode.copy()

    thin = ~thin
    return thin



import PIL.Image
from io import BytesIO

#Use 'jpeg' instead of 'png' (~5 times faster)
def array_to_image(img, fmt='jpeg',width=500):
    #Create binary stream object
    f = BytesIO()
    #Convert array to binary stream object
    new_p = PIL.Image.fromarray(img)
    if new_p.mode != 'RGB':
        new_p = new_p.convert('RGB')
    new_p.save(f, fmt)
    return IPython.display.Image(data=f.getvalue(), width=width)



import IPython.display
import time


# In[9]:


d1 = IPython.display.display("", display_id=1)
d4 = IPython.display.display("", display_id=4)
try: 
    original_img = cv2.imread("textsample.tif")
    gray_img = cv2.cvtColor(original_img.astype(np.float32), cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray_img, thresh=.7, maxval=255, type=cv2.THRESH_BINARY)
#     thresh_img = cv2.adaptiveThreshold(src=original_img,maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,thresholdType=cv2.THRESH_BINARY,blockSize=5,C=-10)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,3))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,1))
    thin = cv2.dilate(thresh_img,kernel,3)
    thin = cv2.erode(thin,kernel2,5)
    thin = cv2.dilate(thresh_img,kernel4,1)
    thin = cv2.erode(thin,kernel3,2)
    d4.update(array_to_image(img=thin, fmt='jpeg', width=500))
    
    template = cv2.imread("template1.jpg",0)
    user_select = False
    
    if (template.shape[0] != 0 and not user_select):
        template_width = template.shape[0]
        template_height = template.shape[1]
    else:
        template_img, template, template_width, template_height = getUserTemplate(thin)
        cv2.imwrite("template.jpg", template)

finally:
    d1.update(array_to_image(img=template, fmt='jpeg', width=100))
    cv2.destroyAllWindows()
#     IPython.display.clear_output()
    


# In[10]:


d2 = IPython.display.display("", display_id=2)
d3 = IPython.display.display("", display_id=3)
d5 = IPython.display.display("", display_id=5)

scores = cv2.matchTemplate(thin.astype(np.float32), template.astype(np.float32), method=cv2.TM_CCOEFF_NORMED).astype(np.float32)
d2.update(array_to_image(img=scores, fmt='jpeg', width=800))

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(scores)
threshold = 0.2
while (threshold == 0.2 or matches[0].shape[0] >= 105):
    matches = np.where(scores.astype(np.float32) >= max_val-threshold)
    threshold -= 0.0001
    d5.update(IPython.display.HTML(f"""<h1>Threshold:\t{np.round(1-threshold,5)}</h1>
                                        <h1>Current Match Count:\t{matches[0].shape[0]}</h1>
                                    """))

last_bound_x = 0
last_bound_y = 0
matching = matches[0].shape[0]
for match in range(matches[0].shape[0]):
    x = matches[0][match]
    y = matches[1][match]
    if (x > last_bound_x+template_width or y > last_bound_y+template_height/2):
        original_img = cv2.rectangle(img=original_img, pt1=(y,x), pt2=(y+template_height,x+template_width), color=(0, 255, 255), thickness=3)
        last_bound_x = x
        last_bound_y = y
    else:
        matching -= 1
        d5.update(IPython.display.HTML(f"""<h1>Threshold:\t{np.round(1-threshold,5)}</h1>
                                        <h1>Current Match Count:\t{matching}</h1>
                                    """))
d3.update(array_to_image(img=original_img, fmt='jpeg', width=800))


# In[ ]:




