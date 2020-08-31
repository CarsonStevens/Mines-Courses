#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

P1 = (np.array([-1, -1, 1]))
P2 = (np.array([1, -1, 1]))
P3 = (np.array([1, 1, 1]))
P4 = (np.array([-1, 1, 1]))
points = [P1, P2, P3, P4]
k=1
frame = 0
marker_start_size = 30
marker_size = marker_start_size


def getCamera(X,z):
    focal_length = 500
    return int((focal_length * X/z))

def normalize(P):
    unique, check = np.unique(P, return_counts=True)
    if (np.isin(3,check)):
        filler = 1/P.size
        return np.full(P.size, filler)
    else:
        return P - P.min()/(P.max()-P.min())

# Read images from a video file in the current folder.
video_capture = cv2.VideoCapture("earth.wmv")  # Open video capture object
got_image, bgr_image = video_capture.read()  # Make sure we can read video
if not got_image:
    print("Cannot read video source")
    sys.exit()
else:
    print("video read in")



# Create output movie file.
# These types of video formats work:
#   *.avi   Use VideoWriter_fourcc('D', 'I', 'V', 'X') or
#              VideoWriter_fourcc('M', 'J', 'P', 'G')
#   *.wmv   Use VideoWriter_fourcc('W', 'M', 'V', '2')
#   *.mp4   Use VideoWriter_fourcc('M', 'P', '4', '2')
# See http://www.fourcc.org/codecs.php for more codecs.
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter(apiPreference=cv2.CAP_FFMPEG, filename="output.avi", fourcc=fourcc, fps=30,
                              frameSize=(int(video_capture.get(3)), int(video_capture.get(4))))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


# Read and show images until end of video is reached.
while True:
    got_image, bgr_img = video_capture.read()
    height, width, channels = bgr_image.shape
    # print(height, width)
    if not got_image:
        break  # End of video; exit the while loop


    for point in points:
        k+=0.1
        cv2.drawMarker(bgr_img, position=(int(getCamera(point[0],k) +width/2),int(height/2+ (getCamera(point[1],k)))),
                       color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=int(marker_size))

    marker_size -= marker_start_size / total_frames
    bgr_img = cv2.putText(bgr_img, text=str(frame), org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5, color=(0, 255, 0))
    cv2.imshow("Earth", bgr_img)
    frame += 1
    videoWriter.write(bgr_img)

    # Wait for xx msec (0 means wait till a keypress).
    cv2.waitKey(0)

videoWriter.release()
print("Output Video Saved")

