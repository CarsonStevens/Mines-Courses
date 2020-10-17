#!/usr/bin/env python
# coding: utf-8

# <h1>Lab 2: <a href="https://elearning.mines.edu/courses/25410/assignments/132703">Image Transformations</a></h1>
# <h2>Resulting Image</h2>
# <img src="https://storage.googleapis.com/general_photo_pdf_storage/BigDipper.jpg" alt="Big Dipper">

# In[1]:


import numpy as np
np.warnings.filterwarnings('ignore')
import cv2


# In[2]:


def debugMatrix(name, M):
    print(name, ":\n", type(M), "shape: ", M.shape, "\n", M)

def getRotationMatrixX(ax, rad=False):
    if not rad:
        ax = np.radians(ax)
    sx = np.sin(ax)
    cx = np.cos(ax)
    return np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))

def getRotationMatrixY(ay, rad=False):
    if not rad:
        ay = math.radians(ay)
    sy = np.sin(ay)
    cy = np.cos(ay)
    return np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))

def getRotationMatrixZ(az, rad=False):
    if not rad:
        az = math.radians(az)
    sz = np.sin(az)
    cz = np.cos(az)
    return np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))

def getRotationMatrix(ax,ay,az,rad=False,order="XYZ",debug=False):
    X = getRotationMatrixX(ax, rad)
    Y = getRotationMatrixY(ay, rad)
    Z = getRotationMatrixZ(az, rad)

    if debug:
        debugMatrix("Rotation X", X)
        debugMatrix("Rotation Y", Y)
        debugMatrix("Rotation Z", Z)

    if order == "XYZ":
        return Z @ Y @ X
    elif order == "ZYX":
        return X @ Y @ Z
    elif order == "YXZ":
        return Z @ X @ Y
    elif order == "YZX":
        return X @ Z @ Y
    elif order == "XZY":
        return Y @ Z @ X
    elif order == "ZXY":
        return Y @ X @ Z

# Return camera to world and world to camera
def getHomoTransforms(rotation_matrix, translation_matrix, debug=False):
    # The only rotation is about x
    debugMatrix("R", rotation_matrix)
    debugMatrix("T", translation_matrix)
    H_c_w = np.block([[rotation_matrix, translation_matrix], [0,0,0,1]]) 
    # Get transformation from world to camera.
    H_w_c = np.linalg.inv(H_c_w)
    if debug:
        debugMatrix("H_w_c", H_w_c)
        debugMatrix("H_c_w", H_c_w)
    return H_c_w, H_w_c

# Return Matrix K
# f: focal_length
# sx: sensor size x
# sy: sensor size y
# center: pixel location [x,y] of optical center of image
def getIntrinsicCamMatrix(f, sx, sy, center=[0,0]):
    return np.array([[f/sx, 0, center[0]], [0, f/sy, center[1]], [0, 0, 1]])

# Returns Mext
def getExtrinsicCamMatrix(H_w_c):
    Mext = H_w_c[0:3, :]
    return Mext

# For no rotation or translation
def threeDToTwoDProjection(f, sx, sy, P, cx=0, cy=0):
    return np.array((f/sx)*P[0]/P[2]+cx, (f/sy)*P[1]/P[2]+cy, 1)


def threeDToTwoDProjection(P_w,                                                                                                              rotation_matrix,
                           translation_matrix,
                           f, sx, sy,
                           cx=0,cy=0,
                           debug=False):
    if (P_w.size == 3):
        P_w = np.array([P_w,1])

    K = getIntrinsicCamMatrix(f,sx,sy,[cx,cy])
    Mext = getExtrinsicCamMatrix(getHomoTransforms(rotation_matrix,translation_matrix, debug=debug)[1])
    p = np.array(K @ Mext @ P_w)
    p /= p[2]
    p = np.hstack(p)

    if debug:
        debugMatrix("p", p)
        debugMatrix("K", K)
        debugMatrix("Mext", Mext)
        debugMatrix("P_w", P_w)

    return p


# In[ ]:


#Camera Angles
ax1 = 1.1
ay1 = -0.5
az1 = 0.1

# Camera Parameters
height = 256
width = 170
focal = 400
cx = width/2.0
cy = height/2.0

# Translation Matrix
translation = np.array([[10,-25,40]]).T

# World Points
points = np.array([[6.8158,-35.1954,43.0640,1],
                   [7.8493,-36.1723,43.7815,1],
                   [9.9479,-25.2799,40.1151,1],
                   [8.8219,-38.3767,46.6153,1],
                   [9.5890,-28.8402,42.2858,1],
                   [10.8082,-48.8146,56.1475,1],
                   [13.269,-58.0988,59.1422,1]])


#  Give the 3x3 rotation matrix corresponding to the rotations above.
R1 = getRotationMatrix(ax1,ay1,az1,rad=True,order="XYZ",debug=False)

# Since the rotation matrix is “orthonormal” matrix 
#(i.e., a square matrix whose rows and columns are orthogonal unit vectors), 
#its inverse is equal to its transpose. Show this.
assert(R1.all() == R1.T.all())

# Give the 3x3 rotation matrix where the same rotations described in part (a)
# are done in the opposite order; i.e., first a rotation about the Z axis
R2 = getRotationMatrix(ax1,ay1,az1,rad=True,order="ZYX",debug=False)
assert(not np.array_equal(R1, R2))

# Compute the homogeneous transformation matrix that represents the pose
# of the camera with respect to the world
# Compute the homogeneous transformation matrix that represents the pose 
# of the world with respect to the camera
H_c_w, H_w_c = getHomoTransforms(R1, translation, debug=False)
debugMatrix("H_c_w", H_c_w)
debugMatrix("H_w_c", H_w_c)

# Write the intrinsic camera calibration matrix K.
K = getIntrinsicCamMatrix(focal, focal/width, focal/height, center=[cx,cy])
debugMatrix("K", K)

# Find new points on image
Mext = H_w_c[0:3, :]
new_points = []
for point in points:
    new_point = K @ Mext @ point
    new_point = new_point/new_point[2]
    new_points.append(new_point)

# Create a blank (zeros) image, and project the 7 points to the image.
# Write white dots into the image at those points. 
image = np.zeros((width,height, 3))
for point in new_points:
    image[int(point[1])][int(point[0])] = (0,0,255)

# Using OpenCV's “line” function, draw lines between the points on the image. 
# Namely draw a line from point 0 to point 1, another line from point 1 
# to point 2, and so forth. 
for i in range(len(new_points) -1):
    cv2.line(image,(int(new_points[i][0]),int(new_points[i][1])), (int(new_points[i+1][0]), int(new_points[i+1][1])),(255,0,0),thickness=1)

# Save image
cv2.imwrite("BigDipper.jpg", image)
cv2.imshow("Big Dipper", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




