import cv2
import numpy as np

# Find vanishing point directions.
# Inputs:
#   houghLines:  An array of size (N,1,4), which is the output of cv2.HoughLinesP
#   bgr_img:  The original input image (in blue-green-red format)
#   num_to_find:  The desired number of vanishing points to find
#   K:  The camera matrix (if known)
# Outputs:
#   vanishing_directions: A list of num_to_find vectors, each 3 dimensional
def find_vanishing_point_directions(houghLines, bgr_img, num_to_find=3, K=None):
    assert(houghLines.shape[1] == 1)
    assert(houghLines.shape[2] == 4)
    vanishing_directions = []

    # For visualizing the lines, draw on a grayscale version of the image.
    bgr_display = cv2.cvtColor(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    display_colors = [(0,0,255), (0,255,0), (255,0,0), (255,0,255), (0,255,255), (255,255,0)]

    # Find each vanishing point in turn.
    for n in range(num_to_find):
        d, contributing_lines = find_vanishing_pt(houghLines, bgr_img)
        print("Found a vanishing point with direction vector: ", d)
        print("Number of contributing lines: ", np.sum(contributing_lines))

        # Show contributing line segments.
        for i in range(0, len(houghLines)):
            if contributing_lines[i]:
                l = houghLines[i][0]
                cv2.line(bgr_display, (l[0], l[1]), (l[2], l[3]), display_colors[n], thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("Contributing line segments", bgr_display)
        cv2.waitKey(0)

        # Eliminate contributing lines and repeat.
        houghLines = houghLines[contributing_lines == False]

    return vanishing_directions

# This function uses RANSAC to find a set of line segments that fit a single vanishing point.
# See this post:  https://yichaozhou.com/post/20190402vanishingpoint/
def find_vanishing_pt(houghLines, bgr_img, K=None):
    best_vanishing_direction = None
    contributing_lines = []

    if K is None:
        # Estimate a camera matrix.
        image_width = bgr_img.shape[1]
        image_height = bgr_img.shape[0]
        f = image_width
        cx = image_width / 2
        cy = image_height / 2
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.double)
    Kinv = np.linalg.inv(K)

    num_lines = len(houghLines)
    if num_lines < 2:       # Need at least two lines
        return best_vanishing_direction, contributing_lines

    # Compute normals for each line segment.
    normals = np.ones((num_lines, 3))
    for i in range(num_lines):
        l = houghLines[i][0]
        e1 = Kinv @ np.array([l[0], l[1], 1])
        e2 = Kinv @ np.array([l[2], l[3], 1])
        v = np.cross(e1, e2)
        normals[i, :] = v / np.linalg.norm(v)

    NUM_ITERATIONS = 3000
    INLIER_THRESH = 0.03        # Maximum value of dot product
    best_num_inliers = 0

    for i in range(NUM_ITERATIONS):
        # Randomly pick two lines.
        i1 = np.random.randint(0, num_lines)
        i2 = np.random.randint(0, num_lines)
        if i1==i2:
            continue
        n1 = normals[i1]
        n2 = normals[i2]

        # Get the 3D line direction vector.
        d = np.cross(n1, n2)
        m = np.linalg.norm(d)
        if m < 1e-6:
            continue    # The two lines are colinear in the image; skip them
        d = d / np.linalg.norm(d)

        # Ok, count how many lines are consistent with direction d.
        # A line is consistent if its norm is perpendicular to d.
        residual_errors = np.zeros(num_lines)
        for j in range(num_lines):
            n = normals[j]
            residual_errors[j] = abs(np.dot(n, d))
        num_inliers = np.sum(residual_errors < INLIER_THRESH)
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_vanishing_direction = d
            contributing_lines = residual_errors < INLIER_THRESH

    return best_vanishing_direction, contributing_lines
