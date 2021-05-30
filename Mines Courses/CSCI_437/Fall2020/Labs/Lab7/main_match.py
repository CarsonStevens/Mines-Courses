import numpy as np
import cv2


##############
# Algorithm parameters.

# Define which image is the query image (img2.png, img3.png, etc).
# The reference image is always img1.png.
index_of_query_image = 2        # Should be in the range 2..6 inclusive

# Define the ratio of best-match-distance to second-best-match-distance.  This is used to
# reject ambiguous matches.  The smaller this ratio, the more potentially false matches
# are rejected, but you also lose more valid matches.
ratio_match_dist = 0.8      # Should be in the range (0..1]
##############

# Ground truth homographies between images 1 and 2, 1 and 3, 1 and 4, etc.
Homographies = np.array([
    np.array([[0.87976964, 0.31245438, -39.430589],     # H_1_2
              [-0.18389418, 0.93847198, 153.15784],
              [0.00019641425, -1.6015275e-05, 1.0]]),
    np.array([[0.76285898, -0.29922929, 225.67123],     # H_1_3
              [0.33443473, 1.0143901, -76.999973],
              [0.00034663091, -1.4364524e-05, 1.0]]),
    np.array([[0.66378505, 0.68003334, -31.230335],     # H_1_4
              [-0.144955, 0.97128304, 148.7742],
              [0.00042518504, -1.3930359e-05, 1.0]]),
    np.array([[0.62544644, 0.057759174, 222.01217],     # H_1_5
              [0.22240536, 1.1652147, -25.605611],
              [0.00049212545, -3.6542424e-05, 1.0]]),
    np.array([[0.4271459, -0.67181765, 453.61534],     # H_1_6
              [0.44106579, 1.013323, -46.534569],
              [0.00051887712, -7.8853731e-05, 1.0]])
])

def main():
    assert(ratio_match_dist > 0 and ratio_match_dist <= 1)
    assert(index_of_query_image >= 2 and index_of_query_image <= 6)

    filename1 = "img1.png"
    filename2 = "img%d.png" % index_of_query_image

    # Get homography between image 1 and the query image.
    H = Homographies[index_of_query_image-2]

    bgr_img1 = cv2.imread(filename1)
    bgr_img2 = cv2.imread(filename2)

    image_height = bgr_img1.shape[0]
    image_width = bgr_img1.shape[1]

    keypoints1, desc1 = detect_features(bgr_img1, show_features=True)
    keypoints2, desc2 = detect_features(bgr_img2, show_features=True)

    # Match image descriptors.
    matcher = cv2.BFMatcher(cv2.NORM_L2)    # Creates a brute force matcher object

    # Use k nearest neighbor matching and apply ratio test.
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio_match_dist * n.distance:
            good.append(m)
    matches = good

    # Analyze matches.
    true_positives = 0   # Number of correct matches
    false_positives = 0   # Number of incorrect matches
    actual_positives = 0    # Number of actual positives (ie, feature is in query image)

    for id1, kp1 in enumerate(keypoints1):
        p1 = np.array(kp1.pt)       # We detected a keypoint here in image 1

        bgr_display1 = bgr_img1.copy()
        bgr_display2 = bgr_img2.copy()

        # Display the feature on image 1.
        x = int(round(p1[0]))
        y = int(round(p1[1]))
        cv2.circle(bgr_display1, center=(x,y), radius=int(kp1.size),
                   color=(0,255,255), thickness=2)
        cv2.putText(img=bgr_display1, org=(x,y-5), text=str(id1), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.5, color=(0, 255, 255), thickness=2)

        # Apply the known transform to the point to see where it should be in image 2.
        pg = H @ np.array([p1[0], p1[1], 1])
        pg /= pg[2]  # Ground truth location

        # See if the feature should have been found (ie, is in the image).
        if pg[0] >= 0 and pg[0] < image_width and pg[1] >= 0 and pg[1] < image_height:
            # print("Actual positive (feature should be matchable)")
            actual_positives += 1

        # Display the ground truth location on image 2.
        x = int(round(pg[0]))
        y = int(round(pg[1]))
        cv2.circle(bgr_display2, center=(x,y), radius=int(kp1.size),
                   color=(0,255,0), thickness=-1)
        cv2.putText(img=bgr_display2, org=(x,y-5), text='G', fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.5, color=(0, 255, 0), thickness=2)

        # Find the match (if any) for this keypoint.
        match = next((match for match in matches if match.queryIdx == id1), None)

        if match is not None:
            # A match was found.
            kp2 = keypoints2[match.trainIdx]
            p2 = np.array(kp2.pt)  # detected match location

            # Display the matching feature in image 2.
            x = int(round(p2[0]))
            y = int(round(p2[1]))
            cv2.circle(bgr_display2, center=(x, y), radius=int(kp2.size),
                       color=(0, 255, 255), thickness=2)
            cv2.putText(img=bgr_display2, org=(x + 5, y), text='D',
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1.5, color=(0, 255, 255), thickness=2)

            # See if the match was correct.
            d = np.linalg.norm(p2 - pg[0:2])  # Error distance
            # Assume match is correct if error is smaller than the feature size.
            if d < kp1.size:
                # print("Found a match: True positive")
                true_positives += 1
            else:
                # print("Found a match: False positive")
                false_positives += 1




    print(f'''Precision:\t{np.round(true_positives/(true_positives+false_positives),8)}''')
    print(f'''Recall:\t{np.round(true_positives/actual_positives, 8)}''')
    print("All done, bye!")
    # cv2.imshow("Image 1", bgr_display1)
    # cv2.imshow("Image 2", bgr_display2)
    # cv2.waitKey(0)

def detect_features(bgr_img, show_features=False):
    detector = cv2.xfeatures2d.SIFT_create(
        nfeatures=1000,
        nOctaveLayers=3,        # default = 3
        contrastThreshold=0.04, # default = 0.04
        edgeThreshold=10,       # default = 10
        sigma=1.6               # default = 1.6
    )

    # Extract keypoints and descriptors from image.
    gray_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_image, mask=None)

    # Optionally draw detected keypoints.
    if show_features:
        # Possible flags: DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, DRAW_MATCHES_FLAGS_DEFAULT
        bgr_display = bgr_img.copy()
        cv2.drawKeypoints(image=bgr_display, keypoints=keypoints,
                          outImage=bgr_display,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("Features", bgr_display)
        print("Number of keypoints: ", len(keypoints))
        # cv2.waitKey(0)

    return keypoints, descriptors

if __name__ == "__main__":
    main()
