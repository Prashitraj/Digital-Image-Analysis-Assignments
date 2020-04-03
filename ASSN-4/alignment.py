import sys
import cv2
import random
import numpy as np
import math
from scipy import ndimage

def rotate_img(img):

    img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    img2 = np.zeros(img_before.shape,np.uint8)
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 3,cv2.LINE_AA)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
    cv2.imwrite('lines.jpg',img2)
    median_angle = np.median(angles)
    img_rotated = ndimage.rotate(img_before, median_angle)

    print "Angle is {}".format(median_angle)
    return img_rotated

infile = sys.argv[1]
outfile = sys.argv[2]
img_before = cv2.imread(infile)
img_rotated = rotate_img(img_before)
cv2.imwrite(outfile, img_rotated)
