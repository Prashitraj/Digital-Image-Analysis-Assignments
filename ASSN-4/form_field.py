import cv2
import numpy as np
import sys

def box_extraction(img_for_box_extraction_path):

    imgc = cv2.imread(img_for_box_extraction_path)  # Read the image
    img = cv2.cvtColor(imgc,cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,11,3)
    (thresh, img_bin1) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin1  # Invert the image

    cv2.imwrite("Image_bin.jpg",img_bin)
   
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40
     
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=6)
    cv2.imwrite("verticle_lines.jpg",verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=5)
    cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=1)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    cv2.imwrite("img_final_bin.jpg",img_final_bin)
    # Find contours for image, which will detect all the boxes
    im2, contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    # (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    idx = 0
    row,col = img_bin.shape
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        
        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (w > 20 and h > 10) and w > 2*h:
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            # cv2.drawContours(imgc, contours, idx, (0, 255,0), 3)
            
            count = 0
            for i in range (y,y+h):
                for j in range (x,x+w):
                    if img_bin1[i][j] > 128:
                        count+=1
            if count >w*h/2:
                cv2.drawContours(imgc, contours, idx, (0, 255,0), 4)
            
            new_img = imgc[y:y+h, x:x+w]
            # cv2.imshow('field',new_img)
            # cv2.waitKey(0)
            print (2*count/(w*h))
        idx+=1
    # For Debugging
    # Enable this line to see all contours.
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    cv2.imwrite("out.jpg", imgc)

infile = sys.argv[1]
box_extraction(infile)