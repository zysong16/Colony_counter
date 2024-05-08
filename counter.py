import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog


def img_preprocess():
    
    root = tk.Tk()
    root.withdraw()
    root.update()
    # Open the file dialog to select an image
    file_path = filedialog.askopenfilename()
    root.update()
    if file_path:
        img = cv.imread(file_path)
        
        # Select the area of interest
        r = cv.selectROI("select the area", img)
        x1,y1,width,height = r
        img_cropped = img[y1:y1+height,x1:x1+width]
        
        # Convert the cropped image to grayscale
        img_gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)
        
        # Convert the grayscale image to binary image using basic binary_inverted thresholding and Otsu's thresholding
        inv = cv.bitwise_not(img_gray)
        ret, thresh = cv.threshold(inv,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        
    return img_cropped, thresh
        
        
def watershed(img, img_bin):
    # Remove the background noise
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(img_bin, cv.MORPH_OPEN, kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    
    # Set the default value for sure foreground threshold coefficient
    fg_val = 70
    result = 0
    def update(_):
        fg_val = cv.getTrackbarPos('fg thresh coeff', window_name)
        
        ret, sure_fg = cv.threshold(dist_transform,(fg_val/100)*dist_transform.max(),255,0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg,sure_fg)
        
        # Marker labeling
        ret, markers = cv.connectedComponents(sure_fg)
        
        cv.imshow("foreground",sure_fg)
        #cv.moveWindow("foreground", 50, -30)
        cv.imshow("background",sure_bg)
        #cv.moveWindow("background", sure_fg.shape[1]+90, -30)
        
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        
        # Mark the region of unknown with zero
        markers[unknown==255] = 0
        
        # Apply the watershed algorithm
        markers = cv.watershed(img, markers)
        img_tmp = img.copy()
        img_tmp[markers == -1] = [255,0,255]
        
        # count the colony numbers
        labels = np.unique(markers)
        nonlocal result
#       old_result = result
        result = len(labels) - 2 # ignore the background and the unknown region
        
#       if result!=old_result:
#           print(f"The total number of colonies is {result}.")
        
        
        cv.imshow(window_name, img_tmp)
        #cv.moveWindow(window_name, 50, sure_fg.shape[0]+40)
        
    window_name = 'Colony Counter'
    cv.namedWindow(window_name,cv.WINDOW_NORMAL)
    cv.createTrackbar('fg thresh coeff', window_name, 0, 100, update)
    cv.setTrackbarMin('fg thresh coeff', window_name, 1)
    
    cv.setTrackbarPos("fg thresh coeff", window_name, fg_val)
    
    while True:
        k= cv.waitKey(0) & 0xFF # Wait for a key press
        if k == ord("p"):
            print(f"The total number of colonies is {result}.")
        else:
            cv.destroyAllWindows()  # Close all OpenCV windows
            break
        
    
def circularity(img, img_bin):
    # get contours
    contours,hierarchy = cv.findContours(img_bin,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
    A_min = 70
    A_max = 5000
    C_min = 70
    C_max = 130
    result = 0
    
    def update(_):
        A_max = cv.getTrackbarPos('area max', window_name)
        A_min = cv.getTrackbarPos('area min', window_name)
        C_max = cv.getTrackbarPos('circularity max', window_name)
        C_min = cv.getTrackbarPos('circularity min', window_name)
        
        contours_area = []
        # calculate area and filter into new array
        for con in contours:
            area = cv.contourArea(con)
            if A_min < area < A_max:
                contours_area.append(con)
                
        contours_circles = []
        
        # check if contour is of circular shape
        for con in contours_area:
            perimeter = cv.arcLength(con, True)
            area = cv.contourArea(con)
            if perimeter == 0:
                break
            #area divided by perimeter squared, then multiply by 4pi. The closer to 1 the more rounded the shape is.
            circularity = 4*3.141592653589793*(area/(perimeter*perimeter))
            if C_min/100 < circularity < C_max/100:
                contours_circles.append(con)
                
        #sorting the contours by area size ascending 1 - 100
        cnts = sorted(contours_circles, key=lambda x: cv.contourArea(x))

        img_tmp = img.copy()
        cv.drawContours(img_tmp,cnts,-1,(0,255,0),1)
        
        nonlocal result
        result = str(len(cnts))
#
        
        cv.imshow(window_name, img_tmp)
        #cv.waitKey(10)

    
    window_name = 'Colony Counter'
    cv.namedWindow(window_name,cv.WINDOW_NORMAL)
    
    cv.createTrackbar('area min', window_name, 0, 140, update)
    cv.createTrackbar('area max', window_name, 0, 10000, update)
    cv.setTrackbarMin('area max', window_name, 1000)
    
    cv.createTrackbar('circularity min', window_name, 0, 99, update)
    cv.createTrackbar('circularity max', window_name, 0, 200, update)
    cv.setTrackbarMin('circularity max', window_name, 100)
    
    cv.setTrackbarPos('area min', window_name, A_min)
    cv.setTrackbarPos('area max', window_name, A_max)
    cv.setTrackbarPos('circularity min', window_name, C_min)
    cv.setTrackbarPos('circularity max', window_name, C_max)
    
    while True:
        k= cv.waitKey(0) & 0xFF # Wait for a key press
        if k == ord("p"):
            print(f"The total number of colonies is {result}.")
        else:
            cv.destroyAllWindows()  # Close all OpenCV windows
            break


def count(algorithm):
    img_cropped, img_bin = img_preprocess()
    if algorithm == 'watershed':
        watershed(img_cropped, img_bin)
    elif algorithm == 'circularity':
        circularity(img_cropped, img_bin)
    else:
        error('Undefined algorithm')


if __name__== '__main__':
    count('watershed')
    

    