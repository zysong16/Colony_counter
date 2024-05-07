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
        
        r = cv.selectROI("select the area", img)
        x1,y1,width,height = r
        
        #select ellipse shape
#       cx = (x1+width//2)
#       cy = (y1+height//2)
#       major_axix = (width)//2
#       minor_axis = (height)//2
#       roi = np.zeros(img.shape[:2], np.uint8)
#       roi = cv.ellipse(roi, (cx,cy), (major_axix,minor_axis), 0, 0, 360, 255, cv.FILLED)
        
        
        roi = np.zeros(img.shape[:2], np.uint8)
        roi = cv.rectangle(roi, (x1,y1), (x1+width,y1+height),255, cv.FILLED)
        
    
        
        # Target image; white background
        mask = np.ones_like(img) * 255
        
        # Copy ROI part from original image to target image
        #img_cropped = cv.bitwise_and(mask, img, mask=roi) + cv.bitwise_and(mask, mask, mask=~roi)
#       cv.imshow("cropped",img_cropped)
        
        img_cropped = img[y1:y1+height,x1:x1+width]
        
#       img_color = cv.cvtColor(img_cropped, cv.COLOR_GRAY2BGR)
        gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)
        
        #thresh1 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        #    cv.THRESH_BINARY,15,4)
        
        inv = cv.bitwise_not(gray)
        ret, thresh3 = cv.threshold(inv,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        
        # noise removal
        kernel = np.ones((1,1),np.uint8)
        opening = cv.morphologyEx(thresh3,cv.MORPH_OPEN,kernel, iterations = 2)
        
        # sure background area
        sure_bg = cv.dilate(opening,kernel,iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
#       ret, sure_fg = cv.threshold(dist_transform,0.05*dist_transform.max(),255,0)
#       
#       # Finding unknown region
#       sure_fg = np.uint8(sure_fg)
#       unknown = cv.subtract(sure_bg,sure_fg)
#       # Marker labelling
#       ret, markers = cv.connectedComponents(sure_fg)
#       
#       cv.imshow("foreground",sure_fg)
#       cv.imshow("background",sure_bg)
#       # Add one to all labels so that sure background is not 0, but 1
#       markers = markers+1
#       
#       # Now, mark the region of unknown with zero
#       markers[unknown==255] = 0
#       
#       markers = cv.watershed(img_cropped,markers)
#       img_cropped[markers == -1] = [255,0,0]
        
        #cv.imshow('result',img_cropped)
        
        
        
        
        # get contours
        contours,hierarchy = cv.findContours(thresh3,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        
        
        fg_val = 5
        A_min = 70
        A_max = 5000
        C_min = 70
        C_max = 130
        
        def update(_):
            fg_val = cv.getTrackbarPos('foreground slider', window_name)
            
            
            ret, sure_fg = cv.threshold(dist_transform,(fg_val/100)*dist_transform.max(),255,0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv.subtract(sure_bg,sure_fg)
            # Marker labelling
            ret, markers = cv.connectedComponents(sure_fg)
            
            cv.imshow("foreground",sure_fg)
            cv.imshow("background",sure_bg)
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1
            
            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0
            
            markers = cv.watershed(img_cropped,markers)
            img_tmp = img_cropped.copy()
            img_tmp[markers == -1] = [255,0,0]
            
            
            
            
            A_max = cv.getTrackbarPos('cell area max', window_name)
            A_min = cv.getTrackbarPos('cell area min', window_name)
            C_max = cv.getTrackbarPos('cell circularity max', window_name)
            C_min = cv.getTrackbarPos('cell circularity min', window_name)
                
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
            print("there are "+str(len(cnts))+" cells!")
            #cv.drawContours(img_tmp,cnts,-1,(0,255,0),1)
            cv.imshow(window_name, img_tmp)
            cv.waitKey(10)
            
            
        window_name = 'cell counter'
        cv.namedWindow(window_name,cv.WINDOW_NORMAL)
        cv.createTrackbar('foreground slider', window_name, 0, 100, update)
        cv.setTrackbarMin('foreground slider', window_name, 1)
        
        cv.createTrackbar('cell area min', window_name, 0, 140, update)
        cv.createTrackbar('cell area max', window_name, 0, 10000, update)
        cv.setTrackbarMin('cell area max', window_name, 1000)
    
        cv.createTrackbar('cell circularity min', window_name, 0, 99, update)
        cv.createTrackbar('cell circularity max', window_name, 0, 200, update)
        cv.setTrackbarMin('cell circularity max', window_name, 100)
        
        cv.setTrackbarPos("foreground slider", window_name, fg_val)
        cv.setTrackbarPos("cell area min", window_name, A_min)
        cv.setTrackbarPos("cell area max", window_name, A_max)
        cv.setTrackbarPos("cell circularity min", window_name, C_min)
        cv.setTrackbarPos("cell circularity max", window_name, C_max)
        
        
#       cv.imshow("origin", img)
#       cv.imshow("gray", img_gray)
#       cv.imshow("binary", img_thresh)
#       cv.imshow("clean", img_cleanup)
        #cv.imshow("noise", img_noise)
        cv.waitKey(0)  # Wait for a key press
        cv.destroyAllWindows()  # Close all OpenCV windows
        
        
if __name__== '__main__':
    img_preprocess()
    

##       kernel = np.array([
##           [0, -1, 0],
##           [-1, 5, -1],
##           [0, -1, 0]], dtype=np.float32)
##       
##       # Sharpen the image
##       img_ker = cv.filter2D(img_cropped, cv.CV_32F, kernel)
#       #img_sharp = np.clip(img_ker, 0, 255).astype('uint8')
#       
#       # Convert to grayscale
#       img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#       
#       # Binarize the grayscale image using the adaptive method
#       img_thresh = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#           cv.THRESH_BINARY,15,4)
#       
##       img_noise = cv.fastNlMeansDenoising(img_thresh, None, 10, 7, 21)
#       # noise removal
#       kernel = np.ones((3,3), np.uint8)
#       opening = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel, iterations=2)
#       
#       # find sure background area
#       sure_bg = cv.dilate(opening, kernel, iterations=3)
#       
#       # find sure foreground area
#       dist_transform = cv.distanceTransform(opening, cv.DIST_L2,5)
#       ret1, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
#       
#       # find unknown region
#       sure_fg = np.uint8(sure_fg)
#       unknown = cv.subtract(sure_bg, sure_fg)
#       
#       # marker labeling
#       ret2, markers = cv.connectedComponents(sure_fg)
#       
#       # add one to all labels so that sure background is not 0 but 1
#       markers = markers + 1
#       
#       # Mark the region of unknown with 0
#       markers[unknown==255] = 0
#       
#       
#       markers = cv.watershed(img, markers)
#       img[markers == -1] = [255, 0, 0]
    
#       kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
#       img_cleanup = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel1)
#       
#       gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#       ret,thresh1 = cv.threshold(gray,127,255,1)
    