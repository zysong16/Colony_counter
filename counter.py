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
        
        #img_cropped = img[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
        
        
        
#       kernel = np.array([
#           [0, -1, 0],
#           [-1, 5, -1],
#           [0, -1, 0]], dtype=np.float32)
#       
#       # Sharpen the image
#       img_ker = cv.filter2D(img_cropped, cv.CV_32F, kernel)
        #img_sharp = np.clip(img_ker, 0, 255).astype('uint8')
        
        # Convert to grayscale
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Binarize the grayscale image using the adaptive method
        img_thresh = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,15,4)
        
        img_noise = cv.fastNlMeansDenoising(img_thresh, None, 10, 7, 21)
        #img_thresh = cv.threshold(img_gray,150,255,cv.THRESH_BINARY_INV)[1]
        
#       kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
#       img_cleanup = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel1)
#       
#       gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#       ret,thresh1 = cv.threshold(gray,127,255,1)
        
        r = cv.selectROI("select the area", img)
        
        x1,y1,width,height = r
        cx = (x1+width//2)
        cy = (y1+height//2)
        major_axix = (width)//2
        minor_axis = (height)//2
        
        
        roi = np.zeros(img.shape[:2], np.uint8)
        roi = cv.ellipse(roi, (cx,cy), (major_axix,minor_axis), 0, 0, 360, 255, cv.FILLED)
        
        # Target image; white background
        mask = np.ones_like(img) * 255
        
        # Copy ROI part from original image to target image
        img_cropped = cv.bitwise_and(mask, img, mask=roi) + cv.bitwise_and(mask, mask, mask=~roi)
#       cv.imshow("cropped",img_cropped)
        
        
#       img_color = cv.cvtColor(img_cropped, cv.COLOR_GRAY2BGR)
        gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)
        ret,thresh1 = cv.threshold(gray,127,255,1)
        
        # get contours
        contours,hierarchy = cv.findContours(thresh1,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        
        contours_area = []
        # calculate area and filter into new array
        for con in contours:
            area = cv.contourArea(con)
            if 70 < area < 5000:
                contours_area.append(con)
        
        contours_circles = []
        
        # check if contour is of circular shape
        for con in contours_area:
            perimeter = cv.arcLength(con, True)
            area = cv.contourArea(con)
            if perimeter == 0:
                break
            circularity = 4*3.141592653589793*(area/(perimeter*perimeter))
            if 0.7 < circularity < 1.3:
                contours_circles.append(con)
        #sorting the contours by area size ascending 1 - 100
        cnts = sorted(contours_circles, key=lambda x: cv.contourArea(x))
        print("there are "+str(len(cnts))+" cells!")
        #       for count, c in enumerate(cnts):
#           M = cv.moments(c)
#           cx = int(M['m10']/M['m00'])
#           cy = int(M['m01']/M['m00'])
#           cv.putText(img_color,str(count),(cx-5, cy+5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        cv.drawContours(gray,cnts,-1,(0,255,0),1)
        
#           
        cv.imshow("origin", img)
#       cv.imshow("gray", img_gray)
        cv.imshow("binary", img_thresh)
#       cv.imshow("clean", img_cleanup)
        cv.imshow("noise", img_noise)
        cv.imshow("color", gray)
        
        cv.waitKey(0)  # Wait for a key press
        cv.destroyAllWindows()  # Close all OpenCV windows
        
        
if __name__== '__main__':
    img_preprocess()