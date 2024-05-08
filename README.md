# Colony_counter

This project is designed to count the number of colonies on a plate using image processing techniques implemented with the OpenCV library.

## Introduction to OpenCV

OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in commercial products. Being a BSD-licensed product, OpenCV makes it easy for businesses to utilize and modify the code.

## Methods Used

This program utilizes two primary methods to differentiate and count colonies on a plate:

### 1. Watershed Algorithm

The Watershed Algorithm is a classical algorithm used for segmentation in image processing. It is particularly useful for separating overlapping objects. In the context of this project, it helps in differentiating touching colonies by treating them as separate catchment basins.

### 2. Circularity Method

The Circularity Method involves calculating the circularity of each detected object in the image. This method is based on the principle that most biological colonies tend to be roughly circular. By setting thresholds for area and circularity, the program can filter out noise and non-colonial artifacts, counting only those structures that qualify as colonies based on their shape.

## How to Use the Program

To use this program, follow these steps:

1. **Start the Program**: Run the script in your Python environment. Ensure that you have all the dependencies installed. Please refer to requirements.txt. To start the program, run `python3 counter.py [method]`. There are two methods to choose from, `w` for Watershed Algorithm and `c` for the Circularity Method. This choice depends on the nature of the colonies and the specific requirements of your analysis.

2. **Select an Image**: Upon running, the program will open a file dialog. Navigate to and select the image of the plate you want to analyze.

3. **Choose the Colony Area**: Once the image is loaded, use the mouse to select the area of the image that contains the colonies. This helps in isolating the region of interest and reduces processing time. Press `SPACE` or `ENTER` button when desired ROI is selected. Press `c` button to cancel selection.

4. **Adjust Parameters**: Interactive sliders will appear, allowing you to fine-tune parameters such as the threshold for foreground detection in the Watershed Algorithm or the minimum and maximum area and circularity in the Circularity Method.

5. **View Results and Count**: After adjusting the parameters, press `p` to print the total number of colonies detected. The results will be displayed visually on the image, and the number of counted colonies will be printed in the console.

## Reference:

https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html <br />
https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html