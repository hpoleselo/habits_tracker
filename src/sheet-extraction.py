import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from PIL import Image
import pytesseract


def remove_shadows(img):
    # Decomposing the image channels (RGB)
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []

    # We do the remove shadows operation to each channel in the image!
    for plane in rgb_planes:
        # Dilating our image, with a 7x7 kernel, with the dilation the foreground objects will increase (white pixels)
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))

        # By using median blur we remove the noise (shadows) from the background, since our dilated_image has the foreground presence
        bg_img = cv2.medianBlur(dilated_img, 21)

        # Calculates the difference between the original image and the background we filtered out, to get black on white result we use abs function along wih diff
        # Bits(pixels) that are identical are close to 0, i.e: black.
        diff_img = 255 - cv2.absdiff(plane, bg_img)

        # Normalize the image to use the image's full dynamic  range
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    # Merges the 3 channels as a numpy array so we can proceed with the processing in OpenCV
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    #cv2.imwrite('shadows_out.png', result)
    #cv2.imwrite('shadows_out_norm.png', result_norm)
    return result_norm

def sort_contours(cnts, method="left-to-right"):    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # Handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # Handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # Construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key = lambda b:b[1][i], reverse=reverse))
    # Return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def main():
    file_img = r'photo.jpg'
    img = cv2.imread(file_img, 0)
    norm_img = remove_shadows(img)

    # Thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(norm_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Inverting the image 
    img_bin = 255 - img_bin
    cv2.imwrite('inverted_binary_image.png',img_bin)
    plotting = plt.imshow(img_bin,cmap='gray')
    plt.show()

    # Detecting horizontal lines
    # Length(width) of kernel as 100th of total width
    # Defining a vertical kernel to detect all vertical lines of image 
    kernel_len = np.array(img).shape[1]//100
    # Defining a horizontal kernel to detect all horizontal lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    print("Vertical Kernel", ver_kernel)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    print("Horizontal Kernel", hor_kernel)
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    print(kernel)

    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    cv2.imwrite("vertical.jpg",vertical_lines)
    plotting = plt.imshow(image_1,cmap='gray')
    plt.show()

    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    cv2.imwrite("horizontal.jpg",horizontal_lines)
    plotting = plt.imshow(image_2,cmap='gray')
    plt.show()

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    # Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("img_vh.jpg", img_vh)
    bitxor = cv2.bitwise_xor(img,img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    plotting = plt.imshow(bitnot, cmap='gray')
    plt.show()

    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method='top-to-bottom')

    # Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    # Get mean of heights
    mean = np.mean(heights)

    # Create list box to store all boxes in  
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w<1000 and h<500):
            image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            box.append([x,y,w,h])
    plotting = plt.imshow(image, cmap='gray')
    plt.show()

main()