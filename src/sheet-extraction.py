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

main()