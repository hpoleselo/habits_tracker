import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from PIL import Image
import pytesseract


def main():
    file_img = r'photo.jpg'
    img = cv2.imread(file_img, 0)
    img.shape

    # Thresholding the image to a binary image
    thresh,img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Inverting the image 
    img_bin = 255-img_bin
    cv2.imwrite('inverted_binary_image.png',img_bin)
    plotting = plt.imshow(img_bin,cmap='gray')
    plt.show()

main()