# Python file with commands for github

# To setup new repo
# Make new folder then navigate to it in Bash
# type "git init" once inside folder
# Then "git add filename" (ensure there are sopme files in the folder)
# then "git commit -m "message"

# Next got to github and create new repo
# Make sure that the name is the same
# Then follow the oncreen instructions
# git remote add origin https://github.com/lambcd15/Hexagon-Packing.git
# git branch -M main
# git push -u origin main

# To cerate another branch "git checkout -b <your-new-branch-name>"

# This code is designed to pack irregular polygons that happen to be hexagons into a hexagonal shape
# The aim is to minimise the spacing between the hexagons or have the smallest area

# Import the images from the folder
# For each image convert to binary
import glob
import cv2
import os
from PIL import Image
import numpy as np

def show_image(img):
    # This function is designed to show the images of the hexagons after they have been binarised
    img[img >= 1] = 255
    cv2.imshow("",img)
    cv2.waitKey()

# Find all the images in folder
images = glob.glob(os.getcwd() + "/Images/" + '/*.png')
# Check to make sure that the folder is not empty
if len(images) == 0:
    exit()
# Create an array for the filepath for the images to be read in
read_array = []
# Create an array to store all the images
img_raw = []

for i in range(len(images)):
    # Fix the windows issue with slashes for all the images
    read_array.append(images[i].replace("\\" , "/"))
    img_colour = cv2.imread(read_array[i],-1)
    alpha = img_colour[:,:,3] # extract it
    # binary = ~alpha   # invert b/w
    alpha[alpha > 0] = 1 # convert from 255 to 1 to permit the overlapping detection
    # Add each image to the array
    img_raw.append(alpha)
# Print the name for the first image in the array
print(os.path.relpath(images[0], (os.getcwd() + "/Images/")))
show_image(img_raw[0])

