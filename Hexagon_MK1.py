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
# This paper will be used as the background math "Mathematical model and efficient algorithms for object packing problem"

# Import the images from the folder
# For each image convert to binary
import glob
import cv2
import os
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def show_image(img):
    # This function is designed to show the images of the hexagons after they have been binarized 0-1\
    if img.shape[0] > 1080 or img.shape[1] > 1920:# height is 0 width is 1
        # Resize the image to display it on screen
        dim = (round(img.shape[1] / 4), round(img.shape[0] / 4)) # Y, X
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        resized[resized >= 1] = 255
        cv2.imshow("",resized)
        cv2.waitKey()
    else:
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
# print(os.path.relpath(images[0], (os.getcwd() + "/Images/")))
# show_image(img_raw[0])

# For this first attempt, I will create an array that holds the seed image (the one in the centre) and the resulting minimum area
# This is done so that I can see which seed produces the lowest area image, the final output will be saved with the raw image replacing each 
# binary image to produce a render of the resulting assembly

# Need to build the tools / fucntions required to move or rotate the images
def rotate_image(image, angle):
    # Function for rotating an image by a specified angle around the centre of the image (which is required for this senario)
    # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def insert_at(output_img, pos, img):
    # Function to insert one array into another at a specific point
    x1 = pos[0]
    y1 = pos[1]
    x2 = x1 + img.shape[0]
    y2 = y1 + img.shape[1]

    assert x2 <= output_img.shape[0], "the position will make the small matrix exceed the boundaries at x"
    assert y2 <= output_img.shape[1], "the position will make the small matrix exceed the boundaries at y"

    output_img[x1:x2, y1:y2] = img

    return output_img

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def angle_between(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return math.degrees(math.atan(yDiff/xDiff))

def extract_and_measure_edges(img_bin,cnt,corner_params):
    output = cv2.cvtColor(np.zeros_like(img_bin), cv2.COLOR_GRAY2BGR)
    # opencv sub pixel corners
    # find Harris corners
    gray = img_bin
    dst = cv2.cornerHarris(img_bin,corner_params[0],corner_params[1],corner_params[2])
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    rot_angle = 0
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    # print(corners)
    for i in enumerate(corners):
        x = int(i[1][0])
        y = int(i[1][1])
        # print(x,y)
        # color = np.random.randint(0, 255, size=(3, ))
        # color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
        # print(color)
        cv2.circle(output, (x,y), 10, (255,255,0), -1)
    # Plots the centre of the hexagon
    cv2.circle(output, (int(corners[0][0]),int(corners[0][1])), 10, [255,0,0], -1)
    corners = corners[1:]
    if len(corners) != 6:
        global angle_num
        global image_num
        print("error: ",len(corners), " Image: ", image_num, " Angle degs: ", angle_num)
        # exit()
        corner_params[2] = corner_params[2] + 0.0001
        print(corner_params)
        extract_and_measure_edges(img_bin,corner_params)
    # If the angle is greater than the toloerance for flatness, fix it by straightening this line to make it horizontal 
    # Not much should be required as the optical inspection system will try to take images at the required angle anyway
    rot_angle = angle_between(corners[0],corners[1])
    # print(corners[0],corners[1],"Angle rot: ",round(rot_angle,2))
    
    if abs(rot_angle) > 0.2:
        return output, round(rot_angle,2), None,None
    else:
        rot_angle = 0
        
    

    # # Get indices of actual, nearest matching contour points
    corners = sorted([np.argmin(np.linalg.norm(c - cnt.squeeze(), axis=1))for c in corners])
    # Extract edges from contour, and measure their lengths
    lengths = []
    re_ordered_corners = []
    for i_c, c in enumerate(corners):
        # Section the contours array to provide each edge
        # i_c is the iterable object and c is the value of corners at iteration
        if i_c == len(corners) - 1:
            edge = np.vstack([cnt[c:, ...], cnt[0:corners[0], ...]])
        else:
            edge = cnt[c:corners[i_c + 1], ...]
        # Use the edges to get the corners
        # print(edge[0][0][0], edge[0][0][1])
        re_ordered_corners.append((edge[0][0][0], edge[0][0][1]))
        # Provide the location for the text 
        loc = tuple(np.mean(edge.squeeze(), axis=0, dtype=int).tolist())
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # determine the length 
        length = cv2.arcLength(edge, False)
        lengths.append(length)
        cv2.polylines(output, [edge], False, color, 2)
        cv2.putText(output, '{:d} '.format(i_c)+'{:.2f}'.format(length), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

        text = str('{:d} '.format(i_c) + str(re_ordered_corners[i_c]))
        cv2.putText(output, text, re_ordered_corners[i_c], cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        # cv2.putText(output, '{:.2f}'.format(i_c), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

    # # Get the angles for each of the corners
    angles = []
    for i in range(len(re_ordered_corners)):
        if i == 4:
            angles.append(round((180 - getAngle(re_ordered_corners[i],re_ordered_corners[i+1],re_ordered_corners[0])),3))
        elif i == 5:
            angles.append(round((180 - getAngle(re_ordered_corners[i],re_ordered_corners[0],re_ordered_corners[1])),3))
        else:
            angles.append(round((180 - getAngle(re_ordered_corners[i],re_ordered_corners[i+1],re_ordered_corners[i+2])),3))
        text = str('{:d} '.format(i) + str(angles[i]))
        x = re_ordered_corners[i][0] + 10
        y = re_ordered_corners[i][1] + 20
        cv2.putText(output, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
    # print(angles, round(sum(angles)))
    return output, rot_angle, angles, lengths

def straighten_image(img):
    # angle_num = 0
    # image_num = 0
    angle_num = 0#random.randint(0, 360)# add a random angle to each image to be removed by this system
    corner_params = [70,23,0.0415]
    color = [0, 0, 0]
    top, bottom, left, right = [150]*4
    thr = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    print("angle: ",angle_num)
    rot = rotate_image(thr,angle_num) ##*****************************************************
    thr = rot
    contours = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cnt = max(contours, key=cv2.contourArea)
    thr = cv2.drawContours(np.zeros_like(thr), [cnt], -1, 255, 1)

    # Extract and measure edges, and visualize output
    out, rot_angle, corner_angles, side_lengths = extract_and_measure_edges(thr,cnt,corner_params)
    # Rotate the image if there if the first two corners are out of alignment to make all the shapes have a horizontal top
    while abs(rot_angle) > 0.2: 
        rot = rotate_image(thr,rot_angle)
        thr = rot
        contours = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(contours)
        contours = contours[0] if len(contours) == 2 else contours[1]
        cnt = max(contours, key=cv2.contourArea)
        thr = cv2.drawContours(np.zeros_like(thr), [cnt], -1, 255, 1)
        out, rot_angle, corner_angles, side_lengths = extract_and_measure_edges(thr,cnt,corner_params)

    # cv2.imshow("output",out)
    # cv2.waitKey()

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1), plt.imshow(rot), plt.title('Original input image')
    plt.subplot(1, 3, 2), plt.imshow(thr, cmap='gray'), plt.title('Contour needed')
    plt.subplot(1, 3, 3), plt.imshow(out), plt.title('Results')
    plt.tight_layout(), plt.show()
    # Return the aligned image, output image and the angle and length parameters
    return thr, out, corner_angles, side_lengths

processed_images = []
graphics_images = []
corner_angle_Array = []
side_length_Array = []


for i in range(len(img_raw)):
    processed, graphic, corner_angle, side_length = straighten_image(img_raw[i])
    processed_images.append(processed)
    graphics_images.append(graphic)
    corner_angle_Array.append(corner_angle)
    side_length_Array.append(side_length)





# Read and pre-process image, extract contour of shape
# TODO: MODIFY TO FIT YOUR INPUT IMAGES
# img = cv2.imread('2B2m4.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thr = cv2.threshold(gray, 16, 255, cv2.THRESH_BINARY_INV)[1]f

    # plt.figure(figsize=(18, 6))
    # plt.subplot(1, 3, 1), plt.imshow(rot), plt.title('Original input image')
    # plt.subplot(1, 3, 2), plt.imshow(thr, cmap='gray'), plt.title('Contour needed')
    # plt.subplot(1, 3, 3), plt.imshow(out), plt.title('Results')
    # plt.tight_layout(), plt.show()

# # Method for array, choose a seed image, insert into the centre of an array defined by the combined area of all the images
# # Then add the next image in squence and work out the location that will produce the smallest boundary area with no overlap
# print(img_raw[0].shape)
# output_image = [img_raw[0].shape[0] * 3,img_raw[0].shape[1] * 3]# 10um, 10um, RGB i.e. 5000 x 5000 is 50mm x 50mm
# # Create the image that will show the gcode, the size of the image will be the bed size of the printer at 1um
# img = np.zeros(output_image)# Maximum of 150000, 150000, 3 , dtype='uint8'
# img[:] = 0 # Make the image have a black background
# result = insert_at(img, (round(img.shape[0] / 2 - img_raw[0].shape[0] / 2), round(img.shape[1] / 2 - img_raw[0].shape[1] / 2)), img_raw[0])

# show_image(result)
# Import the image, analyse the image and straighten it to a random edge by determining the most linear edge
# Need to standarise the input images and extract the side lengths and angles
# Need to get a centre that is the same distance from each edge to permit the addition of images together to allow the assmbly
# This needs to be in respect to the top left of the image as to orientate it correctly

# Once that is done I need a transformation array that contains all the rotary angles that are possible as well as the centre co-ordinates for each of the hexagons
# Then I need to assemble the hexagons together to get the combination with the smallest area overall, this is acomplished by drawing a contor around after closing to make a complete shape
# The smaller the area inside the contor the better backing and therefore the better assembly