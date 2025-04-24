# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:01:15 2021

@author: Talha

modified: Renxi Chen
"""

import os, glob, re
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#image data and json dir
data_dir = 'D:\\GeoData\\Videos\\Drone\\labelimages'


class_names = ['0']

# set it to ture if you are generating masks for instance segementation.
instance_seg = False  # True or False


# dir to save output masks
output_dir = data_dir


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



# path to json files
file_paths = glob.glob(os.path.join(data_dir, '*.json'))


numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# sort the file paths
file_paths = sorted(file_paths, key=numericalSort)

for i in trange(len(file_paths)):
    print('processing ', file_paths[i])
    # read json file
    with open(file_paths[i]) as json_file:
        j_file = json.load(json_file)

    # get the shapes of all items
    shapes = j_file['shapes']
    # get hight
    h = j_file['imageHeight']
    # get width
    w = j_file['imageWidth']

    sem_mask = np.zeros((h, w)).astype(np.uint8)
    if instance_seg:
        inst_mask = np.zeros((h, w)).astype(np.uint8)
        zero_mask = np.zeros((h, w)).astype(np.uint8)
    mask_name = os.path.basename(file_paths[i])[:-5]

    pixel_values = np.arange(0, len(class_names), 1) + 1

    # now iterate over all the clasess present in json file
    for j in range(len(shapes)):
        item_name = shapes[j]['label']
        # print(item_name)
        item_pts = shapes[j]['points']
        item_pts = np.array(item_pts, np.int32)
        item_pts = item_pts.reshape((-1, 1, 2))

        # now we will select a unique pixel value to represent each class
        color = pixel_values[class_names.index(item_name)].astype(np.uint8)
        # if binary mask, the color is set to 255
        if 1==len(item_name) and False==instance_seg:
            color = 255

        sem_mask = cv2.fillPoly(sem_mask, [item_pts], color=int(color))

        # plt.imshow(sem_mask)
        if instance_seg == False:
            cv2.imwrite(os.path.join(output_dir, mask_name + '.png'), sem_mask)
        if instance_seg:
            # if instance segmentation mask gen is set to true than we will save
            # a 3 channel image
            # 1st channel will have semantic IDs
            # 2nd will have instance IDs
            # 3rd channel will be all zeros
            idx = j + 1
            inst_mask = cv2.fillPoly(inst_mask, [item_pts], color=int(idx))
            # plt.imshow(inst_mask)
            full_mask = cv2.merge((sem_mask, inst_mask, zero_mask))
            cv2.imwrite(os.path.join(output_dir, mask_name + '.png'), full_mask)

print('Finished!')
# img = cv2.imread('C:/Users/Talha/Desktop/output/car4.png',-1)
# plt.imshow(img)