import cv2
import numpy as np
import torch
import os

class Image():

    imageIndex = None
    imageContent = None
    topLeftCoordinates = None
    bottomRightCoordinates = None

    def __init__(self, index, content, topLeftCoordinates, bottomRightCoordinates):
        self.imageIndex = index
        self.imageContent = content
        self.topLeftCoordinates = topLeftCoordinates
        self.bottomRightCoordinates = bottomRightCoordinates


    def print_image_data(self):
        print("Index: " + str(self.imageIndex))
        print(topLeftCoordinates)
        print(bottomRightCoordinates)
        print("---------------------")
