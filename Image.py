import cv2
import numpy as np
import torch
import os

class Image():

    imageIndex = None
    imageContent = None
    coordinatesList = []

    def __init__(self, index, content, listOfCoords):
        self.imageIndex = index
        self.imageContent = content
        self.coordinatesList = listOfCoords

    def add_coordinate(self,coordinateToAdd):
        self.coordinatesList.append(coordinateToAdd)

    def print_image_data(self):
        print("Index: " + str(self.imageIndex))
        print(self.coordinatesList)
