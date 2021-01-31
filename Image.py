import cv2
import numpy as np
import torch
import os

class Image():

    imageIndex = None
    imageContent = None
    coordinatesList = []

    def __init__(self, index, content):
        self.imageIndex = index
        self.imageContent = content

    def add_coordinate(self,coordinateToAdd):
        self.coordinatesList.append(coordinateToAdd)
