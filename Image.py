import cv2
import numpy as np
import torch
import os

class Image():

    imageIndex = None
    imageContent = None


    def __init__(self, index, content):
        self.imageIndex = index
        self.imageContent = content
