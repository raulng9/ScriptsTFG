import cv2
import numpy as np
import torch
import os

class Patch():

    patchImage = None
    indexOfImage = None
    coordinatesInOriginalImage = None
    coordinatesWithoutHalo = None
    #extra param which I don't yet understand

    def __init__(self, imageForPatch, index, coordinatesOriginal, coordinatesWithoutHalo):
        self.patchImage = imageForPatch
        self.indexOfImage = index
        self.coordinatesInOriginalImage = coordinatesOriginal
        self.coordinatesWithoutHalo = coordinatesWithoutHalo
