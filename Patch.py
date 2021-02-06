import cv2
import numpy as np
import torch
import os

class Patch():

    patchImage = None
    indexOfImage = None
    coordinatesInOriginalImage = None
    relativeCoordinatesWithoutHalo = None
    coordinatesInOriginalImageWithoutHalo = None


    topLeftCoordinate = None
    topRightCoordinate = None
    bottomLeftCoordinate = None
    bottomRightCoordinate = None

    #   Patch dimensions for relative coordinates calculation
    patchWidth = None
    patchHeight = None

    #   Sides that have a halo include: Top right bottom left
    sidesWithHalo = [True,True,True,True]
    haloSize = None

    #   Variables for calculating the halo substractions
    widthOfMainImage = None
    heightOfMainImage = None


    def calculate_initial_relative_coordinates(self):
        self.topLeftCoordinate = [0,0]
        self.topRightCoordinate = [self.patchWidth,0]
        self.bottomLeftCoordinate = [0,self.patchHeight]
        self.bottomRightCoordinate = [self.patchWidth,self.patchHeight]

    def __init__(self, imageForPatch, index, coordinatesOriginal,haloSizeUsed):
        self.patchImage = imageForPatch
        self.indexOfImage = index
        self.coordinatesInOriginalImage = coordinatesOriginal
        self.haloSize = haloSize
        patchDimensions = imageForPatch.shape
        self.patchWidth = patchDimensions[1]
        self.patchHeight = patchDimensions[0]
        self.calculate_initial_relative_coordinates()

    def set_absolute_image_shape(self, widthOfGeneralImage, heightOfGeneralImage):
        self.widthOfMainImage = widthOfGeneralImage
        self.heightOfMainImage = heightOfGeneralImage

    def calculate_relative_coordinates(self):
        if self.sidesWithHalo[0] == False:
            self.topLeftCoordinate = [0,0]
        if self.sidesWithHalo[1] == False:
            self.topRightCoordinate = [self.patchWidth,0]
        if self.sidesWithHalo[2]
        return

    def calculate_absolute_coordinates_without_halo(self):
        return

    def calculate_halo_sides(self):
        if self.coordinatesInOriginalImage[0][0] == 0:
            sidesWithHalo[-1] = False
        if self.coordinatesInOriginalImage[0][1] == 0:
            sidesWithHalo[0] = False
        if self.coordinatesInOriginalImage[1][0] == widthOfMainImage:
            sidesWithHalo[1] = False
        if self.coordinatesInOriginalImage[1][1] == heightOfMainImage:
            sidesWithHalo[2] = False

    def calculate_extra_coordinates(self):
        self.calculate_halo_sides()
        self.calculate_relative_coordinates()
        self.calculate_absolute_coordinates_without_halo()
