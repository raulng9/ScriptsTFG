import cv2
import numpy as np
import torch
import os

class Patch():

    patchImageName = None
    coordinatesInOriginalImage = None

    topLeftCoordinates = [None,None]
    bottomRightCoordinates = [None,None]

    topLeftRelative = [None,None]
    bottomRightRelative = [None,None]

    topLeftAbsoluteWithoutHalo = [None,None]
    bottomRightAbsoluteWithoutHalo = [None,None]

    #   Patch dimensions for relative coordinates calculation
    width = None
    height = None

    #   Sides that have a halo include: Top right bottom left
    sidesWithHalo = [True,True,True,True]
    haloSize = None

    #   Variables for calculating the halo substractions
    widthOfMainImage = None
    heightOfMainImage = None


    def calculate_initial_relative_coordinates(self):
        self.topLeftRelative = [0 + self.haloSize, 0 + self.haloSize]
        self.bottomRightRelative = [self.width - self.haloSize, self.height - self.haloSize]

    def calculate_initial_coordinates_without_halo(self):
        self.topLeftAbsoluteWithoutHalo = self.coordinatesInOriginalImage[0]
        self.bottomRightAbsoluteWithoutHalo = self.coordinatesInOriginalImage[1]

    def calculate_dimensions_of_patch(self):
        self.width = self.bottomRightCoordinates[0] - self.topLeftCoordinates[0]
        self.height = self.bottomRightCoordinates[1] - self.topLeftCoordinates[1]
        return [self.width,self.height]

    def __init__(self, imageNameForPatch, coordinatesOriginal,haloSizeUsed):
        self.patchImageName = imageNameForPatch
        self.coordinatesInOriginalImage = coordinatesOriginal
        self.haloSize = haloSizeUsed
        self.topLeftCoordinates = coordinatesOriginal[0]
        self.bottomRightCoordinates = coordinatesOriginal[1]

    def set_absolute_image_shape(self, widthOfGeneralImage, heightOfGeneralImage):
        self.widthOfMainImage = widthOfGeneralImage
        self.heightOfMainImage = heightOfGeneralImage

    def calculate_relative_coordinates(self):
        if self.sidesWithHalo[0] == False:
            self.topLeftRelative[1] -= self.haloSize
        if self.sidesWithHalo[1] == False:
            self.bottomRightRelative[0] += self.haloSize
        if self.sidesWithHalo[2] == False:
            self.bottomRightRelative[1] += self.haloSize
        if self.sidesWithHalo[3] == False:
            self.topLeftRelative[0] -= self.haloSize


    def calculate_absolute_coordinates_without_halo(self):
        if self.sidesWithHalo[0] == True:
            self.topLeftAbsoluteWithoutHalo[1] += self.haloSize
        if self.sidesWithHalo[1] == True:
            self.bottomRightAbsoluteWithoutHalo[0] -= self.haloSize
        if self.sidesWithHalo[2] == True:
            self.bottomRightAbsoluteWithoutHalo[1] -= self.haloSize
        if self.sidesWithHalo[3] == True:
            self.topLeftAbsoluteWithoutHalo[0] += self.haloSize


    def calculate_halo_sides(self):
        if self.coordinatesInOriginalImage[0][0] == 0:
            self.sidesWithHalo[-1] = False
        if self.coordinatesInOriginalImage[0][1] == 0:
            self.sidesWithHalo[0] = False
        if self.coordinatesInOriginalImage[1][0] == self.widthOfMainImage:
            self.sidesWithHalo[1] = False
        if self.coordinatesInOriginalImage[1][1] == self.heightOfMainImage:
            self.sidesWithHalo[2] = False


    def calculate_extra_coordinates(self):
        self.calculate_initial_coordinates_without_halo()
        self.calculate_initial_relative_coordinates()
        self.calculate_halo_sides()
        self.calculate_relative_coordinates()
        self.calculate_absolute_coordinates_without_halo()


#
