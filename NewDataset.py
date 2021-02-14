import cv2
import numpy as np
import torch
import os
from Patch import Patch
from Image import Image
import glob


class Dataset(torch.utils.data.Dataset):

    #   Constant declaration
    maxDimensionsForGPU = 512
    HALO_SIZE = 20

    listOfImages = []
    listOfPatches = []
    listOfGTImages = []
    tempGTPatches = []
    listOfDirectoriesForGT = ["FrameRegion", "ImageRegion", "GraphicRegion", "TextRegion/caption", "TextRegion/heading", "TextRegion/paragraph"]

    imageCopyForRects = None

    #   Settings for the testing windows
    WINDOW_SIZE = 300

    def len_images(self):
        return len(self.listOfImages)

    def len_patches(self):
        return len(self.listOfPatches)

    def get_image(self,i):
        return self.listOfImages[i]

    def get_patch(self,i):
        return self.listOfPatches[i]


    def get_patches_from_image(self,imageToPatch,lengthOfHalo):
        dimensionsOfImage = imageToPatch.imageContent.shape
        height = dimensionsOfImage[0]
        width = dimensionsOfImage[1]
        indexForPatches = imageToPatch.imageIndex
        actualImage = imageToPatch.imageContent
        channels = dimensionsOfImage[2]
        #   If the size of the current image is already small enough for our GPU
        #   we can create the patch from it
        if width * height < pow(self.maxDimensionsForGPU, 2):
            #   Create patch and calculate all the coordinates for its parameters
            newPatch = Patch(imageToPatch.imageContent,imageToPatch.imageIndex,[imageToPatch.topLeftCoordinates,imageToPatch.bottomRightCoordinates],lengthOfHalo)
            newPatch.set_absolute_image_shape(width,height)
            newPatch.calculate_extra_coordinates()
            #   Add new patch to the general list
            self.listOfPatches.append(newPatch)
            #   Draw rectangles representing the chopping for testing purposes
            cv2.rectangle(self.imageCopyForRects, (int(imageToPatch.topLeftCoordinates[0]),int(imageToPatch.topLeftCoordinates[1])), (int(imageToPatch.bottomRightCoordinates[0]),int(imageToPatch.bottomRightCoordinates[1])), (255,0,0), 10)
            #print(newPatch.coordinatesInOriginalImage)
            #print("------------------------")
        else:
            #   Computation of the coordinates of the patches with respect to the image
            firstPatchArea = actualImage[0:int(0+height/2+lengthOfHalo), 0:int(0+width/2+lengthOfHalo)]
            coordinatesFirstPatch = [[imageToPatch.topLeftCoordinates[0],imageToPatch.topLeftCoordinates[1]], [imageToPatch.topLeftCoordinates[0] + width/2 +lengthOfHalo,imageToPatch.topLeftCoordinates[1] + height/2 + lengthOfHalo]]
            secondPatchArea = actualImage[0:int(0+height/2+lengthOfHalo), int(0+width/2-lengthOfHalo):width]
            coordinatesSecondPatch = [[imageToPatch.topLeftCoordinates[0]+width/2-lengthOfHalo,imageToPatch.topLeftCoordinates[1]], [imageToPatch.bottomRightCoordinates[0],imageToPatch.topLeftCoordinates[1] + height/2 + lengthOfHalo]]
            thirdPatchArea = actualImage[int(height/2-lengthOfHalo):height, 0:int(width/2+lengthOfHalo)]
            coordinatesThirdPatch = [[imageToPatch.topLeftCoordinates[0],imageToPatch.topLeftCoordinates[1] + height/2 -lengthOfHalo], [imageToPatch.topLeftCoordinates[0]+width/2+lengthOfHalo,imageToPatch.bottomRightCoordinates[1]]]
            fourthPatchArea = actualImage[int(height/2-lengthOfHalo):height, int(width/2-lengthOfHalo):width]
            coordinatesFourthPatch = [[imageToPatch.topLeftCoordinates[0]+width/2-lengthOfHalo,imageToPatch.topLeftCoordinates[1] + height/2 -lengthOfHalo], [imageToPatch.bottomRightCoordinates[0],imageToPatch.bottomRightCoordinates[1]]]

            #   Creation of images for possible further cropping
            firstImage = Image(indexForPatches,firstPatchArea,coordinatesFirstPatch[0],coordinatesFirstPatch[1])
            secondImage = Image(indexForPatches,secondPatchArea,coordinatesSecondPatch[0],coordinatesSecondPatch[1])
            thirdImage = Image(indexForPatches,thirdPatchArea,coordinatesThirdPatch[0],coordinatesThirdPatch[1])
            fourthImage = Image(indexForPatches,fourthPatchArea,coordinatesFourthPatch[0],coordinatesFourthPatch[1])

            #   Recursive call for further chopping if necessary
            self.get_patches_from_image(firstImage,self.HALO_SIZE)
            self.get_patches_from_image(secondImage,self.HALO_SIZE)
            self.get_patches_from_image(thirdImage,self.HALO_SIZE)
            self.get_patches_from_image(fourthImage,self.HALO_SIZE)


    def get_patches_from_ground_truth_image(self,imageToPatch,lengthOfHalo):
        dimensionsOfImage = imageToPatch.imageContent.shape
        height = dimensionsOfImage[0]
        width = dimensionsOfImage[1]
        indexForPatches = imageToPatch.imageIndex
        actualImage = imageToPatch.imageContent
        channels = dimensionsOfImage[2]
        #   If the size of the current image is already small enough for our GPU
        #   we can create the patch from it
        if width * height < pow(self.maxDimensionsForGPU, 2):
            #   Create patch and calculate all the coordinates for its parameters
            newPatch = Patch(imageToPatch.imageContent,imageToPatch.imageIndex,[imageToPatch.topLeftCoordinates,imageToPatch.bottomRightCoordinates],lengthOfHalo)
            newPatch.set_absolute_image_shape(width,height)
            newPatch.calculate_extra_coordinates()
            #   Add new patch to the general list
            self.tempGTPatches.append(newPatch)
            #   Draw rectangles representing the chopping for testing purposes
            cv2.rectangle(self.imageCopyForRects, (int(imageToPatch.topLeftCoordinates[0]),int(imageToPatch.topLeftCoordinates[1])), (int(imageToPatch.bottomRightCoordinates[0]),int(imageToPatch.bottomRightCoordinates[1])), (255,0,0), 10)
            #print(newPatch.coordinatesInOriginalImage)
            #print("------------------------")
        else:
            #   Computation of the coordinates of the patches with respect to the image
            firstPatchArea = actualImage[0:int(0+height/2+lengthOfHalo), 0:int(0+width/2+lengthOfHalo)]
            coordinatesFirstPatch = [[imageToPatch.topLeftCoordinates[0],imageToPatch.topLeftCoordinates[1]], [imageToPatch.topLeftCoordinates[0] + width/2 +lengthOfHalo,imageToPatch.topLeftCoordinates[1] + height/2 + lengthOfHalo]]
            secondPatchArea = actualImage[0:int(0+height/2+lengthOfHalo), int(0+width/2-lengthOfHalo):width]
            coordinatesSecondPatch = [[imageToPatch.topLeftCoordinates[0]+width/2-lengthOfHalo,imageToPatch.topLeftCoordinates[1]], [imageToPatch.bottomRightCoordinates[0],imageToPatch.topLeftCoordinates[1] + height/2 + lengthOfHalo]]
            thirdPatchArea = actualImage[int(height/2-lengthOfHalo):height, 0:int(width/2+lengthOfHalo)]
            coordinatesThirdPatch = [[imageToPatch.topLeftCoordinates[0],imageToPatch.topLeftCoordinates[1] + height/2 -lengthOfHalo], [imageToPatch.topLeftCoordinates[0]+width/2+lengthOfHalo,imageToPatch.bottomRightCoordinates[1]]]
            fourthPatchArea = actualImage[int(height/2-lengthOfHalo):height, int(width/2-lengthOfHalo):width]
            coordinatesFourthPatch = [[imageToPatch.topLeftCoordinates[0]+width/2-lengthOfHalo,imageToPatch.topLeftCoordinates[1] + height/2 -lengthOfHalo], [imageToPatch.bottomRightCoordinates[0],imageToPatch.bottomRightCoordinates[1]]]

            #   Creation of images for possible further cropping
            firstImage = Image(indexForPatches,firstPatchArea,coordinatesFirstPatch[0],coordinatesFirstPatch[1])
            secondImage = Image(indexForPatches,secondPatchArea,coordinatesSecondPatch[0],coordinatesSecondPatch[1])
            thirdImage = Image(indexForPatches,thirdPatchArea,coordinatesThirdPatch[0],coordinatesThirdPatch[1])
            fourthImage = Image(indexForPatches,fourthPatchArea,coordinatesFourthPatch[0],coordinatesFourthPatch[1])

            #   Recursive call for further chopping if necessary
            self.get_patches_from_image(firstImage,self.HALO_SIZE)
            self.get_patches_from_image(secondImage,self.HALO_SIZE)
            self.get_patches_from_image(thirdImage,self.HALO_SIZE)
            self.get_patches_from_image(fourthImage,self.HALO_SIZE)


    def add_image_to_dataset(self, pathToImage):
        imageForPatching = cv2.imread(pathToImage)
        self.listOfImages.append(imageForPatching)
        print(pathToImage)
        self.imageCopyForRects = imageForPatching.copy()
        originCoordinates = [0,0]
        destinationCoordinates = [imageForPatching.shape[1],imageForPatching.shape[0]]

        currentIndex = self.len_images() + 1
        imageToAdd = Image(currentIndex,imageForPatching,originCoordinates,destinationCoordinates)

        cv2.namedWindow('original',cv2.WINDOW_NORMAL)
        cv2.imshow('original', imageToAdd.imageContent)
        cv2.resizeWindow('original', self.WINDOW_SIZE*2,self.WINDOW_SIZE*2)
        print("Generating patches from image...")
        self.get_patches_from_image(imageToAdd,self.HALO_SIZE)


    def add_ground_truth_to_dataset(self, pathToGTImage):
        imageForPatching = cv2.imread(pathToGTImage)
        print(pathToGTImage)
        self.listOfGTImages.append(imageForPatching)
        originCoordinates = [0,0]
        destinationCoordinates = [imageForPatching.shape[1],imageForPatching.shape[0]]

        currentIndex = self.len_images() + 1
        imageToAdd = Image(currentIndex,imageForPatching,originCoordinates,destinationCoordinates)

        cv2.namedWindow('gt',cv2.WINDOW_NORMAL)
        cv2.imshow('gt', imageToAdd.imageContent)
        cv2.resizeWindow('gt', self.WINDOW_SIZE*2,self.WINDOW_SIZE*2)
        cv2.waitKey(0)

        print("Generating patches from GT image...")
        self.get_patches_from_ground_truth_image(imageToAdd,self.HALO_SIZE)
        #   Now tempGTPatches has all the patches for the current GT image
        #print(len(self.tempGTPatches))
        #   We combine them, one in each channel

    def load_ground_truth_for_image(self, imageForGroundTruth):
        print("Loading ground truth")
        counterGTFiles = 0
        groundTruthFinalImage = None
        for root, dirs, files in os.walk(self.groundtruth_path):
             for imageGT in files:
                 if imageGT.endswith('.png') and imageGT == imageForGroundTruth:
                     self.add_ground_truth_to_dataset(self.groundtruth_path + "/" + self.listOfDirectoriesForGT[counterGTFiles]+ "/" + imageGT)
                     counterGTFiles += 1
        print("Loaded ground truth with " + str(counterGTFiles) + " files")
        print("-----------------------")


    def load_input(self, inputPath):
        print("Loading input...")
        counterInputFiles = 0
        for fileInDataset in os.listdir(inputPath):
            if fileInDataset.endswith(".png"):
                self.add_image_to_dataset(inputPath + "/" + fileInDataset)
                print("Input:")
                print(fileInDataset)
                self.load_ground_truth_for_image(fileInDataset)
                counterInputFiles += 1
        print(str(counterInputFiles) + " images have been loaded, along with their corresponding ground truths")


    def create_dataset(self, input_path, groundtruth_path=None):
        self.input_path = input_path
        self.groundtruth_path = groundtruth_path
        self.files = []
        self.load_input(input_path)


if __name__ == '__main__':
    datasetObject = Dataset()
    datasetObject.create_dataset("Dataset/Training","Dataset/GroundTruths")












#
