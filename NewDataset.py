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
    listOfDirectoriesForGT = []

    imageCopyForRects = None

    #settings for the testing windows
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
            print(newPatch.coordinatesInOriginalImage)
            print("------------------------")
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

        cv2.namedWindow('black',cv2.WINDOW_NORMAL)
        cv2.imshow("black", self.imageCopyForRects)
        cv2.resizeWindow('black', self.WINDOW_SIZE*3,self.WINDOW_SIZE*3)
        cv2.waitKey(0)

        #saveName = "chopping_Result_" + str(currentIndex) + ".png"
        #cv2.imwrite(saveName, self.imageCopyForRects)

    def load_ground_truth(self, imageForGroundTruth):
        print("Loading ground truth")
        counterGTFiles = 0
        for root, dirs, files in os.walk(self.groundtruth_path):
             for imageGT in files:
                 if imageGT.endswith('.png') and imageGT == imageForGroundTruth:
                     #print(imageGT)
                     counterGTFiles += 1
        print("Loaded ground truth with " + str(counterGTFiles) + " files")
        print("-----------------------")


    def load_input(self, inputPath):
        print("Loading input...")
        counterInputFiles = 0
        for fileInDataset in os.listdir(inputPath):
            if fileInDataset.endswith(".png"):
                #self.add_image_to_dataset(fileInDataset)
                print("Input:")
                print(fileInDataset)
                self.load_ground_truth(fileInDataset)
                counterInputFiles += 1
        print(str(counterInputFiles) + " images have been loaded, along with their corresponding ground truths")


    def create_dataset(self, input_path, groundtruth_path=None):
        self.input_path = input_path
        self.groundtruth_path = groundtruth_path
        self.files = []
        self.load_input(input_path)
        #self.load_ground_truth(groundtruth_path)
        # print("starting")
        # imagePaths = glob.glob("*.png")
        # for image in imagePaths:
        #     self.imagePathToTest = image
        #     self.add_image_to_dataset()
        #     print("Image added to dataset")
        # print('Total files in dataset: {}'.format(self.len_images()))


if __name__ == '__main__':
    datasetObject = Dataset()
    datasetObject.create_dataset("Dataset/Training","Dataset/GroundTruths")












#
