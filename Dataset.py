import cv2
import numpy as np
import torch
import os
from Patch import Patch
from Image import Image

class Dataset(torch.utils.data.Dataset):

    imagePathToTest = "Image_1.png"
    listOfImages = []
    listOfPatches = []

    maxDimensionsForGPU = 1024
    HALO_SIZE = 20

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
        if width * height < pow(self.maxDimensionsForGPU, 2):
            newPatch = Patch(imageToPatch.imageContent,imageToPatch.imageIndex,[imageToPatch.topLeftCoordinates,imageToPatch.bottomRightCoordinates],1)
            #append to list of patches
            self.listOfPatches.append(newPatch)
            cv2.imshow('ok', newPatch.patchImage)
            print(newPatch.coordinatesInOriginalImage)
            print("------------------------")

        else:
            #   Computation of the coordinates of the patches with respect to the image
            firstPatchArea = actualImage[0:int(0+height/2+lengthOfHalo), 0:int(0+width/2+lengthOfHalo)]
            coordinatesFirstPatch = [[imageToPatch.topLeftCoordinates[0],imageToPatch.topLeftCoordinates[1]], [imageToPatch.topLeftCoordinates[0] + width/2 +lengthOfHalo,imageToPatch.topLeftCoordinates[1] + height/2 + lengthOfHalo]]

            secondPatchArea = actualImage[0:int(0+height/2+lengthOfHalo), int(0+width/2-lengthOfHalo):width]
            coordinatesSecondPatch = [[imageToPatch.topLeftCoordinates[0]+width/2-lengthOfHalo,imageToPatch.topLeftCoordinates[1]], [imageToPatch.bottomRightCoordinates[0],imageToPatch.topLeftCoordinates[1] + height/2 + lengthOfHalo]]

            thirdPatchArea = actualImage[int(height/2-lengthOfHalo):height, 0:int(width/2+lengthOfHalo)]
            coordinatesThirdPatch = [[imageToPatch.topLeftCoordinates[0],imageToPatch.topLeftCoordinates[1] + height/2], [imageToPatch.topLeftCoordinates[0]+width/2+lengthOfHalo,imageToPatch.bottomRightCoordinates[1]]]

            fourthPatchArea = actualImage[int(height/2-lengthOfHalo):height, int(width/2-lengthOfHalo):width]
            coordinatesFourthPatch = [[imageToPatch.topLeftCoordinates[0]+width/2-lengthOfHalo,imageToPatch.topLeftCoordinates[1] + height/2], [imageToPatch.bottomRightCoordinates[0],imageToPatch.bottomRightCoordinates[1]]]


            #   Creation of images for possible further cropping
            firstImage = Image(indexForPatches,firstPatchArea,coordinatesFirstPatch[0],coordinatesFirstPatch[1])
            secondImage = Image(indexForPatches,secondPatchArea,coordinatesSecondPatch[0],coordinatesSecondPatch[1])
            thirdImage = Image(indexForPatches,thirdPatchArea,coordinatesThirdPatch[0],coordinatesThirdPatch[1])
            fourthImage = Image(indexForPatches,fourthPatchArea,coordinatesFourthPatch[0],coordinatesFourthPatch[1])


            #firstImage.print_image_data()
            #secondImage.print_image_data()
            #thirdImage.print_image_data()
            #fsdfsfourthImage.print_image_data()

            #print("------------------------")

            #   Recursive call for further chopping if necessary
            self.get_patches_from_image(firstImage,self.HALO_SIZE)
            self.get_patches_from_image(secondImage,self.HALO_SIZE)
            self.get_patches_from_image(thirdImage,self.HALO_SIZE)
            self.get_patches_from_image(fourthImage,self.HALO_SIZE)

            # cv2.namedWindow('primera',cv2.WINDOW_NORMAL)
            # cv2.namedWindow('segunda',cv2.WINDOW_NORMAL)
            # cv2.namedWindow('tercera', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('cuarta',cv2.WINDOW_NORMAL)
            # cv2.imshow('primera', firstPatch)
            # cv2.imshow('segunda', secondPatch)
            # cv2.imshow('tercera', thirdPatch)
            # cv2.imshow('cuarta', fourthPatch)
            # cv2.resizeWindow('primera', self.WINDOW_SIZE,self.WINDOW_SIZE)
            # cv2.resizeWindow('segunda', self.WINDOW_SIZE,self.WINDOW_SIZE)
            # cv2.resizeWindow('tercera', self.WINDOW_SIZE,self.WINDOW_SIZE)
            # cv2.resizeWindow('cuarta', self.WINDOW_SIZE,self.WINDOW_SIZE)
            #dcv2.waitKey(0)


    def add_image_to_dataset(self):
        imageForPatching = cv2.imread(self.imagePathToTest)
        originCoordinates = [0,0]
        destinationCoordinates = [imageForPatching.shape[1],imageForPatching.shape[0]]
        print("Coordinates of initial image:")
        print(originCoordinates)
        print(destinationCoordinates)
        imageToAdd = Image(7,imageForPatching,originCoordinates,destinationCoordinates)

        #print(imageToAdd.shape)
        cv2.namedWindow('original',cv2.WINDOW_NORMAL)
        cv2.imshow('original', imageToAdd.imageContent)
        cv2.resizeWindow('original', self.WINDOW_SIZE*2,self.WINDOW_SIZE*2)
        print("Generating patches from image...")
        self.get_patches_from_image(imageToAdd,self.HALO_SIZE)
        #self.listOfPatches.append(patches)
        print("Image selected: ")
        print(self.get_patch(0).indexOfImage)
        print("Number of patches generated: ")
        print(self.len_patches())
        print("Recursion level: ")
        recursionLevel = len(self.get_patch(0).coordinatesInOriginalImage)
        print(recursionLevel)
        #vis = np.concatenate((self.get_patch(0).patchImage, self.get_patch(1).patchImage), axis=1)




    def reconstruct_image_from_patches(self):
        return




    # def __init__(self, input_path, groundtruth_path=None):
    #     self.input_path = input_path
    #     self.groundtruth_path = groundtruth_path
    #     self.files = []
    #     print("starting")
    #     for filename in os.listdir(input_path):
    #         if os.path.isfile(os.path.join(input_path, filename)) and \
    #            ((groundtruth_path == None) or \
    #            os.path.isfile(os.path.join(groundtruth_path, filename))):
    #                print('Adding file: {}'.format(filename))
    #                self.files.append(filename)
    #     print('Total files in dataset: {}'.format(len(self.files)))


if __name__ == '__main__':
    datasetObject = Dataset()
    datasetObject.add_image_to_dataset()




####
