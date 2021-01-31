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

    maxDimensionsForGPU = 512
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
            newPatch = Patch(imageToPatch.imageContent,imageToPatch.imageIndex,imageToPatch.coordinatesList,1)
            #append to list of patches
            self.listOfPatches.append(newPatch)
            cv2.imshow('ok', newPatch.patchImage)
            print(newPatch.coordinatesInOriginalImage)
            print(sum(newPatch.coordinatesInOriginalImage))
            cv2.waitKey(0)
            #print("decent for CPU")
            # cv2.namedWindow('ok', cv2.WINDOW_NORMAL)
            # cv2.imshow('ok', imageToPatch)
            # cv2.resizeWindow('ok',self.WINDOW_SIZE, self.WINDOW_SIZE)
        else:
            #print("we have to chop")
            firstPatchArea = actualImage[0:int(0+height/2+lengthOfHalo), 0:int(0+width/2+lengthOfHalo)]
            secondPatchArea = actualImage[0:int(0+height/2+lengthOfHalo), int(0+width/2-lengthOfHalo):width]
            thirdPatchArea = actualImage[int(height/2-lengthOfHalo):height, 0:int(width/2+lengthOfHalo)]
            fourthPatchArea = actualImage[int(height/2-lengthOfHalo):height, int(width/2-lengthOfHalo):width]
            firstImage = Image(indexForPatches,firstPatchArea)
            firstImage.add_coordinate(0)
            secondImage = Image(indexForPatches,secondPatchArea)
            secondImage.add_coordinate(1)
            thirdImage = Image(indexForPatches,thirdPatchArea)
            thirdImage.add_coordinate(2)
            fourthImage = Image(indexForPatches,fourthPatchArea)
            fourthImage.add_coordinate(3)

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
        imageToAdd = Image(7,cv2.imread(self.imagePathToTest))
        #print(imageToAdd.shape)
        cv2.namedWindow('original',cv2.WINDOW_NORMAL)
        cv2.imshow('original', imageToAdd.imageContent)
        cv2.resizeWindow('original', self.WINDOW_SIZE*2,self.WINDOW_SIZE*2)
        patches = self.get_patches_from_image(imageToAdd,self.HALO_SIZE)
        self.listOfPatches.append(patches)
        print(self.len_patches())
        print(self.get_patch(0).indexOfImage)
        cv2.imshow('otra',self.get_patch(0).patchImage)



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
