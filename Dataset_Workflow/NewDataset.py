import cv2
import numpy as np
import torch
import os
from Patch import Patch
import glob


class Dataset(torch.utils.data.Dataset):

    #   Constant declaration
    maxDimensionsForGPU = 512
    HALO_SIZE = 20
    WINDOW_SIZE = 300

    #   Lists for images and patches
    listOfDirectoriesForGT = ["FrameRegion", "ImageRegion", "GraphicRegion",
    "TextRegion/caption", "TextRegion/heading", "TextRegion/paragraph"]

    listOfPatches = []
    currentGTPatch = None
    imageCopyForRects = None
    currentImageBeingComputed = None


    def __img2tensor(self, img):
        img = img.astype(np.float32) / 255.0

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        channels =  img.shape[2]
        if channels == 1:
            img = img[:, :, [0]]
        elif channels == 3:
            img = img[:, :, [2, 1, 0]]
        elif channels == 4:
            img = img[:, :, [2, 1, 0, 3]]

        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        return img


    def __tensor2img(self, tensor, min_max=(0., 1.)):
        tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
        tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

        ndim = tensor.dim()
        if ndim == 4:
            n_img = len(tensor)
            img_np = make_grid(tensor, nrow=int(np.math.sqrt(n_img)), normalize=False).numpy()
            if tensor.size()[1] == 3:
                img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            elif tensor.size()[1] == 4:
                img_np = np.transpose(img_np[[2, 1, 0, 3], :, :], (1, 2, 0))  # HWC, BGR

        elif ndim == 3:
            img_np = tensor.detach().numpy()
            # img_np = tensor.numpy()
            if tensor.size()[0] == 3:
                img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
            elif tensor.size()[0] == 4:
                img_np = np.transpose(img_np[[2, 1, 0, 3], :, :], (1, 2, 0))  # HWC, BGR

        elif ndim == 2:
            img_np = tensor.numpy()

        else:
            raise TypeError(
                'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(ndim))

        img_np = (img_np * 255.0).round()
        return img_np.astype(np.uint8)


    def __len__(self):
        return len(self.listOfPatches)

    def calculate_dimensions_current_image(self):
        dimensionsOfImage = self.currentImageBeingComputed.shape
        return [dimensionsOfImage[0],dimensionsOfImage[1]]


    def add_position_channels_to_patch(self, patch, patchAsImage):
        #   We have to convert twice to obtain a value that we can parse
        #   and also use the div method in PyTorch (only accepts floats)
        width = int(patch.width)
        height = int(patch.height)
        width = float(width)
        height = float(height)

        #   Creation of the tensor for the x coordinates of the patch with
        #   respect to the whole image
        tensorRow = torch.arange(patch.topLeftCoordinates[0],patch.topLeftCoordinates[0] + int(width))
        tensorRowDivided = torch.div(tensorRow,width)
        tensorRowFullSize = tensorRowDivided.repeat([int(height),1])
        #   Saving tensor x to disk as image for testing
        xTensorAsImage = self.__tensor2img(tensorRowFullSize)
        cv2.imwrite("xTensorAsImage4.png", xTensorAsImage)

        #   Creation of the tensor for the y coordinates of the patch with
        #   respect to the whole image
        tensorColumn = torch.arange(patch.topLeftCoordinates[1],patch.topLeftCoordinates[1] + int(height))
        tensorColumnUnsqueezed = torch.unsqueeze(tensorColumn,1)
        tensorColumnDivided = torch.div(tensorColumnUnsqueezed,height)
        tensorColumnFullSize = tensorColumnDivided.repeat([1,int(width)])
        #   Saving y tensor to disk as image for testing
        yTensorAsImage = self.__tensor2img(tensorColumnFullSize)
        cv2.imwrite("yTensorAsImage4.png", yTensorAsImage)

        #   Merging all the channels to obtain a single tensor of the shape
        #   (r,g,b,xTensor,yTensor)
        patchImageAsTensor = self.__img2tensor(patchAsImage)
        dimensionChannelsTensor = torch.stack((tensorColumnFullSize,tensorRowFullSize))
        patchWithPositionChannels = torch.cat((patchImageAsTensor,dimensionChannelsTensor))

        print("Tensor created with the input patch:")
        print(patchWithPositionChannels.size())
        return patchWithPositionChannels

    def generate_patch_from_image(self, patchToBeGenerated):
        imageRoute = patchToBeGenerated.patchImageName
        imageToBeChopped = cv2.imread(str(self.input_path) + "/" + imageRoute)
        #   Obtain the coordinates to crop
        coordinatesForChopping = patchToBeGenerated.coordinatesInOriginalImage
        #   The actual chopping done to the input image by slicing [y1:y2,x1:x2]
        inputPatchGenerated = imageToBeChopped[int(coordinatesForChopping[0][1]):int(coordinatesForChopping[1][1]), int(coordinatesForChopping[0][0]):int(coordinatesForChopping[1][0])]
        finalInputPatch = self.add_position_channels_to_patch(patchToBeGenerated,inputPatchGenerated)
        return finalInputPatch


    def generate_ground_truth_patch_from_image(self, indexOfImage):
        #   First we find the patch data since it will be the same coordinates
        #   for the ground truth
        patchForGTData = self.listOfPatches[indexOfImage]
        coordinatesToApply = patchForGTData.coordinatesInOriginalImage
        nameToSearchFor = patchForGTData.patchImageName
        gtPatchToCreate = None
        counterGTFiles = 0
        #   Now we iterate through the ground truth folders and obtain the tensors
        for root, dirs, files in os.walk(self.groundtruth_path):
             for imageGT in files:
                 if imageGT.endswith('.png') and imageGT == nameToSearchFor:
                     gtColorImageForIteration = cv2.imread(self.groundtruth_path + "/" + self.listOfDirectoriesForGT[counterGTFiles]+ "/" + imageGT)
                     gtImageForIteration = cv2.cvtColor(gtColorImageForIteration, cv2.COLOR_BGR2GRAY)
                     gtPatchGenerated = gtImageForIteration[int(coordinatesToApply[0][1]):int(coordinatesToApply[1][1]), int(coordinatesToApply[0][0]):int(coordinatesToApply[1][0])]
                     if gtPatchToCreate is None:
                         gtPatchToCreate = self.__img2tensor(gtPatchGenerated)
                     else:
                         tempTensor = torch.cat((gtPatchToCreate,self.__img2tensor(gtPatchGenerated)),0)
                         gtPatchToCreate = tempTensor
                     counterGTFiles += 1
        gtPatchAsImage = self.__tensor2img(gtPatchToCreate)
        print("Tensor created with ground truth patch:")
        print(gtPatchToCreate.size())
        return gtPatchToCreate

    #   Generates one sample of data (patch)∫
    def __getitem__(self, indexOfItem):
        print("Retrieving patch number " + str(indexOfItem))
        patchToGenerate = self.listOfPatches[indexOfItem]
        patchObtained = self.generate_patch_from_image(patchToGenerate)
        gtPatchObtained = self.generate_ground_truth_patch_from_image(indexOfItem)
        #   Used for testing, no GT is provided
        if self.groundtruth_path == None:
            return [self.__img2tensor(patchObtained), torch.zeros(1)]
        #   Used in training and validation
        return [patchObtained, gtPatchObtained]


    #   Calculate coordinates for the patching of the image until it fits in GPU
    def calculate_patches_from_image(self,patchSoFar):
        [width,height] = patchSoFar.calculate_dimensions_of_patch()
        filenameForPatches = patchSoFar.patchImageName
        #   If the size of the current image is already small enough for our GPU
        #   we can create the patch from it
        if width * height < pow(self.maxDimensionsForGPU, 2):
            patchSoFar.set_absolute_image_shape(width,height)
            patchSoFar.calculate_extra_coordinates()
            #   Add new patch to the general list
            self.listOfPatches.append(patchSoFar)
        else:
            #   Computation of the coordinates of the patches with respect to the image
            coordinatesFirstPatch = [[patchSoFar.topLeftCoordinates[0],patchSoFar.topLeftCoordinates[1]],
            [patchSoFar.topLeftCoordinates[0] + width/2 +self.HALO_SIZE,patchSoFar.topLeftCoordinates[1] + height/2 + self.HALO_SIZE]]
            coordinatesSecondPatch = [[patchSoFar.topLeftCoordinates[0]+width/2-self.HALO_SIZE,patchSoFar.topLeftCoordinates[1]],
            [patchSoFar.bottomRightCoordinates[0],patchSoFar.topLeftCoordinates[1] + height/2 + self.HALO_SIZE]]
            coordinatesThirdPatch = [[patchSoFar.topLeftCoordinates[0],patchSoFar.topLeftCoordinates[1] + height/2 -self.HALO_SIZE],
            [patchSoFar.topLeftCoordinates[0]+width/2+self.HALO_SIZE,patchSoFar.bottomRightCoordinates[1]]]
            coordinatesFourthPatch = [[patchSoFar.topLeftCoordinates[0]+width/2-self.HALO_SIZE,patchSoFar.topLeftCoordinates[1] + height/2 -self.HALO_SIZE],
            [patchSoFar.bottomRightCoordinates[0],patchSoFar.bottomRightCoordinates[1]]]

            #   Creation of patches for possible further cropping
            firstPatch = Patch(filenameForPatches,coordinatesFirstPatch, self.HALO_SIZE)
            secondPatch = Patch(filenameForPatches,coordinatesSecondPatch, self.HALO_SIZE)
            thirdPatch = Patch(filenameForPatches,coordinatesThirdPatch, self.HALO_SIZE)
            fourthPatch = Patch(filenameForPatches,coordinatesFourthPatch, self.HALO_SIZE)

            #   Recursive call for further chopping if necessary
            self.calculate_patches_from_image(firstPatch)
            self.calculate_patches_from_image(secondPatch)
            self.calculate_patches_from_image(thirdPatch)
            self.calculate_patches_from_image(fourthPatch)


    #   Traverse input directory and calculate chopping coordinates for all
    #   of the images in it
    def load_input_names(self, inputPath):
        print("Loading input images")
        counterInputFiles = 0
        for fileInDataset in os.listdir(inputPath):
            if fileInDataset.endswith(".png"):
                self.currentImageBeingComputed = cv2.imread(self.input_path + "/" + fileInDataset)
                [width,height] = self.calculate_dimensions_current_image()
                initialPatchForCalculation = Patch(fileInDataset,[[0,0],[width,height]],self.HALO_SIZE)
                self.calculate_patches_from_image(initialPatchForCalculation)
                counterInputFiles += 1
        print("The chopping of " + str(counterInputFiles) + " input images has been computed")
        print("-----------------------")


    def create_dataset(self, input_path, groundtruth_path=None):
        self.input_path = input_path
        self.groundtruth_path = groundtruth_path
        self.load_input_names(self.input_path)
        print("The dataset has been fully loaded")


if __name__ == '__main__':
    datasetObject = Dataset()
    datasetObject.create_dataset("Dataset/Training","Dataset/GroundTruths")
    print("Number of patches pre-computed: " + str(datasetObject.__len__()))
    print("-----------------------")
    datasetObject.__getitem__(4)
    print("Obtained input and ground truth patches")






#
