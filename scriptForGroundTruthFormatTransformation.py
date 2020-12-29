import cv2
import numpy as np
import xml.etree.ElementTree as ET
import glob



#parameters for the segmentation tags
#TODO: refactor to dictionary with specific sizes according to region
regionsThatNeedBiggerSize = ["GraphicRegion"]
minimumAreaForSpecificRegions = 1000
regionsWithRestrictedTypes = {"TextRegion":["paragraph", "heading","floating"]}
segmentationTagsIncluded = ['ImageRegion','TextRegion','GraphicRegion']


#variables initialization
tree = None
imTest = None
blackImage = None
root = None


# Variables for concrete test with only one document
# tree = ET.parse('gtForTest5.xml')
# imTest = cv2.imread('imForTest5.tif')
# cv2.namedWindow('imageOG',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('imageOG', 600,600)
# cv2.imshow('imageOG',imTest)
# blackImage = np.zeros((imTest.shape[0], imTest.shape[1], 1), dtype = "uint8")
# root = tree.getroot()



def processImage(segmentationTagForIteration):
    blackImage = np.zeros((imTest.shape[0], imTest.shape[1], 1), dtype = "uint8")
    for child in root:
        for subchild in child:
            tagFormatted = subchild.tag.split("}")[-1]
            if tagFormatted == segmentationTagForIteration:
                for regionChild in subchild:
                    pointsForPolygon = []
                    regionChildFormatted = regionChild.tag.split("}")[-1]
                    if regionChildFormatted == "Coords":
                        for point in regionChild:
                            pointForPolygon = [int(point.attrib['x']),int(point.attrib['y'])]
                            pointsForPolygon.append(pointForPolygon)

                pts = np.array([pointsForPolygon], np.int32)
                pts = pts.reshape((-1,1,2))
                
                #if it is a graphic element we have to calculate the area and
                #only consider it if it exceeds a minimum
                if tagFormatted in regionsThatNeedBiggerSize:
                    area = cv2.contourArea(pts)
                    if area <= minimumAreaForSpecificRegions:
                        print("area insuficient for region consideration")
                        continue

                #fill all polygons obtained
                cv2.fillPoly(blackImage, [pts], color=(255,255,255))

                #only delineate polygons
                #imageWithPolygons = cv2.polylines(blackImage,[pts],True,(255,255,255),10)

    newImageName = str(currentImPath).split(".")[0] + "_" + segmentationTagForIteration + ".png"
    cv2.imwrite("GroundTruthsTransformed/" +newImageName, blackImage)

def processImageWithType(segmentationTagForIteration, typeRequired):
    blackImage = np.zeros((imTest.shape[0], imTest.shape[1], 1), dtype = "uint8")
    for child in root:
        for subchild in child:
            tagFormatted = subchild.tag.split("}")[-1]
            if "type" in subchild.attrib:
             if tagFormatted == segmentationTagForIteration and subchild.attrib["type"] == typeRequired:
                for regionChild in subchild:
                    pointsForPolygon = []
                    regionChildFormatted = regionChild.tag.split("}")[-1]
                    if regionChildFormatted == "Coords":
                        for point in regionChild:
                            pointForPolygon = [int(point.attrib['x']),int(point.attrib['y'])]
                            pointsForPolygon.append(pointForPolygon)

                if pointsForPolygon == []:
                    continue
                pts = np.array([pointsForPolygon], np.int32)
                pts = pts.reshape((-1,1,2))

                #if it is a graphic element we have to calculate the area and
                #only consider it if it exceeds a minimum
                if tagFormatted in regionsThatNeedBiggerSize:
                    area = cv2.contourArea(pts)
                    if area <= minimumAreaForSpecificRegions:
                        print("area insuficient for region consideration")
                        continue

                #fill all polygons obtained
                cv2.fillPoly(blackImage, [pts], color=(255,255,255))
                #only delineate polygons
                #imageWithPolygons = cv2.polylines(blackImage,[pts],True,(255,255,255),10)

    newImageName = str(currentImPath).split(".")[0] + "_" + segmentationTagForIteration + "_" + typeRequired + ".png"
    cv2.imwrite("GroundTruthsTransformed/" + newImageName, blackImage)

def groundTruthTransformation():
    for tag in segmentationTagsIncluded:
        if tag not in regionsWithRestrictedTypes:
            processImage(tag)
        else:
            for typeForRegion in regionsWithRestrictedTypes[tag]:
                processImageWithType(tag,typeForRegion)


def transformGroundTruthFormat():
    global tree, imTest, blackImage, root, currentImPath
    imagePaths = glob.glob("*.png")
    groundTruthPaths = glob.glob("*.xml")

    for im in imagePaths:
         print("transforming image " + im)
         currentImPath = im
         pathGT = im.split(".")[0] + ".xml"
         tree = ET.parse(pathGT)
         root = tree.getroot()
         imTest = cv2.imread(im)
         blackImage = np.zeros((imTest.shape[0], imTest.shape[1], 1), dtype = "uint8")
         groundTruthTransformation()
         print("image transformed")


transformGroundTruthFormat()

# print("Image processed")
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', 600,600)
# cv2.imshow('image',blackImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
