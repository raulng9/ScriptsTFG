import cv2
import numpy as np
import xml.etree.ElementTree as ET



tree = ET.parse('gtForTest.xml')



imTest = cv2.imread('imForTest.tif')


blackImage = np.zeros((imTest.shape[0], imTest.shape[1], 1), dtype = "uint8")

root = tree.getroot()
segmentationTagsIncluded = ['TextRegion']

for child in root:
    for subchild in child:
        tagFormatted = subchild.tag.split("}")[-1]
        if tagFormatted in segmentationTagsIncluded and subchild.attrib["type"] == "paragraph":
            for regionChild in subchild:
                pointsForPolygon = []
                regionChildFormatted = regionChild.tag.split("}")[-1]
                if regionChildFormatted == "Coords":
                    for point in regionChild:
                        pointForPolygon = [int(point.attrib['x']),int(point.attrib['y'])]
                        pointsForPolygon.append(pointForPolygon)
            print(pointsForPolygon)

            pts = np.array([pointsForPolygon], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.fillPoly(blackImage, [pts], color=(255,255,255))
            #only delineate polygons
            #imageWithPolygons = cv2.polylines(blackImage,[pts],True,(255,255,255),10)


cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow('image',blackImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
