# Mirror to the Blob.h and BlobResult.h

import roslib; roslib.load_manifest('cv_blobs')
import rospy
import cv2
import os.path
import math
import numpy as np
import pyblob
from reefbot_msgs.msg import ImageRegion
from reefbot_msgs.msg import SpeciesIDRequest
from sensor_msgs.msg import RegionOfInterest
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Blob:
  FOUR_CONNECTED = 0
  EIGHT_CONNECTED = 1
  
  def __init__(self, id_=-1, connectionType=FOUR_CONNECTED):
    self.id = id_
    self.connectionType = connectionType
    self.points = set()
    self.minX = float("inf")
    self.minY = float("inf")
    self.maxX = float("-inf")
    self.maxY = float("-inf")
    self.bridge = CvBridge()

  def AddPoint(self, point):
    '''Adds a point to the blob. Point must be a tuple of (x,y).'''
    self.points.add(point)
    if point[0] < self.minX:
      self.minX = point[0]
    if point[0] > self.maxX:
      self.maxX = point[0]
    if point[1] < self.minY:
      self.minY = point[1]
    if point[1] > self.maxY:
      self.maxY = point[1]

  def AddPoints(self, pointCollection):
    '''Adds a set of points to the blob.'''
    for point in pointCollection:
      self.AddPoint(tuple(point))

  def AddBox(self, minX, minY, maxX, maxY):
    '''Adds a blob from (minX,minY) to (maxX,maxY) inclusive.'''
    coords = np.transpose(np.meshgrid(np.arange(minX, maxX+1),
                                      np.arange(minY, maxY+1)))
    self.AddPoints(np.reshape(coords,
                              (coords.shape[0]*coords.shape[1],2)))

  def Area(self):
    return len(self.points)

  def Contains(self, point):
    '''Returns true if the blob contains a given point.'''
    return point in self.points

  def GetBoundingBox(self):
    '''Returns the bounding box of the blob as (minX,minY,maxX,maxY).'''
    return (self.minX, self.minY, self.maxX, self.maxY)

  def __iter__(self):
    '''Returns an iterator through the points in the blob.'''
    return self.points.__iter__()

  def __eq__(self, other):
    '''Returns true if the two Blobs describe the same points.'''
    return self.points == other.points

  def __ne__(self, other):
    return not self == other

  def __str__(self):
    return str(self.points)

  def ToImageRegion(self):
    '''Returns a reefbot_msgs::ImageRegion describing the blob.'''
    # First build the bounding box
    bbox = RegionOfInterest(self.minX, self.minY, self.maxY-self.minY+1,
                            self.maxX-self.minX+1, False)

    # Now build the mask image defining which pixels are used
    mask = np.zeros((bbox.height, bbox.width), dtype=np.uint8)
    for x,y in self.points:
      mask[y - self.minY, x - self.minX] = 1

    # Create the image region object
    imageRegion = ImageRegion(bounding_box=bbox,
                              mask=self.bridge.cv_to_imgmsg(mask, "mono8"))

    return imageRegion

  def ToBinaryImageMask(self, shape):
    '''Returns a numpy matrix that specifies the blobs as 1 for pixel in the blobs and a 0 otherwise.'''
    # Create the mask
    mask = np.zeros(shape, np.uint8)
    for x,y in self.points:
      mask[y, x] = 1

    return mask

  def AsciiSerialize(self):
    '''Serializes the blob into a list of locations on a line.

    It is in the format "x,y;x,y;x,y"
    '''
    if len(self.points) > 0:
      return ';'.join(['%i,%i' % x for x in self.points]) + ';\n'
    return ''

  @staticmethod
  def CreateFromString(string):
    '''Loads a description of a blob from a string output by AsciiSerialize.'''
    blob = Blob()
    strList = string.strip().split(';')
    if len(strList) == 1 and strList[0] == '':
      return blob
    for pointStr in strList:
      if pointStr == '':
        continue
      part = pointStr.partition(',')
      if part[1] == '':
        raise ValueError('No coordinate found in: ' + pointStr)

      blob.AddPoint((int(part[0]), int(part[2])))

    return blob
  

class BlobResult:
  '''A group of blobs.
  '''
  
  def __init__(self, connectionType=Blob.FOUR_CONNECTED):
    self.blobs = []
    self.connectionType = connectionType
    self.shape = None

  def FindBlobs(self, image, minimumSize=10):
    '''Finds the nonzero blobs in a binary image.

    image - numpy binary 2D array
    minimumSize - minimum size of a blob in diameter
    connectionType - Type of connectivity for the blobs
    '''
    blobList = pyblob.findblobs(np.nonzero(image), image.shape[0],
                                image.shape[1], self.connectionType)
    self.blobs = []
    for blob in blobList:
      if len(blob) >= minimumSize:
        newBlob = Blob()
        newBlob.AddPoints(blob)
        self.blobs.append(newBlob)
    self.shape = image.shape

  def nBlobs(self):
    '''The number of blobs found.'''
    return len(self.blobs)

  def GetBlob(self, idx):
    '''Get the nth blob from the set.'''
    return self.blobs[idx]

  def AppendBlob(self, blob):
    self.blobs.append(blob)

  def ToBinaryImageMask(self):
    '''Returns a numpy matrix that specifies the blobs as 1 for pixel in the blobs and a 0 otherwise.'''

    # Figure out the size of the mask
    imageShape = self.shape
    if imageShape is None:
      maxX = 0
      maxY = 0
      for blob in self.blobs:
        if blob.maxX > maxX:
          maxX = blob.maxX
        if blob.maxY > maxY:
          maxY = blob.maxY
      imageShape = (maxY+1, maxX+1)

    # Create the mask
    mask = np.zeros(imageShape, np.uint8)
    for blob in self.blobs:
      for x,y in blob.points:
        mask[y, x] = 1

    return mask
    

  def __iter__(self):
    return self.blobs.__iter__()

class BlobSerializer:
  '''A class to read and write blobs to a file.'''

  def __init__(self):
    pass

  def Serialize(self, file, blobs, imgName):
    '''Write a description of the blobs to a file.

    The first line will be the image name and each following line will
    describe a blob.

    Inputs:
    file - File handle to write to
    blobs - BlobResult object to write to disk
    imgName - Filename of the image
    '''
    file.write(imgName)
    file.write('\n')
    file.write(''.join(
      [blobs.GetBlob(x).AsciiSerialize() for x in range(blobs.nBlobs())]))

  def Deserialize(self, file, blobDir=None):
    '''Reads a blob file that was written by Serialize.

    Inputs:
    file - file handle to read from

    Outputs:
    blobs - A BlobResult object describing the blobs in the image
    imgName - Filename of the image that the blobs describe
    '''
    blobs = BlobResult()
    imgName = file.readline().strip()
    if imgName[0] <> '/' and blobDir is not None:
      imgName = os.path.join(blobDir, imgName)
    if not os.path.exists(imgName):
      raise IOError("Image file does not exist: " + imgName)
    for line in file:
      blobs.blobs.append(Blob.CreateFromString(line.strip()))

    return blobs, imgName

  def GetImageName(self, fileStream, blobDir=None):
    '''Reads the image name from a blob file and returns it.'''
    imgName = file.readline().strip()
    if imgName[0] <> '/' and blobDir is not None:
      imgName = os.path.join(blobDir, imgName)
    if not os.path.exists(imgName):
      raise IOError("Image file does not exist: " + imgName)

    return imgName

def OpenBlobAsSpeciesIDRequest(filename):
  '''Function that opens up a blob file and converts it to a SpeciesIDRequest.'''
  serializer = BlobSerializer()

  # Read the blob file
  f = open(filename)
  try:
    blobs, imgFilename = serializer.Deserialize(f, os.path.dirname(filename))
  finally:
    f.close()

  # Now open up the image file associated with the blob
  cvImg = cv2.imread(imgFilename)
  if cvImg.shape[1] == 0 or cvImg.shape[0] == 0:
    raise IOError("Could not open image: " + imgFilename)
  if cvImg.shape[2] <> 3:
    raise IOError("Image %s was not a 3 color image. It has %d channels" %
                  (imgFilename, cvImg.shape[2]))
  if cvImg.dtype <> np.uint8:
    raise IOError("Image %s was not a 8-bit image. It has %d bit depth" %
                  (imgFilename, cvImg.dtype))
  
  bridge = CvBridge()
  # For some reason, opencv puts images in BGR format, so label that
  request = SpeciesIDRequest(image=bridge.cv_to_imgmsg(cvImg, "bgr8"))

  for blob in blobs:
    request.regions.append(blob.ToImageRegion())

  return request
