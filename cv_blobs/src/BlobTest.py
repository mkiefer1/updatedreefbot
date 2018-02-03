#!/usr/bin/python
import roslib; roslib.load_manifest('cv_blobs')
import rospy
import unittest
from cv_blobs import Blob
import StringIO
import os
from cv_bridge import CvBridge
import numpy as np

class BlobTest(unittest.TestCase):
  def setUp(self):
    self.blob = Blob.Blob(id_=3, connectionType=Blob.Blob.EIGHT_CONNECTED)
  
  def test_EmptyInitially(self):
    self.assertEqual(self.blob.id, 3)
    self.assertEqual(self.blob.connectionType,
                     Blob.Blob.EIGHT_CONNECTED)
    self.assertEqual(self.blob.Area(), 0)
    self.assertFalse(self.blob.Contains((5,3)))

  def test_OnePointAdded(self):
    self.blob.AddPoint((5,3))
    self.assertEqual(self.blob.Area(), 1)
    self.assertEqual(self.blob.minX, 5)
    self.assertEqual(self.blob.maxX, 5)
    self.assertEqual(self.blob.minY, 3)
    self.assertEqual(self.blob.maxY, 3)
    self.assertTrue(self.blob.Contains((5,3)))

  def test_OnePointAddedTwice(self):
    self.blob.AddPoint((5,3))
    self.blob.AddPoint((5,3))
    self.assertEqual(self.blob.Area(), 1)
    self.assertEqual(self.blob.minX, 5)
    self.assertEqual(self.blob.maxX, 5)
    self.assertEqual(self.blob.minY, 3)
    self.assertEqual(self.blob.maxY, 3)

  def test_TriangleAdded(self):
    self.blob.AddPoint((5,3))
    self.blob.AddPoint((1,3))
    self.blob.AddPoint((2,8))
    self.assertEqual(self.blob.Area(), 3)
    self.assertEqual(self.blob.minX, 1)
    self.assertEqual(self.blob.maxX, 5)
    self.assertEqual(self.blob.minY, 3)
    self.assertEqual(self.blob.maxY, 8)
    self.assertTrue(self.blob.Contains((5,3)))
    self.assertTrue(self.blob.Contains((1,3)))
    self.assertTrue(self.blob.Contains((2,8)))

  def test_Equals(self):
    otherBlob = Blob.Blob()
    self.assertEqual(self.blob, otherBlob)
    
    self.blob.AddPoint((5,3))
    self.assertNotEqual(self.blob, otherBlob)

    otherBlob.AddPoint((5,3))
    self.assertEqual(self.blob, otherBlob)

class SerializeTest(unittest.TestCase):
  def setUp(self):
    self.blob = Blob.Blob()
    self.blob.AddPoint((5,3))
    self.blob.AddPoint((3,2))

  def test_RoundTripSerialization(self):
    blobString = self.blob.AsciiSerialize()

    newBlob = Blob.Blob.CreateFromString(blobString)
    self.assertEqual(newBlob, self.blob)

  def test_EmptyRoundTripSerialization(self):
    blobString = Blob.Blob().AsciiSerialize()

    newBlob = Blob.Blob.CreateFromString(blobString)
    self.assertEqual(newBlob, Blob.Blob())

  def test_SerializationString(self):
    self.assertEqual(self.blob.AsciiSerialize(),
                     '3,2;5,3;\n')

  def test_EmptySerializationString(self):
    self.assertEqual(Blob.Blob().AsciiSerialize(), '')

  def test_CreateFromNormalString(self):
    newBlob = Blob.Blob.CreateFromString('5,3;3,2;\n')

    self.assertTrue(newBlob.Contains((5,3)))
    self.assertTrue((5,3) in newBlob)
    self.assertTrue(newBlob.Contains((3,2)))
    self.assertEqual(newBlob.Area(), 2)
    self.assertEqual(newBlob.minX, 3)
    self.assertEqual(newBlob.maxX, 5)
    self.assertEqual(newBlob.minY, 2)
    self.assertEqual(newBlob.maxY, 3)

  def test_DifferentPointOrder(self):
    newBlob = Blob.Blob.CreateFromString('3,2;5,3;\n')

    self.assertTrue(newBlob.Contains((5,3)))
    self.assertTrue((5,3) in newBlob)
    self.assertTrue(newBlob.Contains((3,2)))
    self.assertEqual(newBlob.Area(), 2)
    self.assertEqual(newBlob.minX, 3)
    self.assertEqual(newBlob.maxX, 5)
    self.assertEqual(newBlob.minY, 2)
    self.assertEqual(newBlob.maxY, 3)

  def test_CreateFromEmptyString(self):
    newBlob = Blob.Blob.CreateFromString('')

    self.assertEqual(newBlob, Blob.Blob())

  def test_CreateFromStringWithMissingSemiColon(self):
    newBlob = Blob.Blob.CreateFromString('5,3;3,2\n')

    self.assertTrue(newBlob.Contains((5,3)))
    self.assertTrue(newBlob.Contains((3,2)))
    self.assertEqual(newBlob.Area(), 2)
    self.assertEqual(newBlob.minX, 3)
    self.assertEqual(newBlob.maxX, 5)
    self.assertEqual(newBlob.minY, 2)
    self.assertEqual(newBlob.maxY, 3)

  def test_CreateFromStringWithMissingNewline(self):
    newBlob = Blob.Blob.CreateFromString('5,3;3,2;')

    self.assertTrue(newBlob.Contains((5,3)))
    self.assertTrue(newBlob.Contains((3,2)))
    self.assertEqual(newBlob.Area(), 2)
    self.assertEqual(newBlob.minX, 3)
    self.assertEqual(newBlob.maxX, 5)
    self.assertEqual(newBlob.minY, 2)
    self.assertEqual(newBlob.maxY, 3)

  def test_CreateFromStringWithExtraWhitespace(self):
    newBlob = Blob.Blob.CreateFromString('  5,3;3,2;  \n')

    self.assertTrue(newBlob.Contains((5,3)))
    self.assertTrue(newBlob.Contains((3,2)))
    self.assertEqual(newBlob.Area(), 2)
    self.assertEqual(newBlob.minX, 3)
    self.assertEqual(newBlob.maxX, 5)
    self.assertEqual(newBlob.minY, 2)
    self.assertEqual(newBlob.maxY, 3)

  def test_CreateFromBadStrings(self):
    self.assertRaises(ValueError, Blob.Blob.CreateFromString, ('5,3;a,b;'))
    self.assertRaises(ValueError, Blob.Blob.CreateFromString, ('5,3;3,2;4,'))
    self.assertRaises(ValueError, Blob.Blob.CreateFromString, ('5,3;3,2;,'))
    self.assertRaises(ValueError, Blob.Blob.CreateFromString, ('monkey'))

class ToImageRegionTest(unittest.TestCase):
  def setUp(self):
    self.blob = Blob.Blob()

  def test_FullBoundingBox(self):
    self.blob.AddPoint((2,4))
    self.blob.AddPoint((2,5))
    self.blob.AddPoint((3,4))
    self.blob.AddPoint((3,5))
    self.blob.AddPoint((4,4))
    self.blob.AddPoint((4,5))

    imageRegion = self.blob.ToImageRegion()

    self.assertEqual(imageRegion.bounding_box.x_offset, 2)
    self.assertEqual(imageRegion.bounding_box.y_offset, 4)
    self.assertEqual(imageRegion.bounding_box.width, 3)
    self.assertEqual(imageRegion.bounding_box.height, 2)

  def test_PartialBox(self):
    self.blob.AddPoint((2,4))
    self.blob.AddPoint((3,5))
    self.blob.AddPoint((4,5))

    imageRegion = self.blob.ToImageRegion()

    self.assertEqual(imageRegion.bounding_box.x_offset, 2)
    self.assertEqual(imageRegion.bounding_box.y_offset, 4)
    self.assertEqual(imageRegion.bounding_box.width, 3)
    self.assertEqual(imageRegion.bounding_box.height, 2)

    self.assertEqual(imageRegion.mask.height, 2)
    self.assertEqual(imageRegion.mask.width, 3)
    self.assertEqual(imageRegion.mask.data[0], chr(1))
    self.assertEqual(imageRegion.mask.data[1], chr(0))
    self.assertEqual(imageRegion.mask.data[2], chr(0))
    self.assertEqual(imageRegion.mask.data[3], chr(0))
    self.assertEqual(imageRegion.mask.data[4], chr(1))
    self.assertEqual(imageRegion.mask.data[5], chr(1))

  def test_WithBridge(self):
    self.blob.AddPoint((2,4))
    self.blob.AddPoint((3,5))
    self.blob.AddPoint((4,5))

    imageRegion = self.blob.ToImageRegion()

    bridge = CvBridge()
    cvImg = bridge.imgmsg_to_cv(imageRegion.mask, "mono8")

    self.assertEqual(cvImg.rows, 2)
    self.assertEqual(cvImg.cols, 3)
    self.assertEqual(cvImg[0,0], 1)
    self.assertEqual(cvImg[0,1], 0)
    self.assertEqual(cvImg[0,2], 0)
    self.assertEqual(cvImg[1,0], 0)
    self.assertEqual(cvImg[1,1], 1)
    self.assertEqual(cvImg[1,2], 1)

class BlobSerializeTest(unittest.TestCase):
  def setUp(self):
    self.serializer = Blob.BlobSerializer()
    self.buf = StringIO.StringIO()

    self.result = Blob.BlobResult()
    
    blob1 = Blob.Blob()
    blob1.AddPoint((2,4))
    blob1.AddPoint((3,5))
    self.result.blobs.append(blob1)

    blob2 = Blob.Blob()
    blob2.AddPoint((6,10))
    self.result.blobs.append(blob2)

  def tearDown(self):
    self.buf.close()

  def FinishWrite(self):
    self.buf.seek(os.SEEK_SET, 0)

  def test_RoundTripSerialization(self):

    self.serializer.Serialize(self.buf, self.result, "test/testImage.jpg")
    self.FinishWrite()
    newBlobs, imgName = self.serializer.Deserialize(self.buf, blobDir='')

    self.assertEqual(imgName, "test/testImage.jpg")
    self.assertEqual(newBlobs.nBlobs(), self.result.nBlobs())
    self.assertEqual(newBlobs.GetBlob(0), self.result.GetBlob(0))
    self.assertEqual(newBlobs.GetBlob(1), self.result.GetBlob(1))

  def test_NormalSerializationString(self):
    self.serializer.Serialize(self.buf, self.result, "test/testImage.jpg")
    self.assertEqual(self.buf.getvalue(), "test/testImage.jpg\n2,4;3,5;\n6,10;\n")

  def test_NoBlobsString(self):
    self.serializer.Serialize(self.buf, Blob.BlobResult(), "image1")
    self.assertEqual(self.buf.getvalue(), "image1\n")

  def test_BlobOrderCorrect(self):
    newBlobs, imgName = self.serializer.Deserialize(
      StringIO.StringIO("test/testImage.jpg\n6,10;\n2,4;3,5;\n"),
      blobDir='')

    self.assertEqual(imgName, "test/testImage.jpg")
    self.assertEqual(newBlobs.nBlobs(), self.result.nBlobs())
    self.assertEqual(newBlobs.GetBlob(0), self.result.GetBlob(1))
    self.assertEqual(newBlobs.GetBlob(1), self.result.GetBlob(0))

  def test_NoImageName(self):
    self.assertRaises(IOError, self.serializer.Deserialize, (
      StringIO.StringIO("6,10;\n2,4;3,5;\n")), '')

  def test_InvalidImageName(self):
    self.assertRaises(IOError, self.serializer.Deserialize, (
      StringIO.StringIO("my_unknown_image.jpg\n6,10;\n2,4;3,5;\n")),
      '')

  def test_DeserializeMissingFinalSemiColons(self):
    newBlobs, imgName = self.serializer.Deserialize(
      StringIO.StringIO("test/testImage.jpg\n2,4;3,5\n6,10\n"),
      blobDir='')

    self.assertEqual(imgName, "test/testImage.jpg")
    self.assertEqual(newBlobs.nBlobs(), self.result.nBlobs())
    self.assertEqual(newBlobs.GetBlob(0), self.result.GetBlob(0))
    self.assertEqual(newBlobs.GetBlob(1), self.result.GetBlob(1))

  def test_MissingFinalNewline(self):
    newBlobs, imgName = self.serializer.Deserialize(
      StringIO.StringIO("test/testImage.jpg\n2,4;3,5;\n6,10;"),
      blobDir='')

    self.assertEqual(imgName, "test/testImage.jpg")
    self.assertEqual(newBlobs.nBlobs(), self.result.nBlobs())
    self.assertEqual(newBlobs.GetBlob(0), self.result.GetBlob(0))
    self.assertEqual(newBlobs.GetBlob(1), self.result.GetBlob(1))

  def test_MissingFinalNewlineAndSemiColon(self):
    newBlobs, imgName = self.serializer.Deserialize(
      StringIO.StringIO("test/testImage.jpg\n2,4;3,5;\n6,10"),
      blobDir='')

    self.assertEqual(imgName, "test/testImage.jpg")
    self.assertEqual(newBlobs.nBlobs(), self.result.nBlobs())
    self.assertEqual(newBlobs.GetBlob(0), self.result.GetBlob(0))
    self.assertEqual(newBlobs.GetBlob(1), self.result.GetBlob(1))

  def test_CreateFromBadStrings(self):
    self.assertRaises(ValueError, self.serializer.Deserialize, (
      StringIO.StringIO("test/testImage.jpg\n5,3;a,b;\n")), '')

    self.assertRaises(ValueError, self.serializer.Deserialize, (
      StringIO.StringIO("test/testImage.jpg\n2,4;3,5\n6;\n")),
      '')

class BlobLabelingTest(unittest.TestCase):
  def setUp(self):
    self.image = np.zeros((10,15))

  def test_NoBlobs(self):
    result = Blob.BlobResult()
    result.FindBlobs(self.image, 1)
    self.assertEqual(result.nBlobs(), 0)

  def test_OneBlob(self):
    self.image[2:4, 3:5] = 1
    
    result = Blob.BlobResult()
    result.FindBlobs(self.image, 1)
    self.assertEqual(result.nBlobs(), 1)

    blob = result.GetBlob(0)
    self.assertEqual(blob.Area(), 4)
    self.assertEqual(blob.minX, 3)
    self.assertEqual(blob.maxX, 4)
    self.assertEqual(blob.minY, 2)
    self.assertEqual(blob.maxY, 3)

  def test_One2PointAdjBlob(self):
    self.image[3:5,4] = 1;

    result = Blob.BlobResult(Blob.Blob.FOUR_CONNECTED)
    result.FindBlobs(self.image, 1)
    self.assertEqual(result.nBlobs(), 1)

    blob = result.GetBlob(0)
    self.assertEqual(blob.Area(), 2)
    self.assertEqual(blob.minX, 4)
    self.assertEqual(blob.maxX, 4)
    self.assertEqual(blob.minY, 3)
    self.assertEqual(blob.maxY, 4)

  def test_One2PointDiagBlob(self):
    self.image[3,4] = 1;
    self.image[4,5] = 1;

    eightResult = Blob.BlobResult(Blob.Blob.EIGHT_CONNECTED)
    eightResult.FindBlobs(self.image, 1)
    self.assertEqual(eightResult.nBlobs(), 1)

    blob = eightResult.GetBlob(0)
    self.assertEqual(blob.Area(), 2)
    self.assertEqual(blob.minX, 4)
    self.assertEqual(blob.maxX, 5)
    self.assertEqual(blob.minY, 3)
    self.assertEqual(blob.maxY, 4)

    fourResult = Blob.BlobResult(Blob.Blob.FOUR_CONNECTED)
    fourResult.FindBlobs(self.image, 1)
    self.assertEqual(fourResult.nBlobs(), 2)

    self.assertEqual(fourResult.GetBlob(0).Area(), 1);
    self.assertEqual(fourResult.GetBlob(1).Area(), 1);

  def test_OneBlobEightConnected(self):
    for row in range(2,4):
      for col in range(row-1,5,2):
        self.image[row,col] = 1

    result = Blob.BlobResult(Blob.Blob.EIGHT_CONNECTED)
    result.FindBlobs(self.image, 1)
    self.assertEqual(result.nBlobs(), 1)

    blob = result.GetBlob(0)
    self.assertEqual(blob.Area(), 4)
    self.assertEqual(blob.minX, 1)
    self.assertEqual(blob.maxX, 4)
    self.assertEqual(blob.minY, 2)
    self.assertEqual(blob.maxY, 3)

  def test_FilterSmallBlobs(self):
    # Creates a 4x2 blob
    self.image[2:6, 11:13] = 1;

    # Create a single pixel blob
    self.image[8,9] = 1;

    result = Blob.BlobResult(Blob.Blob.EIGHT_CONNECTED)
    result.FindBlobs(self.image, minimumSize=2)
    self.assertEqual(result.nBlobs(), 1)

    blob = result.GetBlob(0)
    self.assertEqual(blob.Area(), 8)
    self.assertEqual(blob.minX, 11)
    self.assertEqual(blob.maxX, 12)
    self.assertEqual(blob.minY, 2)
    self.assertEqual(blob.maxY, 5)

if __name__ == '__main__':
  unittest.main()
