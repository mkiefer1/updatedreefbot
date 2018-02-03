#!/usr/bin/python
'''Displays an image with red boxes around all the blobs.'''
usage='DisplayBlobBoxes.py'

import roslib; roslib.load_manifest('cv_blobs')
import rospy
import cv2
from cv_blobs import Blob
import numpy as np
from optparse import OptionParser
import os.path

if __name__ == '__main__':
  # All of the command line flags

  parser = OptionParser(usage=usage)

  parser.add_option('--input', '-i', dest='input',
                    help='Input blob file',
                    default=None)
  parser.add_option('--output', '-o', dest='output',
                    help='Optional output image file',
                    default=None)
  parser.add_option('--skip_display', action='store_false',
                    dest='show_display', default=True,
                    help='The image will not be shown if this flag is included.')

  (options, args) = parser.parse_args()

  # Open the blob file
  blobSerializer = Blob.BlobSerializer()
  f = open(options.input)
  try:
    (blobs, imgName) = blobSerializer.Deserialize(
      f,os.path.dirname(options.input))
  finally:
    f.close()

  # Open the image
  image = cv2.imread(imgName)

  # Draw the red boxes around the blobs
  for blob in blobs:
    box = blob.GetBoundingBox()
    cv2.rectangle(image, box[0:2], box[2:4], (0, 0, 255, 0.5), 1)

  if options.show_display:
    cv2.imshow(os.path.basename(imgName), image)
    cv2.waitKey()

  if options.output is not None:
    cv2.imwrite(options.output, image)
