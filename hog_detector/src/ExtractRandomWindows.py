#!/usr/bin/python
'''Extracts windows from a random location in an image.'''
usage='ExtractRandomWindows.py [options]'

import os.path
import os
from optparse import OptionParser
import cv2
import random

if __name__ == '__main__':
  parser = OptionParser(usage=usage)

  parser.add_option('-i', '--input', default=None,
                    help='File containing list of images to sample from')
  parser.add_option('--output_dir', default=None,
                    help='Directory to write the resulting images to')
  parser.add_option('--seed', default=195354, type='int',
                    help='Seed for the random number generator')
  parser.add_option('--width', default=96, type='int',
                    help='Width of the windows to extract')
  parser.add_option('--height', default=160, type='int',
                    help='Height in pixels of the windows to extract')
  parser.add_option('--nWindows', default=10, type='int',
                    help='Number of windows per image to extract')

  (options, args) = parser.parse_args()

  random.seed(options.seed)

  if not os.path.exists(options.output_dir):
    os.mkdir(options.output_dir)

  for line in open(options.input):
    imageFilename = line.strip()
    print 'Processing ' + imageFilename
    
    image = cv2.imread(imageFilename)

    for i in range(options.nWindows):
      x = random.randint(0, image.shape[1]-options.width-1)
      y = random.randint(0, image.shape[0]-options.height-1)
      window = image[y:(y+options.height), x:(x+options.width), :]

      windowFn = '%s.%i.jpg' % (os.path.basename(imageFilename), i)
      cv2.imwrite(os.path.join(options.output_dir, windowFn),
                  window)
      
