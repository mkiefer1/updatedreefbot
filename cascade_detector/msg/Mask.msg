# this message is used to mark where an object is present in an image
# this can be done either by a roi region on the image (less precise)
# or a mask (more precise)

sensor_msgs/RegionOfInterest roi

# in the case when mask is used, 'roi' specifies the image region and 'mask'
# (which should be of the same size) a binary mask in that region
sensor_msgs/Image mask