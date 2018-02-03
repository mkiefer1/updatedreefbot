# A message specifying the visual utility scores for a bunch of frames in an image
Header header

# Filename of the image
string image

# Processing time for all the regions of interest in the image
std_msgs/Duration processing_time

# The regions evaluated
sensor_msgs/RegionOfInterest[] regions

# The scores for each region
float32[] scores

# The maximum overlap between the region and a real object
float32[] overlaps
