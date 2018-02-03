from FishFinder import *
import sys
import cv
import cvHelper

def get_output_name(file_name):
  file_names = file_name.split('.')
  return file_names[0] + '_reefbot_annotated_surprise.mpeg'

###############################################################################

def find_fish(file_name,num_filters,input_file):
  finder = FishFinder(file_name,num_filters) 
  capture_file = cv.CreateFileCapture(input_file)
  output_file = get_output_name(input_file)


  loop = True
  frame = cv.QueryFrame(capture_file)
  video_writer = cv.CreateVideoWriter(output_file,cv.CV_FOURCC('P','I','M','1'),27.96,cv.GetSize(frame),1)
  while(loop):
    if (frame == None): print "no frame"; exit()

    #annotated_frame = finder.find_fish(frame)
    annotated_frame = finder.get_annotated_image(frame)
   
    cv.WriteFrame(video_writer,annotated_frame)#cvHelper.gray_colour_image(annotated_frame)) 
    frame = cv.QueryFrame(capture_file)
  ####  end while loop######

###############################################################################

if __name__=="__main__":

  file_name = "/home/furlong/projects/compression_saliency/code/neuro_16px_filters.txt"
  num_filters = 5
  input_file = "/home/furlong/Videos/fishVideos/MVI_5483.AVI"

  input_files = ['/home/furlong/Videos/fishVideos/MVI_5479.AVI','/home/furlong/Videos/fishVideos/MVI_5480.AVI','/home/furlong/Videos/fishVideos/MVI_5481.AVI','/home/furlong/Videos/fishVideos/MVI_5482.AVI','/home/furlong/Videos/fishVideos/MVI_5483.AVI','/home/furlong/Videos/fishVideos/MVI_5484.AVI','/home/furlong/Videos/fishVideos/MVI_5485.AVI','/home/furlong/Videos/fishVideos/MVI_5486.AVI','/home/furlong/Videos/fishVideos/MVI_5487.AVI','/home/furlong/Videos/fishVideos/MVI_5488.AVI','/home/furlong/Videos/fishVideos/MVI_5489.AVI','/home/furlong/Videos/fishVideos/MVI_5490.AVI','/home/furlong/Videos/fishVideos/MVI_5491.AVI']
  if len(sys.argv) > 1:
    file_name = sys.argv[1]
    num_filters = int(sys.argv[2])
    input_file = sys.argv[3]

  for in_file in input_files:
    find_fish(file_name,num_filters,in_file)
    
  
