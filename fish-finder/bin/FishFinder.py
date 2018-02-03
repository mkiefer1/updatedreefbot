from cv import *
from cvHelper import *
from math import sqrt, pi, log
import pickle

class Statistics:
  '''An implementation of Gaussian distribution'''
  def __init__(self):
    self.num_samples = 1.
    self.total_val = 0.
    self.total_squared = 1.
  ###########################################################################

  def mean(self):
    return self.total_val/self.num_samples
  ###########################################################################

  def variance(self):
    m = self.total_val/self.num_samples
    return (self.total_squared/self.num_samples) - m*m
  ###########################################################################
  def standard_deviation(self):
    return sqrt(self.variance())
  ###########################################################################

  def update(self,sample_count,vals,val_squared):
    self.num_samples += sample_count
    self.total_val += vals
    self.total_squared += val_squared
  ###########################################################################

class FishFinder:
  '''A class to examine an image for regions that are potentially fish.'''
  ###########################################################################
  def __init__(self,filter_file_name,num_filters,num_salient_points=5):
    self.filters = self.load_filters(filter_file_name,num_filters)
    self.statistics = []
    self.num_points = num_salient_points
    for filter in self.filters:
      self.statistics.append(Statistics())

  ###########################################################################
  def number_salient_points(num_salient=None):
    if not num_salient == None:
      self.num_points = num_salient
    return self.num_points
  ###########################################################################
  def load_filters(self,filter_file,num_filters):
    filters = pickle.load(open(filter_file))
    return map(filter_from_array,filters[0:num_filters]) 
    #raise NotImplementedError()
  ###########################################################################
  def get_annotated_image(self,image):
    points = self.find_fish(image,True)
    return self.annotate_image(image,points)
  ###########################################################################
  def find_fish(self,image,annotate=False):
    if self.filters == None:
      return None

    result = CloneImage(image)
    blank = CreateImage(GetSize(image),IPL_DEPTH_32F,1)
    SetZero(blank)
    
    grey_image = intensity(image)
    filtered_images = self.filter_image(grey_image)
    images_and_stats = zip(filtered_images,self.statistics)
    surprise_maps = [self.get_surprise(x[0],x[1]) for x in images_and_stats]
    surprise_map = reduce(add,surprise_maps)
    if annotate:
      return self.find_attended_points(surprise_map)
    else:
      m = merge(blank,blank,surprise_map)
      Convert(m,result)
      return add(image,result)
    #return self.find_attended_points(surprise_map)
  
  ###########################################################################
  def filter_image(self,image):
    return [filter(image,x) for x in self.filters]
  ###########################################################################
  def get_surprise(self,filtered_image,stats):
    img_size = GetSize(filtered_image)
    mean_val = stats.mean()
    var_val = stats.variance()

    mean_image = const_image(img_size,mean_val)
    var_image = const_image(img_size,1./var_val)
    scale_image = const_image(img_size,-1.*log(1./sqrt(pi*var_val)))

    result = CloneImage(filtered_image)

    Sub(filtered_image,mean_image,result)
    Mul(result,result,result,1.)
    Mul(result,var_image,result,1.)
    Add(result,scale_image,result)

    self.update_statistics(filtered_image,stats)
    return result
    
  ###########################################################################
  def update_statistics(self,filtered_image,stats):
    img_size = GetSize(filtered_image)
    num_pixels = img_size[0]*img_size[1]
    temp = CloneImage(filtered_image)
    total_pixels = Sum(temp)
    Mul(temp,temp,temp,1.)
    total_squared = Sum(temp)
    
    stats.update(num_pixels,total_pixels[0],total_squared[0])
    
  ###########################################################################
  def find_attended_points(self,map,supress_radius=16):
    points = []
    num_points_found = 0
    while (num_points_found < self.num_points):
      num_points_found += 1
      (min_val,max_val,min_loc,max_loc) = MinMaxLoc(map)
      points.append(max_loc)
      p1 = (max_loc[0]-supress_radius,max_loc[1]-supress_radius)
      p2 = (max_loc[0]+supress_radius,max_loc[1]+supress_radius)

      Rectangle(map,p1,p2,CV_RGB(0,0,0),-1,CV_AA,0)
    return points
  ###########################################################################
  def annotate_image(self,image,points,attention_radius=16):
    result = CloneImage(image)
    for point in points:
      p1 = (point[0]-attention_radius,point[1]-attention_radius)
      p2 = (point[0]+attention_radius,point[1]+attention_radius) 
      Rectangle(result,p1,p2,CV_RGB(255,0,0),4,CV_AA,0)
    return result
  ###########################################################################
