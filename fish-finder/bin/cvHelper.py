from cv import *

################################################################################
def im2float(gray_img,img_size=None):
	"""
	Converts a one channel image with integer pixels (IPL_DEPTH_8U) to 
	a floating point image with pixel values in the range [0.0,1.0]
	"""
        if (img_size == None):
	if ((img_size = GetSize(gray_img) 
	grayscale = CreateImage(img_size,IPL_DEPTH_32F,1)
	ConvertScale(gray_img,grayscale,1.0/255.0)
	return grayscale

################################################################################
def get_channels(colour_img):
	"""
	Extracts the red, blue and green colour channels as
	32float pixel images 

	@returns a tuple with the (r,g,b) channels.
	"""
	img_size = GetSize(colour_img);

	blue_ch = CreateImage(img_size,IPL_DEPTH_8U,1)
	green_ch = CreateImage(img_size,IPL_DEPTH_8U,1)
	red_ch = CreateImage(img_size,IPL_DEPTH_8U,1)

	Split(colour_img,blue_ch,green_ch,red_ch,None)
        r = im2float(red_ch,img_size)
        g = im2float(green_ch,img_size)
        b = im2float(blue_ch,img_size)
	return (r,g,b)

################################################################################
def ohta_space(img):
  (r,g,b) = get_channels(img)
  i1 = scale(reduce(add,[r,g,b]),1./3.)
  i2 = sub(r,b) 
  i3 = scale(sub(sub(scale(g,2.),r),b),2)
  i4 = sub(r,g)
  i5 = sub(g,b)

  return (i1,i2,i3,i4,i5)

################################################################################
def scale(img,v):
  copy = CloneImage(img)
  ConvertScale(img,copy,v)
  return copy
  
def intensity(colour_image):
  img_size = GetSize(colour_image)
  grey_int_img = CreateImage(img_size,IPL_DEPTH_8U,1)
  intensity = CreateImage(img_size,IPL_DEPTH_32F,1)

  CvtColor(colour_image,grey_int_img,CV_BGR2GRAY)
  ConvertScale(grey_int_img,intensity,1.0/255.0)
  return intensity
# intensity ####################################################################

def filter(img,filter):
  result = CloneImage(img)
  Filter2D(img,result,filter)
  return result
# end filter ###################################################################

def add(a,b):
  result = CloneImage(a)
  Add(a,b,result)
  return result
# end add ######################################################################
def sub(a,b):
  result = CloneImage(a)
  Sub(a,b,result)
  return result
# end sub ######################################################################

def divide(a,b):
  """
  Does pixel-wise division of image a by image b
  i.e. returns a(i,j)/b(i,j) for all i,j
  """
  copy = CloneImage(a)
  Div(a,b,copy)
  return copy
# divide #######################################################################

def multiply(a,b,scale=1.0):
  result = CloneImage(a)
  Mul(a,b,result,scale)
  return result
# end multiply #################################################################

def log(a):
  copy = CloneImage(a)
  Log(a,copy)
  return copy

# end log ######################################################################

def const_image(size,value):
  img = CreateImage(size,IPL_DEPTH_32F,1)
  SetZero(img)
  AddS(img,value,img)
  return img
# const_image ##################################################################

def note_points(frame,points,side=8,colour=CV_RGB(255,255,255)):
  img_size = GetSize(frame)
  result = CloneImage(frame)
  for p in points:
    p1 = (p[0]-side/2,p[1]-side/2)
    p2 = (p[0]+side/2,p[1]+side/2)
    Rectangle(result,p1,p2,colour,4,CV_AA,0)

  return result
# end note_points ##############################################################

def filter_from_array(filter):
  cv_filter = CreateMat(filter.shape[0],filter.shape[1],CV_32FC1)
  for i in range(filter.shape[0]):
    for j in range(filter.shape[1]):
      Set2D(cv_filter,i,j,filter[i,j])

  return cv_filter
# end filter_from_array ############S

def normalize_range(img):
  result = CloneImage(img)
  (min_val,max_val,min_loc,max_loc) = MinMaxLoc(img)
  SubS(img,min_val,result)
  result = scale(result,255./(float(max_val-min_val)))
  return result

################################################################################

def gray_colour_image(src):
  result = CreateImage(GetSize(src),IPL_DEPTH_8U,3)
  gray_image = CreateImage(GetSize(src),IPL_DEPTH_8U,1)
  src_scaled= scale(src,255.)
  Convert(src_scaled,gray_image)
  Merge(gray_image,gray_image,gray_image,None,result)
  return result

################################################################################

def merge(r,g,b):
  result = CreateImage(GetSize(r),IPL_DEPTH_32F,3)
  Merge(r,g,b,None,result)
  return result

###############################################################################

def pow(img,p):
  result = CloneImage(img)
  Pow(img,result,float(p))
  return result

###############################################################################

def mean_variance(img):
  (mean,stddev) = AvgSdv(img)
  return (mean[0],stddev[0]**2)
  
################################################################################
