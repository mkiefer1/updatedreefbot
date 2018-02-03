from FishFinder import *
from numpy import * 
import sys

if __name__=="__main__":
  dist = Statistics()

  print dist.mean()
  print dist.variance()
  print dist.standard_deviation()

  num_samples = int(sys.argv[1]) 
  s = random.normal(0,0.1,num_samples)

  dist.update(num_samples,sum(s),sum(s**2.))

  print dist.mean()
  print dist.variance()
  print dist.standard_deviation()

  

  
