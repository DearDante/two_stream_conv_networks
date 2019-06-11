import numpy as np
import optical_flow_prep as ofp
import sys, os
import pickle
import gc

def writeOF():
  '''create OpticalFlow images
  '''

  root = '../dataset/train'
  w = 224
  h = 224
  c = 0
  data = {}
  
  for path, dirs, files in os.walk(root):
    for filename in files:
      filename = filename[:-4]+'.h264'
      print(filename)
      count = ofp.writeOpticalFlow(path, filename, w, h, c)
      if count:
        data[filename] = count
      c += 1
      with open("done.txt", "a") as myfile:
        myfile.write(filename+'_'+str(c)+'\n')

  with open('../dataset/frame_count.pickle', 'wb') as f:
    pickle.dump(data, f)

writeOF()
