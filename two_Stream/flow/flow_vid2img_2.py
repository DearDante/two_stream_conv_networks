import numpy as np
import sys, os
import pickle
import gc

def data_prep():
  with open('../dataset/frame_count.pickle', 'rb') as f1:
    frame_count = pickle.load(f1)
  with open('../dataset/merged_data.pickle', 'rb') as f2:
    merged_data = pickle.load(f2)

  root = './optical_flow_images'
  data = {}
  misplaced_data = []
  count = 0
  for path, dirs, files in os.walk(root):
    for filename in files:
      count += 1
      try:
        vidname = '_'.join(filename.split('.')[0].split('_')[1:])  + '.avi'
        fc = frame_count[vidname]
        index = merged_data[vidname]
        for i in range(1, fc//5+1):
          data[vidname+'@'+str(i)]=index
      except:
        misplaced_data.append(filename)
  print(data)
  with open('../dataset/flow_train_data.pickle', 'wb') as f3:
    pickle.dump(data, f3)
  
  with open('../dataset/misplaced_data.pickle', 'wb') as f4:
    pickle.dump(misplaced_data, f4)

if __name__ == "__main__":
  data_prep()
