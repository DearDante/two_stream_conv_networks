import numpy as np
import cv2
import sys
import os
import pickle
import shutil
import gc

def write_images():
  root = '../dataset/train/'
 
#  with open('../dataset/merged_data.pickle', 'rb') as f:
#    var1 = pickle.load(f)

  for path, subdirs, files in os.walk(root):
    for filename in files:
      folder = './sp_images/' + filename.split('.')[0].split('_')[0]
      if not os.path.exists(folder):
        os.makedirs(folder)
      try:
        cnt = 0
        full_path = os.path.join(path, filename)
        cap = cv2.VideoCapture(full_path)
        fcnt = 1
        print(filename)   
        while(cap.isOpened()):
          ret, frame = cap.read()
          if not ret: break
          print(cnt)
          vid_name = filename.split('.')[0]
          img_path = folder + '/' + vid_name + '_{0}.jpg'.format(cnt+1)
          img_name = vid_name + '_{0}'.format(cnt+1)
      
          if fcnt % 1 == 0:
            print(img_name)
            cv2.imwrite(img_path, frame)
            cnt += 1
          fcnt += 1
        if cnt:
          with open("count.txt", "a") as txt:
            text = str(cnt) + " " + img_name.split('.')[0] + '\n'
            txt.write(text)
        cap.release()
        cv2.destroyAllWindows()
      except Exception as e:
        with open("logfile.txt", "a") as h:
          h.write(str(e))
        print("Some Error happened", e)
        cap.release()
        cv2.destroyAllWindows()

def data_prep():
  root = './sp_images/'
  
  with open('../dataset/merged_data.pickle', 'rb')as f:
    var1 = pickle.load(f)
  dic = {}
  vidno = 0
  for path, subdirs, files in os.walk(root):
    for filename in files:
      print(filename)
      vidname = '_'.join(filename.split('.')[0].split('_')[:-1])+'.avi'
      if vidname in var1:
      	dic[filename] = var1[vidname]
      else:
        with open('not_here.txt', 'a') as f:
          f.write(vidname+'\n')
    print(vidno)
    vidno += 1

  with open('../dataset/spacial_train_data_new.pickle', 'wb') as f:
    pickle.dump(dic, f)

if __name__ == '__main__':
  write_images()
  gc.collect()
  data_prep()
