import cv2
import numpy as np
import pickle
from PIL import Image
import os
import gc

def stackOpticalFlow(blocks, optical_train_data, img_rows, img_cols):
  '''
  
  Args:
    blocks:
    optical_train_data:
    img_rows:
    img_cols:
  
  Returns:
  '''
  firstTime = 1
  
  try:
    firstTimeOuter = 1
    for block in blocks:
      fx = []
      fy = []
      filename, blockNo = block.split('@')
      path = './optical_flow_images'
      blockNo = int(blockNo)  #用来叠加的帧数

      for i in range((blockNo*5)-4, (blockNo*5)+1):
        imgH = Image.open(path + '/' + 'h' + str(i) + '_' + filename.split('.')[0] + '.jpg')
        imgV = Image.open(path + '/' + 'v' + str(i) + '_' + filename.split('.')[0] + '.jpg')
        imgH = imgH.resize((img_rows, img_cols))
        imgV = imgV.resize((img_rows, img_cols))
        fx.append(imgH)
        fy.append(imgV)
      
      flowX = np.dstack((fx[0], fx[1], fx[2], fx[3], fx[4]))
      flowY = np.dstack((fy[0], fy[1], fy[2], fy[3], fy[4]))
      inp = np.dstack((flowX, flowY))
      inp = np.expand_dims(inp, axis=0)
      if not firstTime:
        inputVec = np.concatenate((inputVec, inp))
        labels = np.append(labels, int(optical_train_data[block])-1)
      else:
        inputVec = inp
        labels = np.array(int(optical_train_data[block])-1)
        firstTime = 0

    inputVec = np.rollaxis(inputVec, 3, 1)
    inputVec = inputVec.astype('float16', copy=False)
    labels = labels.astype('int', copy=False)
    gc.collect()
  
    return (inputVec, labels)

  except Exception as e:
    print(e)
    return (None, None)


def writeOpticalFlow(path, filename, w, h, c):
  count = 0
  try:
    cap = cv2.VideoCapture(os.path.join(path, filename))
    ret, frame1 = cap.read()#读取一帧图像存入frame1,若读取至文件尾则ret为False
    if not ret: return count
    
    frame1 = cv2.resize(frame1, (w,h))
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)#转换图像为灰度图
    
    folder = os.path.join('./optical_flow_images', filename)
    dir = './optical_flow_images'
    if not os.path.exists(dir):
      os.mkdir(dir)
    while(True):
      ret, frame2 = cap.read()
      if not ret: break
      count += 1
      print('count {0}'.format(count))
      if count % 1 == 0:
        frame2 = cv2.resize(frame2, (w, h))
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        horz = horz.astype('uint8')
        vert = vert.astype('uint8')

        cv2.imwrite(os.path.join(dir, 'h'+str(count)+'_'+filename[:-4]+'.jpg'), horz, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        cv2.imwrite(os.path.join(dir, 'v'+str(count)+'_'+filename[:-4]+'.jpg'), vert, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
   
        prvs = next_frame

    cap.release()
    cv2.destroyAllWindows()
    return count
  except Exception as e:
    print(e)
    print('Wasted')
    return count

