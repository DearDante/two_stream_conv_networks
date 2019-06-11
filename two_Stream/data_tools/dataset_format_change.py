import sys, os
import pickle

data = {}

with open('../dataset/trainlist01.txt', 'r') as f1:
  for i in f1.readlines():
    tmp = i.split(' ')
    print(tmp)
    data[tmp[0]] = tmp[1][:-1]

with open('../dataset/merged_data.pickle', 'wb') as file:
  pickle.dump(data, file)
