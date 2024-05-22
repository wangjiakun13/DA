#split part of train dataset to val dataset
import os
import random
import shutil
import numpy as np

train_file_path = '/data/jiakunwang/dataset/MMWHS/data_np/data_list/train_mr.txt'
train_gt_path = '/data/jiakunwang/dataset/MMWHS/data_np/data_list/train_mr_gt.txt'

val_file_path = '/data/jiakunwang/dataset/MMWHS/data_np/data_list/val_mr_1.txt'
val_gt_path = '/data/jiakunwang/dataset/MMWHS/data_np/data_list/val_mr_gt_1.txt'

train_data_id = []
train_gt_id = []


with open(train_file_path,'r') as fp:
    train_file_list = fp.readlines()
    #获取每一行
    for i in range(len(train_file_list)):
        train_file_list[i] = train_file_list[i].strip('\n')
        #获取最后一个/后的内容
        # train_file_list[i] = train_file_list[i].split('/')[-1]


with open(train_gt_path,'r') as fp:
    train_gt_list = fp.readlines()
    for i in range(len(train_gt_list)):
        train_gt_list[i] = train_gt_list[i].strip('\n')
        # train_gt_list[i] = train_gt_list[i].split('/')[-1]

id_list = []
for i in range(len(train_file_list)):
    num = ''.join([x for x in train_file_list[i] if x.isdigit()])
    for j in range(len(train_gt_list)):
        if num in train_gt_list[j]:
            id_list.append(num)


#random split id list train and val
train_id = id_list[:int(len(id_list)*0.8)]
val_id = id_list[int(len(id_list)*0.8):]

for i in train_id:
    for j in range(len(train_file_list)):
        if i in train_file_list[j]:
            train_data_id.append(train_file_list[j])

    for k in range(len(train_gt_list)):
        if i in train_gt_list[k]:
            train_gt_id.append(train_gt_list[k])


print(len(train_data_id))



