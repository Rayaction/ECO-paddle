import os
import pickle

import numpy as np
classInd_src_path = "data/classInd.txt"
f = open(classInd_src_path)
label_dic = f.read().split('\n')[0:-1]
print(label_dic)


# split01_src_path = "data/trainlist01.txt"
split01_src_path = "data/testlist01.txt"
f = open(split01_src_path)
labels = f.read().split('\n')
label_washed = []
for label in labels:
    slash_index = label.find('/')+1
    avi_index = label.find('.avi')
    label_washed.append(label[slash_index:avi_index])

source_dir = 'data/data48916/UCF-101/'
target_train_dir = 'data/data48916/train_split01'
target_test_dir = 'data/data48916/test_split01'
target_val_dir = 'data/data48916/val_split01'
if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)
if not os.path.exists(target_val_dir):
    os.mkdir(target_val_dir)

for key in label_dic:
    label_name = key.split(' ')[1]
    label = key.split(' ')[0]
    each_mulu = label_name + '_jpg'
    print(each_mulu, label_name, label)
    label = str(int(label) - 1)
    label_dir = os.path.join(source_dir, each_mulu)
    label_mulu = os.listdir(label_dir)
    tag = 1
    for each_label_mulu in label_mulu:
        image_file = os.listdir(os.path.join(label_dir, each_label_mulu))
        image_file.sort()
        image_name = image_file[0][:-6]
        image_num = len(image_file)
        frame = []
        vid = image_name
        if vid in label_washed:
            for i in range(image_num):
                image_path = os.path.join(os.path.join(label_dir, each_label_mulu), image_name + '_' + str(i + 1) + '.jpg')
                frame.append(image_path)

            output_pkl = vid + '.pkl'
            # output_pkl = os.path.join(target_train_dir, output_pkl)
            output_pkl = os.path.join(target_test_dir, output_pkl)
            tag += 1
            f = open(output_pkl, 'wb')
            pickle.dump((vid, label, frame), f, -1)
            f.close()
