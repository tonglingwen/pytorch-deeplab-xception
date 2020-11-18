import Augmentation
import os
import cv2
import random
from PIL import Image
import numpy as np


def aug(path,path0):
    color_bright=Augmentation.DataAugmentation([Augmentation.ColorBrighten()])
    #color_contrast=Augmentation.DataAugmentation([Augmentation.ColorContrast()])
    #color_saturability=Augmentation.DataAugmentation([Augmentation.ColorSaturability()])

    geomrtry_clip=Augmentation.DataAugmentation([Augmentation.GeometryClip()])
    geomrtry_flip=Augmentation.DataAugmentation([Augmentation.GeometryFlip()])
    geomrtry_rotation=Augmentation.DataAugmentation([Augmentation.GeometryRotation(angel=20)])

    train_images_path=os.path.join(path,'train/images')
    train_labels_path=os.path.join(path,'train/labels')

    aug_train_images_path=os.path.join(path0,'train/images')
    aug_train_labels_path=os.path.join(path0,'train/labels')

    group=[]
    group.append([])
    group.append([])
    group.append([])
    group.append([])

    for dir in os.listdir(train_images_path):
        group[random.randint(0,3)].append(dir)

    for i in range(len(group)):
        for j in range(len(group[i])):
            if i==3:
                break

            img = np.asarray(Image.open(os.path.join(train_images_path, group[i][j])))
            label = np.asarray(Image.open(os.path.join(train_labels_path, group[i][j])))

            if i==0:
                img,label=geomrtry_clip(img,label)
                last_name=os.path.splitext(group[i][j])[0]+'_clip.png'

            if i==1:
                img,label=geomrtry_flip(img,label)
                last_name=os.path.splitext(group[i][j])[0]+'_flip.png'

            if i==2:
                img, label = geomrtry_rotation(img, label)
                last_name = os.path.splitext(group[i][j])[0] + '_rotation.png'


            cv2.imwrite(os.path.join(aug_train_images_path, last_name),img)
            cv2.imwrite(os.path.join(aug_train_labels_path, last_name),label)

            #img = cv2.imread(os.path.join(train_images_path, group[i][j]))
            #label=cv2.imread(os.path.join(train_labels_path,group[i][j]))
            #if i==0:
                #img,label=color_bright(img,label)
                #cv2.imwrite(group[i][j]+".png",img)
                #cv2.imwrite(group[i][j]+"_label.png",label)
            #break

def synchronization_data(path):
    train_images_path = os.path.join(path, 'train/images')
    train_labels_path = os.path.join(path, 'train/labels')

    removes = []

    for dir in os.listdir(train_labels_path):
        image_file = os.path.join(train_images_path, dir)
        if not os.path.exists(image_file):
            removes.append(os.path.join(train_labels_path, dir))

    for dir in removes:
        os.remove(dir)


path=r'E:\huawei\data'
path1=r'E:\huawei\data_aug'
#synchronization_data(path)
aug(path=r'E:\huawei\data',path0=path1)