import torch
import cv2
import random
import numpy as np

class DataAugmentation():

    def __init__(self,transforms=[]):
        self.transforms=transforms
        return

    def __call__(self, img,label):
        result_img=img
        result_label=label
        for trans in self.transforms:
            result_img,result_label=trans(result_img,result_label)
        return result_img,result_label

    def string(self):
        res=''
        for trans in self.transforms:
            res=res+trans.string()
        return res


class ColorBrighten():

    def __init__(self,beta=100):
        self.beta=beta
        return

    def __call__(self, img,label):
        height=img.shape[0]
        weight=img.shape[1]
        channels=img.shape[2]


        for row in range(height):
            for col in range(weight):
                for c in range(channels):
                    img[row,col,c]=min(img[row,col,c]+self.beta,255)
        return img,label

    def string(self):
        return  ('cb'+str(self.beta)).replace('.','_')

class ColorContrast():
    def __init__(self,alpha=0.9):
        self.alpha=alpha
        return

    def __call__(self, img,label):
        height=img.shape[0]
        weight=img.shape[1]
        channels=img.shape[2]

        for row in range(height):
            for col in range(weight):
                for c in range(channels):
                    img[row,col,c]=min(img[row,col,c]*self.alpha,255)
        return img,label

    def string(self):
        return ('cc'+str(self.alpha)).replace('.','_')

class ColorSaturability():
    def __init__(self,increment=1.0):
        self.increment=increment
        return

    def __call__(self, img,label):
        height=img.shape[0]
        weight=img.shape[1]
        channels=img.shape[2]

        for row in range(height):
            for col in range(weight):
                t1 = img[row,col,0]
                t2 = img[row,col,1]
                t3 = img[row,col,2]

                minVal = float(min(min(t1, t2), t3))
                maxVal = float(max(max(t1, t2), t3))
                delta = (maxVal - minVal) / 255.0
                L = 0.5 * (maxVal + minVal) / 255.0+1.0e-5
                S = max(0.5 * delta / L, 0.5 * delta / (1 - L))

                if self.increment > 0:
                    alpha = max(S, 1 - self.increment)+1.0e-5
                    alpha = 1.0 / alpha - 1
                    img[row,col,0] = img[row,col,0] + (img[row,col,0]- L * 255.0) * alpha
                    img[row,col,1] = img[row,col,1] + (img[row,col,1] - L * 255.0) * alpha
                    img[row,col,2] = img[row,col,2] + (img[row,col,2] - L * 255.0) * alpha
                else:
                    alpha = self.increment
                    img[row, col, 0] = L * 255.0 + (img[row,col,0] - L * 255.0) * (1 + alpha)
                    img[row,col,1]  = L * 255.0 + (img[row,col,1]  - L * 255.0) * (1 + alpha)
                    img[row,col,2]  = L * 255.0 + (img[row,col,2]  - L * 255.0) * (1 + alpha)
        return img,label

    def string(self):
        return ('cs'+str(self.increment)).replace('.','_')

class NoiseSaltpepper():

    def __init__(self,prob=0.01):
        self.prob=prob
        return

    def __call__(self, img,label):
        #channels = img.shape[2]
        thres = 1 - self.prob
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                #for c in range(channels):
                rdn = random.random()
                if rdn < self.prob:
                    img[i][j][0] = 0
                    img[i][j][1] = 0
                    img[i][j][2] = 0
                elif rdn > thres:
                    img[i][j][0] = 255
                    img[i][j][1] = 255
                    img[i][j][2] = 255
                #else:
                #    img[i][j][c] = img[i][j][c]
        return img,label

    def string(self):
        return ('ns'+str(self.prob)).replace('.','_')


class GeometryFlip():

    def __init__(self,geo_type=1):
        self.geo_type=geo_type
        return

    def __call__(self, img,label):
        img=cv2.flip(img, self.geo_type)
        label=cv2.flip(label, self.geo_type)
        return img,label

    def string(self):
        return ('gf'+str(self.geo_type)).replace('.','_')

class GeometryRotation():
    def __init__(self,angel=10.0):
        self.angel=angel
        return

    def __call__(self, img,label):
        rows, cols = img.shape[:2]
        rotMat = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.angel, 1.0);
        img=cv2.warpAffine(src=img,M=rotMat,dsize=(cols,rows));
        label=cv2.warpAffine(src=label,M=rotMat,dsize=(cols,rows),borderValue=1)
        return img,label

    def string(self):
        return ('gr'+str(self.angel)).replace('.','_')

class GeometryClip():
    def __init__(self,ratio=0.6):
        self.ratio=ratio
        return

    def __call__(self, img,label):
        h, w = img.shape[:2]

        ratio = random.random()

        scale = self.ratio + ratio * (1 - self.ratio)

        new_h = int(h * scale)
        new_w = int(w * scale)

        y = np.random.randint(0, h - new_h)
        x = np.random.randint(0, w - new_w)

        img = img[y:y + new_h, x:x + new_w, :]
        label=label[y:y + new_h, x:x + new_w]
        return img,label

    def string(self):
        return ('gc'+str(self.ratio)).replace('.','_')