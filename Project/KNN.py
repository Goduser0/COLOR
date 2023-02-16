import paddle
import paddle as P
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout, AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
import numpy as np
import PIL
from sklearn.neighbors import KNeighborsRegressor
import os
from skimage import io
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
import matplotlib.pyplot as plt
import math
import data 

place = P.CUDAPlace(0)
P.disable_static(place)
image_style ='./images/countryside.jpg'
print('风格图像：')
plt.imshow(io.imread(image_style))
plt.show()

image_content ='./images/girl.jpg'
print('内容图像：')
plt.imshow(io.imread(image_content))
plt.show()

edge_size =1
k =4

L_train =[]
ab_train =[]
for file in os.listdir(image_style):
    L0, ab0, _, _ = data.create_dataset(image_style+file)
    L_train.extend(L0)
    ab_train.extend(ab0)

knnr = KNeighborsRegressor(n_neighbors=k, weights='distance')
knnr.fit(L_train, ab_train)

