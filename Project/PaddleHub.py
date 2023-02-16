import paddlehub as hub
import cv2
import matplotlib.image as mpimg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

#! hub install stylepro_artistic

stylepro_artistic = hub.Module(name="stylepro_artistic")
results = stylepro_artistic.style_transfer(images=[{'content': cv2.imread("./images/countryside.jpg"),'styles':[cv2.imread("./images/girl.jpg")]}],
        alpha =1.0,
        visualization =True)# 原图展示

test_img_path ="./images/countryside.jpg"
img = mpimg.imread(test_img_path)
plt.figure(figsize=(10,10))
plt.imshow(img) 
plt.axis('off') 
plt.show()# 原图展示

test_img_path ="./images/girl.jpg"
img = mpimg.imread(test_img_path)
plt.figure(figsize=(10,10))
plt.imshow(img) 
plt.axis('off') 
plt.show()# 预测结果展示

test_img_path ="transfer_result/ndarray_1620094320.1111157.jpg"
img = mpimg.imread(test_img_path)# 展示预测结果图片
plt.figure(figsize=(10,10))
plt.imshow(img)  
plt.show()