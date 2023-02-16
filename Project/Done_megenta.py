import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#原始图片
img=plt.imread('./images/girl.jpg')
img=img/255.
img = tf.image.resize(img, (256, 256))
#plt.imshow(img)
#plt.show()
#print(img)

#风格图片
style_img=plt.imread('./images/5.jpg')
style_img=style_img/255.
style_img = tf.image.resize(style_img, (256, 256))

hub_model = hub.load('F:/Downloads/magenta_arbitrary-image-stylization-v1-256_2')

# 把输入规范一下，
# 改变维度
before_img_ = img[np.newaxis,:,:,:]
style_img_ = style_img[np.newaxis,:,:,:]
 
 
# 传入的是Tensor对象
before_img_ = tf.convert_to_tensor(before_img_,dtype=tf.float32)
style_img_ = tf.convert_to_tensor(style_img_,dtype=tf.float32)
 
outputs = hub_model(before_img_,style_img_)

# 输出有趣的图片[[[]]]
# print(outputs[0][0])
# 创建子图
plt.subplot(1,3,1)
plt.xlabel('before')
plt.xticks([])
plt.yticks([])
plt.imshow(img)
 
plt.subplot(1,3,2)
plt.xlabel('style')
plt.xticks([])
plt.yticks([])
plt.imshow(style_img)
 
plt.subplot(1,3,3)
plt.xlabel("after")
plt.xticks([])
plt.yticks([])
plt.imshow(outputs[0][0])
 
 
plt.show()
 
 
 
# 图片的保存
X = (outputs[0][0]) * 255
print(X)
# 将X转化为Tensor对象
img = tf.cast(X,dtype=tf.uint8)
# 编码回图片，二进制
img = tf.image.encode_png(img)
 
print(img)
# 图片保存的路径
save_path = './data/1.jpg'
 
# 文件的保存
with tf.io.gfile.GFile(save_path,'wb') as file:
    file.write(img.numpy())