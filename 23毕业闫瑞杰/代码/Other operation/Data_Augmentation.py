from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import random
import  random
import os

def image_rotate(image,label):

    """
    对图像进行一定角度的旋转
    :param image_path:  图像路径
    :param save_path:   保存路径
    :param angle:       旋转角度
    :return:
    """
    image_rotated = image.transpose(Image.ROTATE_90).convert('RGB')
    label_rotated = label.transpose(Image.ROTATE_90)
    return image_rotated,label_rotated
def image_rotate1(image,label):

    """
    对图像进行一定角度的旋转
    :param image_path:  图像路径
    :param save_path:   保存路径
    :param angle:       旋转角度
    :return:
    """
    image_rotated = image.transpose(Image.ROTATE_270).convert('RGB')
    label_rotated = label.transpose(Image.ROTATE_270)
    return image_rotated,label_rotated
def bright(image):
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.2
    image_brightened = enh_bri.enhance(brightness)

    return image_brightened.convert('RGB')
def ruidu(image):

    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = 2.3
    image_sharped = enh_sha.enhance(sharpness)

    return image_sharped.convert('RGB')
def sedu(image):
    enh_col = ImageEnhance.Color(image)
    color = 1.2
    image_colored = enh_col.enhance(color)

    return image_colored.convert('RGB')
def duibidu(image):
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.3
    image_contrasted = enh_con.enhance(contrast)

    return image_contrasted.convert('RGB')
def image_flip(image,label):
    image_transpose = image.transpose(Image.FLIP_LEFT_RIGHT).convert('RGB')
    label_transpose = label.transpose(Image.FLIP_LEFT_RIGHT)
    return image_transpose,label_transpose
def image_color(image,label):
    image_transpose = image.transpose(Image.FLIP_TOP_BOTTOM).convert('RGB')
    label_transpose = label.transpose(Image.FLIP_TOP_BOTTOM)
    return image_transpose,label_transpose


path_img = r'E:\实验\代码\Other operation\before_augmentation\images'
path_label = r'E:\实验\代码\Other operation\before_augmentation\label'
path_new_img = r'E:\实验\代码\Other operation\data_augmentation\images'
path_new_label = r'E:\实验\代码\Other operation\data_augmentation\label'
img_list = os.listdir(path_img)
label_list = os.listdir(path_label)

k=0
for i in range(len(img_list)):
    img = Image.open(path_img + '/' + img_list[i])
    label =Image.open(path_label + '/' + img_list[i][0:-4] + '.png')
	#保存原图
    img.convert('RGB').save(path_new_img + '/' + str(("%05d" % (k))) + '.jpg')
    label.save(path_new_label + '/' + str(("%05d" % (k))) + '.png')
    k += 1
	#角度旋转第一次
    img1,mask = image_rotate(img,label)
    img1.save(path_new_img + '/' + str(("%05d" % (k))) + '.jpg')
    mask.save(path_new_label + '/' + str(("%05d" % (k))) + '.png')
    k+=1
	#角度旋转第二次
    img1,mask = image_rotate1(img,label)
    img1.save(path_new_img + '/' + str(("%05d" % (k))) + '.jpg')
    mask.save(path_new_label + '/' + str(("%05d" % (k))) + '.png')
    k+=1
	#调整对比度
    img1 = duibidu(img)
    img1.save(path_new_img + '/' + str(("%05d" % (k))) + '.jpg')
    label.save(path_new_label + '/' + str(("%05d" % (k))) + '.png')
    k+=1
	#调整锐度
    img1 = ruidu(img)
    img1.save(path_new_img + '/' + str(("%05d" % (k))) + '.jpg')
    label.save(path_new_label + '/' + str(("%05d" % (k))) + '.png')
    k+=1
	#左右翻转
    img1,mask = image_flip(img,label)
    img1.save(path_new_img + '/' + str(("%05d" % (k))) + '.jpg')
    mask.save(path_new_label + '/' + str(("%05d" % (k))) + '.png')
    k+=1
	#上下翻转
    img1,mask = image_color(img,label)
    img1.save(path_new_img + '/' + str(("%05d" % (k))) + '.jpg')
    mask.save(path_new_label + '/' + str(("%05d" % (k))) + '.png')
    k += 1



    print(img_list[i] + 'is finished')

