import random
import gdal
import numpy as np

# 随机裁剪tif遥感影像
#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


'''
随机裁剪函数
ImagePath 原始影像路径
LabelPath 标签影像路径
IamgeSavePath 原始影像裁剪后保存目录
LabelSavePath 标签影像裁剪后保存目录
CropSize 裁剪尺寸
CutNum 裁剪数量
'''


def RandomCrop(ImagePath, IamgeSavePath, CropSize, CutNum):
    dataset_img = readTif(ImagePath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取哟昂数据

    #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
    fileNum = 588
    new_name = fileNum + 1
    while (new_name < CutNum + fileNum + 1):
        #  生成剪切图像的左上角XY坐标
        UpperLeftX = random.randint(0, height - CropSize)
        UpperLeftY = random.randint(0, width - CropSize)
        imgCrop = img[:,
                      UpperLeftX: UpperLeftX + CropSize,
                      UpperLeftY: UpperLeftY + CropSize]
        writeTiff(imgCrop, geotrans, proj, IamgeSavePath + "/%d.tif" % new_name)
        new_name = new_name + 1


#  裁剪得到300张256*256大小的训练集
RandomCrop(r"E:/实验/2020/2020cai/2020cai5.tif",
           r"E:/实验/2020/2020cut/",
           256,30)