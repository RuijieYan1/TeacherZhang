import numpy as np
# m = np.array([np.arange(5), np.arange(3)], dtype=str)  # 创建一个二维数组
m = np.array(np.arange(5),dtype=str)  # 创建一个二维数组
m[0] = "Test9_efficientNet/weights/model-99.pth"
print(m)
print(m[0][1])
