import numpy as np

import torch
import torchvision
import pylab
from PIL import Image
from torchvision import transforms as T
from model import efficientnet_b0 as create_model
import matplotlib.pyplot as plt
import torch
import torchvision
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

feature_extractor = create_model(num_classes=10)
# model = resnet34(num_classes=5)
# load model weights
# model_weight_path = "./AlexNet.pth"  # "./resNet34.pth"
model_weight_path = "/tmp/Test9_efficientNet/weights/model-99.pth"  # "./resNet34.pth"
feature_extractor.load_state_dict(torch.load(model_weight_path))
# feature_extractor = torchvision.models.resnet34(pretrained=True)
if torch.cuda.is_available():
	feature_extractor.cuda()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


save_output = SaveOutput()

hook_handles = []

for layer in feature_extractor.modules():
	if isinstance(layer, torch.nn.Conv2d):
		handle = layer.register_forward_hook(save_output)
		hook_handles.append(handle)

image = Image.open('/tmp/Test9_efficientNet/test_pic/b6.jpg')
print(image)
transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
X = transform(image).unsqueeze(dim=0).to(device)

out = feature_extractor(X)


def grid_gray_image(imgs, each_row: int):
    '''
    imgs shape: batch * size (e.g., 64x32x32, 64 is the number of the gray images, and (32, 32) is the size of each gray image)
    '''
    row_num = imgs.shape[0]//each_row
    for i in range(row_num):
        img = imgs[i*each_row]
        img = (img - img.min()) / (img.max() - img.min())
        for j in range(1, each_row):
            tmp_img = imgs[i*each_row+j]
            tmp_img = (tmp_img - tmp_img.min()) / (tmp_img.max() - tmp_img.min())
            img = np.hstack((img, tmp_img))
        if i == 0:
            ans = img
        else:
            ans = np.vstack((ans, img))
    return ans


img0 = save_output.outputs[6].cpu().detach().squeeze(0)
img0 = grid_gray_image(img0.numpy(), 8)
# img1 = save_output.outputs[1].cpu().detach().squeeze(0)
# img1 = grid_gray_image(img1.numpy(), 8)
# img6 = save_output.outputs[6].cpu().detach().squeeze(0)
# img6 = grid_gray_image(img6.numpy(), 8)
# img15 = save_output.outputs[15].cpu().detach().squeeze(0)
# img15 = grid_gray_image(img15.numpy(), 16)
# img29 = save_output.outputs[28].cpu().detach().squeeze(0)
# img29 = grid_gray_image(img29.numpy(), 16)


plt.figure(figsize=(15, 15))
plt.imshow(img0, cmap='gray')
plt.savefig('/tmp/Test9_efficientNet/analyze_weights_featuremap/result/map2.png', bbox_inches='tight')
pylab.show()



