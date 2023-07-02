import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.datasets import ImageFolder
# from utils.transform import get_transform_for_test
# from senet.se_resnet import FineTuneSEResnet50
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from torchvision import transforms, datasets
from model import efficientnet_b0 as create_model
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
 
data_root = r'D:\TJU\GBDB\set113\set113_images\test1'    # 测试集路径
# pth=np.zeros(shape=(100,4)).astype(np.str_)
# pth[pth == '0.0'] = ''


m = np.array(np.arange(5),dtype=str)  # 创建一个二维数组
test_weights_path1= r"/tmp/Test9_efficientNet/weights/model-43.pth"    # 预训练模型参数
test_weights_path2= r"/tmp/Test9_efficientNet/weights/model-35.pth"    # 预训练模型参数
test_weights_path3= r"/tmp/Test9_efficientNet/weights/model-33.pth"    # 预训练模型参数
test_weights_path4= r"/tmp/Test9_efficientNet/weights/model-31.pth"    # 预训练模型参数

# m[0]=test_weights_path
# pth[0]= "/tmp/Test9_efficientNet/weights/model-99.pth"
num_class = 10    # 类别数量
gpu = "cuda:0"    
 
 
# mean=[0.948078, 0.93855226, 0.9332005], var=[0.14589554, 0.17054074, 0.18254866]
def test(model, test_path):
# def test(model):
    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "/root"))  # get data root path
    image_path = os.path.join(data_root, "/root/autodl-tmp", "MIT indoor_data")  # flower data set path
    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "MIT Indoor_10_val"),
                                            transform=data_transform)

    batch_size = 16
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)
    # 加载测试集和预训练模型参数
    # test_dir = os.path.join(data_root, 'test_images')
    # class_list = list(os.listdir(test_dir))
    # class_list.sort()
    # transform_test = get_transform_for_test(mean=[0.948078, 0.93855226, 0.9332005],
    #                                         var=[0.14589554, 0.17054074, 0.18254866])
    # test_dataset = ImageFolder(test_dir, transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=1)
    # checkpoint = torch.load(test_path)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()
    # net = create_model(num_classes=10)
    # load pretrain weights
    # model_weight_path = "/tmp/Test9_efficientNet/weights/model-45.pth"
    # assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    model.load_state_dict(torch.load(test_path, map_location=device))
    model.to(device)
    model.eval()

    score_list = []     # 存储预测得分
    label_list = []     # 存储真实标签
    for i, (inputs, labels) in enumerate(validate_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
 
        outputs = model(inputs)
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
        score_tmp = outputs  # (batchsize, nclass)
 
        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())
 
    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
    print("score_array:", score_array.shape)  # (batchsize, classnum) softmax
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum]) onehot
 
    # 调用sklearn库，计算每个类别对应的precision和recall
    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    for i in range(num_class):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(label_onehot[:, i], score_array[:, i])
        average_precision_dict[i] = average_precision_score(label_onehot[:, i], score_array[:, i])
        print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])
 
    # micro
    precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(label_onehot.ravel(),
                                                                              score_array.ravel())
    average_precision_dict["micro"] = average_precision_score(label_onehot, score_array, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision_dict["micro"]))
    
    # 绘制所有类别平均的pr曲线
    plt.figure()
    plt.plot(recall_dict['micro'], precision_dict['micro'], label="Model1")

    # ax=plt.subplots()
    # ax.plot(recall_dict['micro'], precision_dict['micro'], label='Model1')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend("Model1",loc='lower left')
    # plt.title(
    #     'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    #     .format(average_precision_dict["micro"]))
    plt.savefig("/tmp/Test9_efficientNet/analyze/result/model2-pr_curve.jpg")
    plt.show()
    return precision_dict["micro"], recall_dict["micro"]
 
 
if __name__ == '__main__':
    # 加载模型
    precision_dic1 = dict()
    recall_dic1 = dict()
    precision_dic2 = dict()
    recall_dic2 = dict()
    precision_dic3 = dict()
    recall_dic3 = dict()
    precision_dic4 = dict()
    recall_dic4 = dict()
    model= create_model(num_classes=10)
    # seresnet = FineTuneSEResnet50(num_class=num_class)
    device = torch.device(gpu)
    # seresnet = seresnet.to(device)
    model = model.to(device)
    # test(seresnet, test_weights_path)
    # test(model, test_weights_path)
    precision_dic1["micro"], recall_dic1["micro"]=test(model,test_weights_path1)
    precision_dic2["micro"], recall_dic2["micro"]=test(model,test_weights_path2)
    precision_dic3["micro"], recall_dic3["micro"]=test(model,test_weights_path3)
    precision_dic4["micro"], recall_dic4["micro"]=test(model,test_weights_path4)
    
    plt.figure()
    plt.plot(recall_dic1['micro'], precision_dic1['micro'], label="Model1")
    plt.plot(recall_dic2['micro'], precision_dic2['micro'], label="Model2")
    plt.plot(recall_dic3['micro'], precision_dic3['micro'], label="Model3")
    plt.plot(recall_dic4['micro'], precision_dic4['micro'], label="Model4")
    # ax=plt.subplots()
    # ax.plot(recall_dict['micro'], precision_dict['micro'], label='Model1')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.legend(["Model1","Model2","Model3","Model4"],loc='lower left',fontsize=12,frameon=True,ncol=1,markerfirst=True)
    plt.legend(["Method1","Method2","Method3","Method4"],loc='lower left',fontsize=12,frameon=True,ncol=1,markerfirst=True)
    # plt.title(
    #     'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    #     .format(average_precision_dict["micro"]))
    plt.savefig("/tmp/Test9_efficientNet/analyze/result/mode2-pr_new22.jpg")
    plt.show()