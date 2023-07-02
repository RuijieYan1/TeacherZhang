import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from model import efficientnet_b0 as create_model
from tqdm import tqdm

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # ###############
    # data_root = os.path.abspath(os.path.join(os.getcwd(), "/root"))  # get data root path
    # image_path = os.path.join(data_root, "/root/autodl-tmp", "MIT indoor_data")  # flower data set path
    # assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)
    # validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test_val"),
    #                                         transform=data_transform)

    # batch_size = 16
    # validate_loader = torch.utils.data.DataLoader(validate_dataset,
    #                                               batch_size=batch_size, shuffle=False,
    #                                               num_workers=2)
    #########################
    # load image
    img_path = "/tmp/Test9_efficientNet/testpic/pic-tran8.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = '/tmp/Test9_efficientNet/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=10).to(device)
    # load model weights
    model_weight_path = "/tmp/Test9_efficientNet/weights/model-34.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        ##########
        # for val_data in tqdm(validate_loader):
        # for x, y in tqdm(validate_loader):
            # val_images, val_labels = val_data
        ###########
        # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            # output = torch.squeeze(model(val_images.to(device))).cpu()
            # output = torch.squeeze(model(x.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            print(predict)
            predict_cla = torch.argmax(predict).numpy()
            print(predict_cla)
            likelihood = torch.softmax(output, dim=0).cpu().numpy().max()
            print(likelihood)

            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                  predict[predict_cla].numpy())
            plt.title(print_res)
            for i in range(len(predict)):
              print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                   predict[i].numpy()))
            plt.savefig('/tmp/Test9_efficientNet/result/pre_result/B3pre.png')
            plt.show()


if __name__ == '__main__':
    main()
