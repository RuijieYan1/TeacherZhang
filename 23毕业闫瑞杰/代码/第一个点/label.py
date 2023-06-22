from datasets.change_detection import ChangeDetection
from models.model_zoo import get_model
from utils.options import Options
from utils.palette import color_map

import numpy as np
from PIL import Image
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    args = Options().parse()

    dataset = ChangeDetection(root=args.data_root, mode='pseudo_labeling')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=4, drop_last=False)


    model1 = get_model('pspnet', 'hrnet_w40', False, len(dataset.CLASSES), False)

    model1.load_state_dict(
        torch.load('outdir/models/pspnet_hrnet_w40_48.74.pth'), strict=True)

    model = model1

    model = DataParallel(model).cuda()
    model.eval()

    tbar = tqdm(dataloader)
    cmap = color_map()
    for img1, mask1, id in tbar:
        img1 = img1.cuda()

        pseudo_mask1_list = []
        mask1 = mask1.numpy()
        with torch.no_grad():
            out1 = model(img1, True)

        out1 = torch.argmax(out1, dim=1).cpu().numpy() + 1

        pseudo_mask1 = np.zeros_like(out1)

        pseudo_mask1[mask1 != 0] = mask1[mask1 != 0]

        pseudo_mask1_list.append(np.arange(7) == pseudo_mask1[..., None])

        pseudo_mask1 = np.stack(pseudo_mask1_list, axis=0)
        pseudo_mask1 = np.sum(pseudo_mask1, axis=0).astype(np.float)

        out1 = np.argmax(pseudo_mask1, axis=3)

        for i in range(out1.shape[0]):
            mask = Image.fromarray(out1[i].astype(np.uint8), mode="P")
            mask.putpalette(cmap)
            mask.save("outdir/masks/train/im1/" + id[i])