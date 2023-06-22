from datasets.change_detection import ChangeDetection
from models.model_zoo import get_model
from utils.options import Options
from utils.palette import color_map

import numpy as np
import os
from PIL import Image
import shutil
import time
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metric import IOUandSek

if __name__ == "__main__":
    """
    Since the final evaluation is limited in 400 seconds in this challenge and the online inference speed 
    is hard to estimate accurately, we compute the inference speed in earlier iterations during inference 
    and choose not to use test-time augmentation in later iterations if time is not enough.
    """
    
    START_TIME = time.time()
    LIMIT_TIME = 400 - 20
    PAST_TIME = 0
    NO_TTA_TIME = 0
    TTA_TIME = 0

    args = Options().parse()

    torch.backends.cudnn.benchmark = True
    
    print(torch.cuda.is_available())
    testset = ChangeDetection(root=args.data_root, mode="test")
    testloader = DataLoader(testset, batch_size=8, shuffle=False,
                            pin_memory=True, num_workers=4, drop_last=False)

    model1 = get_model('pspnet', 'hrnet_w40', False, len(testset.CLASSES), False)
    model1.load_state_dict(torch.load('pspnet_hrnet_w40_48.74.pth'), strict=True)

    model = model1.cuda()
    model.eval()

    cmap = color_map()

    tbar = tqdm(testloader)
    TOTAL_ITER = len(testloader)
    CHECK_ITER = TOTAL_ITER // 5
    NO_TTA_ITER = TOTAL_ITER

    with torch.no_grad():
        for k, (img1, id) in enumerate(tbar):
            if k == CHECK_ITER - 1:
                iter_start_time = time.time()
            if k == CHECK_ITER + 1:
                PAST_TIME = time.time() - START_TIME
                NO_TTA_ITER = (LIMIT_TIME - PAST_TIME - NO_TTA_TIME * TOTAL_ITER +
                               (CHECK_ITER + 1) * TTA_TIME) / (TTA_TIME - NO_TTA_TIME)

            img1 = img1.cuda(non_blocking=True)

            out1_list = []

            if k < CHECK_ITER:
                    out1 = model(img1, True)
                    out1 = torch.softmax(out1, dim=1)
                    out1_list.append(out1)

            elif k == CHECK_ITER:
                    start = time.time()
                    out1 = model(img1, False)
                    out1 = torch.softmax(out1, dim=1)
                    out1_list.append(out1)
                    end = time.time()
                    NO_TTA_TIME = end - start

                    start = time.time()

                    out1 = model(img1, True)
                    out1 = torch.softmax(out1, dim=1)
                    out1_list.append(out1)
                    end = time.time()
                    TTA_TIME = end - start

                   # NO_TTA_TIME = PER_ITER_TIME - TTA_TIME + NO_TTA_TIME
                    # TTA_TIME = PER_ITER_TIME

            else:
                if k < NO_TTA_ITER:
                    use_tta = True
                else:
                    use_tta = False
                    out1 = model(img1, use_tta)
                    out1 = torch.softmax(out1, dim=1)
                    out1_list.append(out1)

            out1 = torch.stack(out1_list, dim=0)
            out1 = torch.sum(out1, dim=0)

            out1 = torch.argmax(out1, dim=1) + 1
            out1 = out1.cpu().numpy()

            for i in range(out1.shape[0]):
                mask = Image.fromarray(out1[i].astype(np.uint8), mode="P")
                mask.putpalette(cmap)
                mask.save("outdir/masks/test/im2/" + id[i])

            if k == CHECK_ITER - 1:
                iter_end_time = time.time()
                #PER_ITER_TIME = iter_end_time - iter_start_time

    END_TIME = time.time()
    print("Inference Time: %.1fs" % (END_TIME - START_TIME))