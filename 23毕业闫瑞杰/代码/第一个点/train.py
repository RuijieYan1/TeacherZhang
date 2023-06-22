from datasets.change_detection import ChangeDetection
from models.model_zoo import get_model
from utils.options import Options
from utils.palette import color_map
from utils.metric import IOUandSek
from utils.callbacks import LossHistory

import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Trainer:
    def __init__(self, args):
        self.args = args

        trainset = ChangeDetection(root=args.data_root, mode="train", use_pseudo_label=args.use_pseudo_label)
        valset = ChangeDetection(root=args.data_root, mode="val")
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=False, num_workers=2, drop_last=True)
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=2, drop_last=False)

        self.model = get_model(args.model, args.backbone, args.pretrained,
                               len(trainset.CLASSES), False)
        if args.pretrain_from:
            self.model.load_state_dict(torch.load(args.pretrain_from), strict=False)

        if args.load_from:
            self.model.load_state_dict(torch.load(args.load_from), strict=True)

        self.criterion = CrossEntropyLoss(ignore_index=-1)

        self.optimizer = Adam([{"params": [param for name, param in self.model.named_parameters()
                                           if "backbone" in name], "lr": args.lr},
                               {"params": [param for name, param in self.model.named_parameters()
                                           if "backbone" not in name], "lr": args.lr * 10.0}],
                              lr=args.lr, weight_decay=args.weight_decay)

        self.model = DataParallel(self.model).cuda()

        self.iters = 0
        self.total_iters = len(self.trainloader) * args.epochs
        self.previous_best = 0.0

    def training(self):
        tbar = tqdm(self.trainloader)
        self.model.train()
        total_loss = 0.0
        loss_history = LossHistory("logs/")

        for i, (img1, mask1) in enumerate(tbar):
            img1 = img1.cuda()
            mask1 = mask1.cuda()

            out1 = self.model(img1)

            loss1 = self.criterion(out1, mask1 - 1)

            loss = loss1

            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iters += 1
            lr = self.args.lr * (1 - self.iters / self.total_iters) ** 0.9
            self.optimizer.param_groups[0]["lr"] = lr
            self.optimizer.param_groups[1]["lr"] = lr * 10.0

            tbar.set_description("Loss: %.3f" %(total_loss / (i + 1)))

    def validation(self):
        tbar = tqdm(self.valloader)
        self.model.eval()
        metric = IOUandSek(num_classes=len(ChangeDetection.CLASSES))
        if self.args.save_mask:
            cmap = color_map()

        with torch.no_grad():
            for img1, mask1, id in tbar:
                img1 = img1.cuda()

                out1 = self.model(img1, self.args.tta)
                out1 = torch.argmax(out1, dim=1).cpu().numpy() + 1

                if self.args.save_mask:
                    for i in range(out1.shape[0]):
                        mask = Image.fromarray(out1[i].astype(np.uint8), mode="P")
                        mask.putpalette(cmap)
                        mask.save("outdir/masks/val/im1/" + id[i])

                metric.add_batch(out1, mask1.numpy()-1)

                score, miou, sek, kappa = metric.evaluate()

                tbar.set_description("Score: %.2f, IOU: %.2f, SeK: %.2f, Kappa: %.2f" % (score * 100.0, miou * 100.0, sek * 100.0, kappa *100.0))

        if self.args.load_from:
            exit(0)

        score *= 100.0
        if score >= self.previous_best:
            if self.previous_best != 0:
                model_path = "outdir/models/%s_%s_%.2f.pth" % \
                             (self.args.model, self.args.backbone, self.previous_best)
                if os.path.exists(model_path):
                    os.remove(model_path)

            torch.save(self.model.module.state_dict(), "outdir/models/%s_%s_%.2f.pth" %
                       (self.args.model, self.args.backbone, score))
            self.previous_best = score


if __name__ == "__main__":
    args = Options().parse()
    trainer = Trainer(args)

    if args.load_from:
        trainer.validation()

    for epoch in range(args.epochs):
        print("\n==> Epoches %i, learning rate = %.5f\t\t\t\t previous best = %.2f" %
              (epoch, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))
        trainer.training()
        trainer.validation()