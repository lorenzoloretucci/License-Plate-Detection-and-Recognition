# Code in cnn_fn_pytorch.py
from __future__ import print_function, division
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import os
import argparse
from time import time
from load_data import *
from torch.optim import lr_scheduler
from tool_metrics import *

# import minerl
# import gym

# def main():
# do your main minerl code
# env = gym.make('MineRLTreechop-v0')

# if __name__ == '__main__':
# main()

def main():
    # torch.backends.cudnn.enabled = False

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
                    help="path to the input file")
    ap.add_argument("-v", "--validation_images", required=True,
                    help="path to the validation input file")
    ap.add_argument("-n", "--epochs", default=25,
                    help="epochs for train")
    ap.add_argument("-s", "--start", default=0,
                    help="start epoch")
    ap.add_argument("-b", "--batchsize", default=4,
                    help="batch size for train")
    ap.add_argument("-r", "--resume", default='111',
                    help="file for re-train")
    ap.add_argument("-w", "--writeFile", default='wR2_MishAdaBN_out\wR2_MishAdaBN_try3-2.out',
                    help="file for output")
    args = vars(ap.parse_args())

    if torch.cuda.is_available():
        use_gpu = torch.device("cuda")
        print("working on gpu")
    else:
        use_gpu = torch.device("cpu")
        print("working on cpu")

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    class AdaptiveBatchNorm2d(nn.Module):
        def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
            super(AdaptiveBatchNorm2d, self).__init__()
            self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
            self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
            self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))

        def forward(self, x):
            return self.a * x + self.b * self.bn(x)

    class Mish(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x * (torch.tanh(torch.nn.functional.softplus(x)))
            return x

    class wR2(nn.Module):
        def __init__(self, num_classes=1000):
            super(wR2, self).__init__()
            hidden1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2, stride=2),
                AdaptiveBatchNorm2d(48),
                Mish(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(0.2)
            )
            hidden2 = nn.Sequential(
                nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
                AdaptiveBatchNorm2d(64),
                Mish(),
                nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2)
            )
            hidden3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
                AdaptiveBatchNorm2d(128),
                Mish(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(0.2)
            )
            hidden4 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
                AdaptiveBatchNorm2d(num_features=160),
                Mish(),
                nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2)
            )
            hidden5 = nn.Sequential(
                nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
                AdaptiveBatchNorm2d(192),
                Mish(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(0.2)
            )
            hidden6 = nn.Sequential(
                nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
                AdaptiveBatchNorm2d(192),
                Mish(),
                nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2)
            )
            hidden7 = nn.Sequential(
                nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
                AdaptiveBatchNorm2d(192),
                Mish(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(0.2)
            )
            hidden8 = nn.Sequential(
                nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
                AdaptiveBatchNorm2d(192),
                Mish(),
                nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2)
            )
            hidden9 = nn.Sequential(
                nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
                AdaptiveBatchNorm2d(192),
                Mish(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(0.2)
            )
            hidden10 = nn.Sequential(
                nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
                AdaptiveBatchNorm2d(192),
                Mish(),
                nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                nn.Dropout(0.2)
            )

            self.features = nn.Sequential(
                hidden1,
                hidden2,
                hidden3,
                hidden4,
                hidden5,
                hidden6,
                hidden7,
                hidden8,
                hidden9,
                hidden10
            )
            self.classifier = nn.Sequential(
                nn.Linear(23232, 100),
                # nn.ReLU(inplace=True),
                nn.Linear(100, 100),
                # nn.ReLU(inplace=True),
                nn.Linear(100, num_classes),
            )

        def forward(self, x):
            x1 = self.features(x)
            x11 = x1.view(x1.size(0), -1)
            x = self.classifier(x11)
            return x

    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train_model(model, criterion, optimizer, start_epoch, num_epochs, lr, learning_rate_decay, trainloader, valloader):
        # since = time.time()

        for epoch in range(start_epoch, num_epochs):
            lossAver = []
            lossAver_val = []

            model.train(True)
            start = time()

            IoU = []
            IoU_val = []

            for i, (XI, YI) in tqdm(enumerate(trainloader)):
                # print('%s/%s %s' % (i, times, time()-start))
                YI = np.array([el.numpy() for el in YI]).T
                if use_gpu:
                    x = Variable(XI.cuda(0))
                    y = Variable(torch.FloatTensor(YI).cuda(0), requires_grad=False)
                else:
                    x = Variable(XI)
                    y = Variable(torch.FloatTensor(YI), requires_grad=False)
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = model(x)

                # Compute and print loss
                loss = 0.0
                if len(y_pred) == batchSize:
                    loss += 0.8 * nn.L1Loss().cuda()(y_pred[:][:2], y[:][:2])
                    loss += 0.2 * nn.L1Loss().cuda()(y_pred[:][2:], y[:][2:])
                    #loss = criterion(y_pred, y)
                    lossAver.append(loss.data)

                    # Zero gradients, perform a backward pass, and update the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lrScheduler.step()
                    torch.save(model.state_dict(), storeName)
                    for k in range(batchSize):
                        [cx, cy, w, h] = y_pred.data.cpu().numpy()[k].tolist()
                        bbox_a = [(cx - w / 2) * 720, (cy - h / 2) * 1160, (cx + w / 2) * 720, (cy + h / 2) * 1160]

                        [cx_o, cy_o, w_o, h_o] = y.data.cpu().numpy()[k].tolist()
                        bbox_b = [(cx_o - w_o / 2) * 720, (cy_o - h_o / 2) * 1160, (cx_o + w_o / 2) * 720, (cy_o + h_o / 2) * 1160]

                        iou = intersection_over_union(bbox_a, bbox_b)
                        # if iou > 0.5:
                        IoU.append(iou)
                if i % 50 == 1:
                    with open('wR2_MishAdaBN_out\\wR2_MishAdaBN_iter_try3.out', 'a') as outF:
                        outF.write('train %s images, use %s seconds, loss %s\n' % (
                            i * batchSize, end - start, sum(lossAver[-50:]) / len(lossAver[-50:])))


            #lr *= learning_rate_decay
            #update_lr(optimizer, lr)

            end = time()
            start_v = time()
            model.eval()
            with torch.no_grad():

                for j, (XI, YI) in tqdm(enumerate(valloader)):
                    # print('%s/%s %s' % (i, times, time()-start))
                    YI = np.array([el.numpy() for el in YI]).T
                    if use_gpu:
                        x = Variable(XI.cuda(0))
                        y = Variable(torch.FloatTensor(YI).cuda(0), requires_grad=False)
                    else:
                        x = Variable(XI)
                        y = Variable(torch.FloatTensor(YI), requires_grad=False)
                        # Forward pass: Compute predicted y by passing x to the model
                        y_pred = model(x)

                    # Compute and print loss
                    loss = 0.0
                    if len(y_pred) == batchSize:
                        loss += 0.8 * nn.L1Loss().cuda()(y_pred[:][:2], y[:][:2])
                        loss += 0.2 * nn.L1Loss().cuda()(y_pred[:][2:], y[:][2:])
                        #loss = criterion(y_pred, y)
                        lossAver_val.append(loss.item())

                        for k in range(batchSize):
                            [cx, cy, w, h] = y_pred.data.cpu().numpy()[k].tolist()
                            bbox_a = [(cx - w / 2) * 720, (cy - h / 2) * 1160, (cx + w / 2) * 720, (cy + h / 2) * 1160]

                            [cx_o, cy_o, w_o, h_o] = y.data.cpu().numpy()[k].tolist()
                            bbox_b = [(cx_o - w_o / 2) * 720, (cy_o - h_o / 2) * 1160, (cx_o + w_o / 2) * 720,
                                      (cy_o + h_o / 2) * 1160]

                            iou = intersection_over_union(bbox_a, bbox_b)
                            # if iou > 0.5:
                            IoU_val.append(iou)

                            #loss += 1 - iou

                            #lossAver_val.append(loss.item())

                        #iou = bboxes_iou(y_pred, y, xyxy=False)
                        #IoU_val.append(iou)

                    end_v = time()
                    if j % 50 == 0:
                        with open('wR2_MishAdaBN_out\\wR2_val_MishAdaBN_iter_try3.out', 'a') as outF_v:
                            outF_v.write('train %s images, use %s seconds, loss %s\n' % (j * batchSize, end - start_v, sum(lossAver_val[-50:]) / len(lossAver_val[-50:])))

            end_f = time()
            print('%s %s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), sum(IoU) / len(IoU), end_f - start))
            print('%s %s %s %s\n' % (epoch, sum(lossAver_val) / len(lossAver_val), sum(IoU_val) / len(IoU_val), end_f - start))
            with open(args['writeFile'], 'a') as outF:
                outF.write('Epoch: %s %s %s %s\n' % (epoch, sum(lossAver) / len(lossAver), sum(IoU) / len(IoU), end_f - start))
            with open('wR2_MishAdaBN_out\\wR2_val_MishAdaBN_try3.out', 'a') as outF_v:
                outF_v.write('Epoch: %s %s %s %s\n' % (epoch, sum(lossAver_val) / len(lossAver_val), sum(IoU_val) / len(IoU_val), end_f - start))
            torch.save(model.state_dict(), storeName + str(epoch) + str(lr))
        return model

    numClasses = 4
    imgSize = (480, 480)
    batchSize = int(args["batchsize"]) if use_gpu else 8
    modelFolder = 'wR2_MishAdaBN_try3/'
    storeName = modelFolder + 'wR2_MishAdaBN_try3.pth'

    if not os.path.isdir(modelFolder):
        os.mkdir(modelFolder)

    epochs = int(args["epochs"])
    start = int(args["start"])
    #   initialize the output file
    with open(args['writeFile'], 'wb') as outF:
        pass
    resume_file = str(args["resume"])
    if not resume_file == '111':
        # epoch_start = int(resume_file[resume_file.find('pth') + 3:]) + 1
        if not os.path.isfile(resume_file):
            print("fail to load existed model! Existing ...")
            exit(0)
        print("Load existed model! %s" % resume_file)
        model_conv = wR2(numClasses)
        model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
        model_conv.load_state_dict(torch.load(resume_file))
        model_conv = model_conv.cuda()

    else:
        model_conv = wR2(numClasses)
        if use_gpu:
            model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
            model_conv = model_conv.cuda()

    print(model_conv)
    print(get_n_params(model_conv))

    criterion = nn.MSELoss()
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    lrScheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

    # optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.01)

    # dst = LocDataLoader([args["images"]], imgSize)
    dst = ChaLocDataLoader(args["images"].split(','), imgSize)
    trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers=4)

    dst_val = ChaLocDataLoader(args["validation_images"].split(','), imgSize)
    valloader = DataLoader(dst_val, batch_size=batchSize, shuffle=True, num_workers=4)

    lr_decay = 0.97
    momentum = 0.9
    lr = 0.001   ##1e-5 for 25k and 10k #0.00001 #0.000001 retrained with this one 1e-6 for 50k images

    model_conv = train_model(model_conv, criterion, optimizer_conv, start, epochs, lr, lr_decay, trainloader, valloader)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    print('loop')
    main()

# python rpnet\wR2_MishAdaBN.py -i rpnet\img_train -v rpnet\img_val -n 10 -b 4
# python rpnet\wR2_MishAdaBN.py -i rpnet\newtrain -v rpnet\newvalidation -n 10 -b 4
# python rpnet\wR2_MishAdaBN.py -i rpnet\train_50k -v rpnet\val_50k -n 10 -b 8
# python rpnet\wR2_MishAdaBN.py -i rpnet\train_50k -v rpnet\val_50k -n 10 -s 5 -b 4 -r wR2_MishAdaBN_try3\wR2_MishAdaBN_try3.pth47.737809374999997e-07
# python rpnet\wR2_MishAdaBN.py -i rpnet\bho -v rpnet\validation_AML -n 15 -s 6 -b 4 -r wR2_MishAdaBN_try2\wR2_MishAdaBN_try2.pth6