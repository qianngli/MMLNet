# -*-coding:utf-8-*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
import torch
import torch.nn as nn
import torch.utils.data as Data
from argparse import ArgumentParser
from tqdm import tqdm
import os
import os.path as ops
import numpy as np
from data_utils import DatasetLoader
from model_base import ISNet as EGPNet
from loss import SoftLoULoss1, SoftLoULoss
import torch.nn.functional as F
from metric import mIoU



def parse_args():

    # Setting parameters
    parser = ArgumentParser(description='Implement of EGPNet model')
    parser.add_argument('--dataset', type=str, default='MDvsFA_cGAN', help='sirst_aug IRSTD-1k MDvsFA_cGAN')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=20,help='batch_size for training')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--warm-up-epochs', type=int, default=0, help='warm up epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')

    args = parser.parse_args()
    return args

        
class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        ## dataset
        trainset = DatasetLoader(args, mode='train')
        valset = DatasetLoader(args, mode='test')
        self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        self.val_data_loader = Data.DataLoader(valset, batch_size=1)
        
        print(len(self.val_data_loader))
        
        self.gradmask  = GradientMask()
        layer_blocks = [4] * 3
        self.net = EGPNet(layer_blocks, [16, 32, 64, 128])

        device = torch.device("cuda")
        self.net.apply(self.weight_init)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net = nn.DataParallel(self.net).cuda()

        self.net.to(device)

        ## criterion
        self.criterion1 = SoftLoULoss1()
        self.criterion2 = nn.BCELoss()
        self.bce = nn.BCELoss()
        
        self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.learning_rate, weight_decay=1e-4)

        self.best_miou = 0
        self.mIoU = mIoU(1)

        self.save_folder = ops.join('/data/liqiang/zw/result/', self.args.dataset)
        self.save_pkl = ops.join(self.save_folder, 'checkpoint_3_1_4_1k')
        
        if not ops.exists('result'):
            os.mkdir('result')
            
        if not ops.exists(self.save_folder):
            os.mkdir(self.save_folder)
        if not ops.exists(self.save_pkl):
            os.mkdir(self.save_pkl)


    def training(self, epoch):

        losses = []
        losses_edge = []
        self.net.train()

        lr = args.learning_rate /(2**(epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        tbar = tqdm(self.train_data_loader)
        for i, (data, labels, name) in enumerate(tbar):

            data = data.cuda()
            labels = labels[:,0:1,:,:].cuda()
            edge_gt = self.gradmask(labels.cuda())
            output, edge_out = self.net(data)
            
            loss_io = self.criterion1(output, labels)
            loss_edge = 10 * self.criterion2(edge_out, edge_gt)+ self.criterion1(edge_out, edge_gt)
            loss = loss_io + loss_edge

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses_edge.append(loss_edge.item())
            losses.append(loss.item())

            tbar.set_description('Epoch:%3d, lr:%f, train loss:%f, edge_loss:%f'
                                  % (epoch, trainer.optimizer.param_groups[0]['lr'], np.mean(losses), np.mean(losses_edge)))

    def validation(self, epoch):
        self.mIoU.reset()
        eval_losses = []
        eval_losses_edge = []

        self.net.eval()
        tbar = tqdm(self.val_data_loader)
        for i, (data, labels, name) in enumerate(tbar):
            if data.shape[2] % 8 != 0:
               data = data[:,:,:-(data.shape[2]%8),:]
               labels = labels[:,:,:-(labels.shape[2]%8),:]
               
            if data.shape[3] % 8 != 0:
               data = data[:,:,:, :-(data.shape[3]%8)]
               labels = labels[:,:,:, :-(labels.shape[3]%8)]

            with torch.no_grad():
                edge_gt = self.gradmask(labels.cuda())
                output, edge_out = self.net(data.cuda())
                labels = labels[:,0:1,:,:].cpu()
                output = output.cpu()
                edge_out = edge_out.cpu()
                edge_gt = edge_gt.cpu()

            loss_io = self.criterion1(output, labels)
            loss_edge = 10 * self.bce(edge_out, edge_gt)+ self.criterion1(edge_out, edge_gt)
            loss = loss_io + loss_edge
            eval_losses.append(loss.item())
            eval_losses_edge.append(loss_edge.item())

            self.mIoU.update(output, labels)
            _, mIoU = self.mIoU.get()

            tbar.set_description('Epoch:%3d, eval loss:%f, eval_edge:%f, mIoU:%f'
                                 %(epoch, np.mean(eval_losses), np.mean(eval_losses_edge), mIoU))


        pkl_name = 'Epoch-%3d_IoU-%.4f.pkl' % (epoch, mIoU)

        if mIoU > self.best_miou:
            torch.save(self.net, ops.join(self.save_pkl, pkl_name))
            self.best_iou = mIoU



    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.normal_(m.bias, 0)


class GradientMask(nn.Module):
    def __init__(self):
        super(GradientMask, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0
if __name__ == '__main__':
    args = parse_args()

    trainer = Trainer(args)
    for epoch in range(1, args.epochs+1):
        trainer.training(epoch)
        trainer.validation(epoch)

    print('Best IoU: %.5f' % (trainer.best_miou))






