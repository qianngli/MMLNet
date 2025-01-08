# -*-coding:utf-8-*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.utils.data as Data
from argparse import ArgumentParser
from tqdm import tqdm
import os.path as ops
import numpy as np
from data_utils import DatasetLoader
from metric import PD_FA, ROCMetric, mIoU, nIoU
from sklearn.metrics import auc
from torchvision.utils import save_image
from scipy.io import savemat


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of EGPNet model')
    parser.add_argument('--dataset', type=str, default='MDvsFA_cGAN', help='sirst_aug IRSTD-1k   MDvsFA_cGAN')

    args = parser.parse_args()
    return args


class Main(object):
    def __init__(self, args):
        self.args = args
  
        valset = DatasetLoader(args, mode='test')
        self.val_data_loader = Data.DataLoader(valset, batch_size=1)

        print(len(self.val_data_loader))

        self.ROC_metric = ROCMetric(1, 20)
        self.PD_FA_metric = PD_FA(1, 100)
        self.mIoU_metric = mIoU(1)
        self.nIoU_metric = nIoU(1)
        
        #Epoch-4_IoU-0.4637.pkl    MDvsFA_cGAN
        #Epoch-120_IoU-0.7382.pkl sirst_aug
        #Epoch-116_IoU-0.6490.pkl   1k
        
        # + SA  
        #aug   Epoch- 43_IoU-0.7545.pkl
        #cGAN  Epoch-  5_IoU-0.4506.pkl
        #MDFA Epoch- 32_IoU-0.5313.pkl
        #1k Epoch- 52_IoU-0.6466.pkl
        #87
        folder_name = 'Epoch- 99_IoU-0.5133.pkl'
        # folder_name = 'Epoch-56_IoU-0.7659.pkl'   
        # folder_name = 'Epoch- 99_IoU-0.5133.pkl'
        #path =  './result/' + self.args.dataset  + '/checkpoint'
        # path =  '/data/liqiang/zw/result/' + self.args.dataset  + '/checkpoint_3_1_4_4_2_e_1025_1k'
        path =  '/data/liqiang/zw/result/' + self.args.dataset  + '/checkpoint_3_1_4_0926_cgan'
        # path =  '/data/liqiang/zw/result/' + self.args.dataset  + '/checkpoint_3_1_4_0926_cgan'
        self.load_pkl = ops.join(path, folder_name)
        self.net = torch.load(self.load_pkl)

        self.save_img = './result/' + self.args.dataset + '/predict/'
        self.save_label= './result/' + self.args.dataset + '/label/'
        if not ops.exists(self.save_img):
            os.mkdir(self.save_img)
        if not ops.exists(self.save_label):
            os.mkdir(self.save_label)        

    def test(self):

        self.PD_FA_metric.reset()
        self.mIoU_metric.reset()
        self.ROC_metric.reset()
        self.nIoU_metric.reset()

        self.net.eval()

        tbar = tqdm(self.val_data_loader)

        for i, (data, labels, img_name) in enumerate(tbar):

            if data.shape[2] % 8 != 0:
               data = data[:,:,:-(data.shape[2]%8),:]
               labels = labels[:,:,:-(labels.shape[2]%8),:]
               
            if data.shape[3] % 8 != 0:
               data = data[:,:,:, :-(data.shape[3]%8)]
               labels = labels[:,:,:, :-(labels.shape[3]%8)]

            with torch.no_grad():
                #output, edge_out = self.net(data.cuda(), data.cuda())
                output, edge_out = self.net(data.cuda())
                labels = labels[:,0:1,:,:].cpu()
                output = output.cpu()

            output[output>1.]=1
            output[output<0]=0 
            self.ROC_metric.update(output, labels)
            
            
            output[output>=0.5]=1
            output[output<0.5]=0             
            
            self.mIoU_metric.update(output, labels)
            self.PD_FA_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)

            tpr, fpr, recall, precision = self.ROC_metric.get()
            _, mIOU = self.mIoU_metric.get()
            FA, PD = self.PD_FA_metric.get(len(self.val_data_loader), labels)
            _, nIOU = self.nIoU_metric.get()

            auc_value = auc(fpr, tpr)

            tbar.set_description('mIoU:%f, nIOU:%f, AUC:%f, FA:%f, PD:%f' 
                                 %(mIOU, nIOU, auc_value, FA[0]*1000000, PD[0]))

            output = output[0,:]
            labels = labels[0,:]
            save_image(output, self.save_img + img_name[0], normalize=True)
            save_image(labels, self.save_label + img_name[0], normalize=True)
            savemat('evulate_' + self.args.dataset + '_' + '.mat', {'miou': mIOU, 'prec': precision, 'recall': recall,  'auc': auc_value, 'tpr': tpr, 'fpr': fpr, 'FA': FA[0]*1000000, 'PD': PD[0]})

        
if __name__ == '__main__':
    args = parse_args()
    test_img = Main(args)
    test_img.test()







