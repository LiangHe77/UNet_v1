# -*- coding: utf-8 -*
'''
计算mae,mse,以及csi,pod,far等评分指标
'''

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

#计算MAE
def calculate_mae(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    return np.mean(np.abs(pred - label))

#计算MSE
def calculate_mse(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    return np.sqrt(np.mean((label - pred) ** 2))

#计算CSI，POD，FAR
def calculate_scores(pred, label, threshold):
    pred = np.array(pred.cpu())
    label = np.array(label.cpu())
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1
    label[label < threshold] = 0
    label[label >= threshold] = 1

    # compute TP, FP, FN, TN
    TP = float(np.sum(label[pred == label]))
    FP = float(np.sum(pred == 1) - TP)
    FN = float(np.sum(label == 1) - TP)
    TN = float(np.sum(label[pred == label] == 0))

    # compute CSI, POD, FAR
    if TP + FP + FN == 0:
        print("There is no CI")
        CSI = 0
    else:
        CSI = TP / (TP + FP + FN)

    if TP + FN == 0:
        print('There is no CI')
        POD = 0
    else:
        POD = TP / (TP + FN)

    if FP + TP == 0:
        FAR = 0
    else:
        FAR = FP / (FP + TP)

    # compute CI, NCI accuracy
    # acc = (TP + TN) / (TP + TN + FP + FN)
    # CI_acc = TP / ( TP + FN )
    # NCI_acc = TN / ( TN + FP )

    #print("CSI", CSI, "POD", POD, "FAR", FAR)
    return (CSI, POD, FAR)
  
class weight_loss2(nn.Module):
    def __init__(self):
        super(weight_loss2,self).__init__()
        
        self.loss = 0.0
        
    def forward(self,label,predict):
    
        #dBZ在0-20之间，权值设置为1；dBZ在20-30之间，权值设置为2；
        #dBZ在30-40之间，权值设置为5；dBZ在40-60权值设置为10；
        #dBZ在60-80之间，权值设置为30
    
        label = label.flatten()
        predict = predict.flatten()
        
        for i in range(len(label)):
            if 0 <= label[i] < 20:
                self.loss += (label[i]-predict[i])**2
            elif 20 <= label[i] < 30:
                self.loss += 2*((label[i]-predict[i]))**2
            elif 30 <= label[i] <40:
                self.loss += 5*((label[i]-predict[i]))**2
            elif 40 <= label[i] <60:
                self.loss += 10*((label[i]-predict[i]))**2
            elif 60<= label[i]:
                self.loss += 30*((label[i]-predict[i]))**2
        
        print(num)
    
        return self.loss/float(len(label))

#在该损失函数中加入带权重的MSE和以35为阈值的二分类BCE
class weight_loss(nn.Module):
    def __init__(self):
        super(weight_loss, self).__init__()
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, label, predict, base_weight=[1,1,2,5,10,30], thresholds=[0,0.125,0.25,0.375,0.5]):
    
        #dBZ在0-10(0-0.125)之间，权值设置为1；dBZ在10-20(0.125-0.25)之间，权值设置为2；
        #dBZ在20-30(0.25-0.375)之间，权值设置为5；dBZ在30-40(0.375,0.5),权值设置为10；
        #dBZ在40-80(0.5-1)之间，权值设置为30

        diff = np.diff(np.array(base_weight))
        
        label = label.cpu()  
        predict = predict.cpu()      

        mask = torch.ones_like(label) * base_weight[0]
        threshold_mask = torch.unsqueeze(label, dim=4) >= torch.Tensor(thresholds)
        base_weight = torch.Tensor(diff)\
            .reshape((1, 1, 1, 1, len(base_weight)-1))

        mask += torch.sum(threshold_mask.float() * base_weight, dim=-1)
           
        self.loss1 = mask * ((label-predict)**2)
        self.loss1 = torch.mean(self.loss1)
        
        label_ = label
        predict_ = predict*2 - 1

        label_[label_<0.4375] = 0
        label_[label_>=0.4375] = 1

        
        self.loss2 = self.loss_fn(predict_, label_)
        
        #print(self.loss1.item(), self.loss2.item()*0.5)
        
        return self.loss1 + self.loss2 * 0.1

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']






        

