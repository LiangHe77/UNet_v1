# -*- coding: utf-8 -*
import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from model.UNet import UNet
from utils.RAdam import RAdam
from evaluation import HKOEvaluation
from torch.optim import lr_scheduler
from ium_data.bj_iterator import BJIterator
from utils.utils import weight_loss, calculate_scores

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

#设置显卡优先级
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#设置参数
parser = argparse.ArgumentParser()
parser.add_argument('--Batch_size', help='batchsize', default=2, type=int)
parser.add_argument('--LR', help='learning rate', default=0.001, type=float)
parser.add_argument('--EPOCH', help='epoch number', default=100000, type=int)
parser.add_argument('--Step_size', help='warmup step size', default=2000, type=int)
parser.add_argument('--Gamma', help='warmup gamma', default=0.5, type=float)
parser.add_argument('--Optim', help='optimizer', default="Adam", type=str)
parser.add_argument('--Loss', help='the loss function', default="weight_loss" ,type=str)
parser.add_argument('--Model', help='model', default="UNet", type=str)
args = parser.parse_args()

#设置模型
if args.Model is "UNet":
    model = UNet(5,1)
    


#加快运算速度,设置这个 flag可以让内置的 cuDNN的 auto-tuner
#自动寻找最适合当前配置的高效算法,来达到优化运行效率的问题。   
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

#设置优化方式
if args.Optim is "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR)
    
elif args.Optim is "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.LR, momentum=0.9)

elif args.Optim is "RAdam":
    optimizer = RAdam(model.parameters())
    
#设置预热方式
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.Step_size, gamma=args.Gamma)

#设置损失函数
if args.Loss is 'weight_loss':
    loss_fn = weight_loss()
    
elif args.Loss is "MSE":
    loss_fn = nn.MSELoss()
    
#训练集，验证集，测试集迭代器
train_bj_iter = BJIterator(datetime_set="bj_train_set.txt",sample_mode="random",
                           seq_len=15,width=600,height=600)
                          
#训练函数
def train():
    
    global model
    
    #记录训练开始时间
    begin_time = time.time()

    model = model.cuda()
    model.train()
    
    epo_losses = []
    save_path = os.getcwd()
    iter_num = 0
    
    POD_List_10, CSI_List_10, FAR_List_10 = [], [], []
    POD_List_20, CSI_List_20, FAR_List_20 = [], [], []
    POD_List_30, CSI_List_30, FAR_List_30 = [], [], []
    POD_List_40, CSI_List_40, FAR_List_40 = [], [], []

    epo_loss = 0
    
    for i in range(args.EPOCH):
        
        frame_dat, mask_dat, datetime_batch, _ = \
                    train_bj_iter.sample(batch_size=args.Batch_size)  #迭代产生训练数据
        
        iter_num += args.Batch_size
        
        #(seq_num,batch_size,channels,height,width)
        frame_dat = torch.from_numpy(frame_dat) 
        
        #交换维度，交换维度为(batch_size,channels,seq_num,height,width)       
        frame_dat = frame_dat.permute(1,2,0,3,4).contiguous()  

        #前5帧作为输入帧,后5帧作为标签 
        input = frame_dat[:,:,0:5,:,:].cuda()                     
        label = frame_dat[:,:,14,:,:].cuda()                          

        optimizer.zero_grad()
       
        predict_images = model(input)
        loss = loss_fn(label,predict_images) 
        loss.backward()
        iter_loss = loss.item()
        epo_loss += iter_loss
        optimizer.step()
        
        #记录从开始到每次迭代后耗时
        end_time = time.time()
        
        print("epoch:{}/{},iter of the epoch:{}, loss:{:.6f}, consume_time:{:.4f}".format(i,\
                                            args.EPOCH, iter_num, iter_loss, end_time-begin_time)) 

        #每迭代5000次保存一次模型
        if (i+1) % 5000 == 0:  
            
            with open('./results/loss.txt','a') as l:
                l.write("  Loss: %s\n" % str(epo_loss/5000))  
      
            l.close()
   
            epo_loss = 0
        
            torch.save(model, save_path+"/checkpoints_60_v2/trained_model_{}.pth".format(i+1)) 
            
            (CSI1, POD1, FAR1, CSI2, POD2, FAR2, CSI3, POD3, FAR3, CSI4, POD4, FAR4) = \
                           validation((i+1)/5000,model)
            
            #记录每一个模型的CSI, POD, FAR,方便后续绘图
            CSI_List_10.append(CSI1)
            POD_List_10.append(POD1)
            FAR_List_10.append(FAR1)
            CSI_List_20.append(CSI2)
            POD_List_20.append(POD2)
            FAR_List_20.append(FAR2)
            CSI_List_30.append(CSI3)
            POD_List_30.append(POD3)
            FAR_List_30.append(FAR3)
            CSI_List_40.append(CSI4)
            POD_List_40.append(POD4)
            FAR_List_40.append(FAR4)

        #if args.Optim is not "RAdam":
	#    scheduler.step() 

	epo_losses.append(iter_loss)  

    ''' 
    绘制训练损失函数曲线
    '''    
    plt.plot(epo_losses,color=(0,0,1),label='loss') 
    min_index = np.argmin(epo_losses)
    min_data = np.min(epo_losses)
    plt.plot(min_index,min_data,"ks")
    show_min = "[{0},{1}]".format(min_index,min_data)
    plt.title("Train_Loss")
    plt.annotate(show_min,xytext=(min_index,min_data),xy=(min_index,min_data))
    plt.savefig('./log_60/tcn_train_loss.jpg')
    plt.close()    
    
    '''
    绘制验证CSI曲线
    '''
    plt.plot(CSI_List_10,color=(0,0,1),label='CSI') 
    max_index = np.argmax(CSI_List_10)
    max_data = np.max(CSI_List_10)
    plt.plot(max_index,max_data,"ks")
    show_max = "[{0},{1}]".format(max_index,max_data)
    plt.title("Validation_CSI_10")
    plt.annotate(show_max,xytext=(max_index,max_data),xy=(max_index,max_data))
    plt.savefig('./log_60/validation_csi_10.jpg')
    plt.close() 

    plt.plot(CSI_List_20,color=(0,0,1),label='CSI') 
    max_index = np.argmax(CSI_List_20)
    max_data = np.max(CSI_List_20)
    plt.plot(max_index,max_data,"ks")
    show_max = "[{0},{1}]".format(max_index,max_data)
    plt.title("Validation_CSI_20")
    plt.annotate(show_max,xytext=(max_index,max_data),xy=(max_index,max_data))
    plt.savefig('./log_60/validation_csi_20.jpg')
    plt.close() 

    plt.plot(CSI_List_30,color=(0,0,1),label='CSI') 
    max_index = np.argmax(CSI_List_30)
    max_data = np.max(CSI_List_30)
    plt.plot(max_index,max_data,"ks")
    show_max = "[{0},{1}]".format(max_index,max_data)
    plt.title("Validation_CSI_30")
    plt.annotate(show_max,xytext=(max_index,max_data),xy=(max_index,max_data))
    plt.savefig('./log_60/validation_csi_30.jpg')
    plt.close() 

    plt.plot(CSI_List_40,color=(0,0,1),label='CSI') 
    max_index = np.argmax(CSI_List_40)
    max_data = np.max(CSI_List_40)
    plt.plot(max_index,max_data,"ks")
    show_max = "[{0},{1}]".format(max_index,max_data)
    plt.title("Validation_CSI_40")
    plt.annotate(show_max,xytext=(max_index,max_data),xy=(max_index,max_data))
    plt.savefig('./log_60/validation_csi_40.jpg')
    plt.close() 
    
    '''
    绘制验证POD曲线
    '''
    plt.plot(POD_List_10,color=(0,0,1),label='POD') 
    max_index = np.argmax(POD_List_10)
    max_data = np.max(POD_List_10)
    plt.plot(max_index,max_data,"ks")
    show_max = "[{0},{1}]".format(max_index,max_data)
    plt.title("Validation_POD_10")
    plt.annotate(show_max,xytext=(max_index,max_data),xy=(max_index,max_data))
    plt.savefig('./log_60/validation_pod_10.jpg')
    plt.close() 

    plt.plot(POD_List_20,color=(0,0,1),label='POD') 
    max_index = np.argmax(POD_List_20)
    max_data = np.max(POD_List_20)
    plt.plot(max_index,max_data,"ks")
    show_max = "[{0},{1}]".format(max_index,max_data)
    plt.title("Validation_POD_20")
    plt.annotate(show_max,xytext=(max_index,max_data),xy=(max_index,max_data))
    plt.savefig('./log_60/validation_pod_20.jpg')
    plt.close() 

    plt.plot(POD_List_30,color=(0,0,1),label='POD') 
    max_index = np.argmax(POD_List_30)
    max_data = np.max(POD_List_30)
    plt.plot(max_index,max_data,"ks")
    show_max = "[{0},{1}]".format(max_index,max_data)
    plt.title("Validation_POD_30")
    plt.annotate(show_max,xytext=(max_index,max_data),xy=(max_index,max_data))
    plt.savefig('./log_60/validation_pod_30.jpg')
    plt.close() 

    plt.plot(POD_List_40,color=(0,0,1),label='POD') 
    max_index = np.argmax(POD_List_40)
    max_data = np.max(POD_List_40)
    plt.plot(max_index,max_data,"ks")
    show_max = "[{0},{1}]".format(max_index,max_data)
    plt.title("Validation_POD_40")
    plt.annotate(show_max,xytext=(max_index,max_data),xy=(max_index,max_data))
    plt.savefig('./log_60/validation_pod_40.jpg')
    plt.close() 
    
    ''' 
    绘制验证FAR曲线
    '''    
    plt.plot(FAR_List_10,color=(0,0,1),label='FAR') 
    min_index = np.argmin(FAR_List_10)
    min_data = np.min(FAR_List_10)
    plt.plot(min_index,min_data,"ks")
    show_min = "[{0},{1}]".format(min_index,min_data)
    plt.title("Validation_FAR_10")
    plt.annotate(show_min,xytext=(min_index,min_data),xy=(min_index,min_data))
    plt.savefig('./log_60/validation_far_10.jpg')
    plt.close() 
    
    plt.plot(FAR_List_20,color=(0,0,1),label='FAR') 
    min_index = np.argmin(FAR_List_20)
    min_data = np.min(FAR_List_20)
    plt.plot(min_index,min_data,"ks")
    show_min = "[{0},{1}]".format(min_index,min_data)
    plt.title("Validation_FAR_20")
    plt.annotate(show_min,xytext=(min_index,min_data),xy=(min_index,min_data))
    plt.savefig('./log_60/validation_far_20.jpg')
    plt.close() 

    plt.plot(FAR_List_30,color=(0,0,1),label='FAR') 
    min_index = np.argmin(FAR_List_30)
    min_data = np.min(FAR_List_30)
    plt.plot(min_index,min_data,"ks")
    show_min = "[{0},{1}]".format(min_index,min_data)
    plt.title("Validation_FAR_30")
    plt.annotate(show_min,xytext=(min_index,min_data),xy=(min_index,min_data))
    plt.savefig('./log_60/validation_far_30.jpg')
    plt.close() 

    plt.plot(FAR_List_40,color=(0,0,1),label='FAR') 
    min_index = np.argmin(FAR_List_40)
    min_data = np.min(FAR_List_40)
    plt.plot(min_index,min_data,"ks")
    show_min = "[{0},{1}]".format(min_index,min_data)
    plt.title("Validation_FAR_40")
    plt.annotate(show_min,xytext=(min_index,min_data),xy=(min_index,min_data))
    plt.savefig('./log_60/validation_far_40.jpg')
    plt.close() 

#验证函数
def validation(t, model):

    val_bj_iter = BJIterator(datetime_set="bj_valid_set.txt",sample_mode="random",
                           seq_len=15,width=600,height=600,begin_idx=0,end_idx=6000)

    np.seterr(divide='ignore',invalid='ignore')   #忽视分母为0的情况

    print('****begin validating {} model****'.format(t))
    
    valid_model = model
    valid_model.eval()
  
    CSI_Total_10, POD_Total_10, FAR_Total_10 = [], [], []
    CSI_Total_20, POD_Total_20, FAR_Total_20 = [], [], []
    CSI_Total_30, POD_Total_30, FAR_Total_30 = [], [], []
    CSI_Total_40, POD_Total_40, FAR_Total_40 = [], [], []
    
    #不进行梯度运算
    with torch.no_grad():
        
        for i in range(6000):
            
            frame_data, mask_data, Datetime_batch, _ = \
                        val_bj_iter.sample(batch_size=args.Batch_size)
                        
            frame_data = torch.from_numpy(frame_data)
            frame_data = frame_data.permute(1, 2, 0, 3, 4).contiguous()
            
            val_input = frame_data[:, :, 0:5, :, :].cuda()
            val_label = frame_data[:, :, 14, :, :].cuda()
            
            predict = valid_model(val_input)
            
            #反归一化
            predict = predict * 80
            val_label = val_label * 80
            
            print('validating dataset {}'.format(i))
            
            evaluation = HKOEvaluation(seq_len=1, use_central=False)
            
            val_label = val_label.cpu().detach().numpy()
            val_label = np.expand_dims(val_label,axis=1).transpose(2, 0, 1, 3, 4)
            predict = predict.cpu().detach().numpy()
            predict = np.expand_dims(predict,axis=1).transpose(2, 0, 1, 3, 4)
            
            evaluation.update(val_label, predict, mask=None)
            
        POD, CSI, FAR = evaluation.calculate_stat()
            
        #将结果写进txt文件
        evaluation.print_stat_readable()
        evaluation.save_txt_readable(t,'./results/result_60_600.txt')

        #将nan定义为0        
        POD, CSI, FAR = np.nan_to_num(POD), \
                        np.nan_to_num(CSI), \
                        np.nan_to_num(FAR)

        #CSI[x, y]中x代表计算第几帧的评价指标，0,1,...，4分别代表6min,12min,...,30min
        #y代表阈值的索引, thresholds=[10,20,30,40], 0代表阈值为10         
        CSI_Total_10.append(CSI[0,0])
        POD_Total_10.append(POD[0,0])
        FAR_Total_10.append(FAR[0,0])

        CSI_Total_20.append(CSI[0,1])
        POD_Total_20.append(POD[0,1])
        FAR_Total_20.append(FAR[0,1])

        CSI_Total_30.append(CSI[0,2])
        POD_Total_30.append(POD[0,2])
        FAR_Total_30.append(FAR[0,2])

        CSI_Total_40.append(CSI[0,3])
        POD_Total_40.append(POD[0,3])
        FAR_Total_40.append(FAR[0,3])
            
    #print("FINAL score :CSI={:.4f}, POD={:.4f}, FAR={:.4f}".format(\
    #      np.mean(CSI_Total), np.mean(POD_Total), np.mean(FAR_Total)))
            
    print('****end validating****')        
    
    return np.mean(CSI_Total_10), np.mean(POD_Total_10), np.mean(FAR_Total_10),\
           np.mean(CSI_Total_20), np.mean(POD_Total_20), np.mean(FAR_Total_20),\
           np.mean(CSI_Total_30), np.mean(POD_Total_30), np.mean(FAR_Total_30),\
           np.mean(CSI_Total_40), np.mean(POD_Total_40), np.mean(FAR_Total_40)
            
            
if __name__ == '__main__':
    
    train()    
            
    
    
    
                       
    

