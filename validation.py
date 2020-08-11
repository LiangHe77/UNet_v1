# -*- coding: utf-8 -*- 
import torch
import torch.nn as nn
import os
import numpy as np
from ium_data.bj_iterator import BJIterator
from evaluation import HKOEvaluation
import matplotlib.pyplot as plt

device = torch.device('cuda:0')

if __name__ == '__main__':
    
    #获取模型的列表，并按迭代次数排序
    models_path = './checkpoints'
    models_list = os.listdir(models_path)
    models_list.sort(key=lambda x:int(x[14:-4]))
    print(models_list)
    
    #以CSI为评价指标，选取最优模型
    CSI_result = []
    
    #加载模型，进行验证
    for index,model in enumerate(models_list):
        
        val_model = torch.load(os.path.join(models_path,model))
        val_model.to(device)
        val_model.eval()
        
        val_bj_iter = BJIterator(datetime_set="bj_valid_set.txt",sample_mode="sequent",
                                 seq_len=15,width=600,height=600,begin_idx=0,end_idx=12226)
        
        for i in range(6113):
            frame_dat, mask_dat, datetime_batch, _ = val_bj_iter.sample(batch_size=2)
            frame_dat = torch.from_numpy(frame_dat)
            frame_dat = frame_dat.permute(1,2,0,3,4).contiguous() 
            
            val_input = frame_dat[:,:,0:5,:,:].to(device)                    
            val_label = frame_dat[:,:,9,:,:].to(device)
            
            with torch.no_grad():
                val_output = val_model(val_input)
                
            val_label = val_label * 80
            predict = val_output * 80
            
            print('validating dataset {}'.format(i))
            
            evaluation = HKOEvaluation(seq_len=1, use_central=False)
            
            val_label = val_label.cpu().detach().numpy()
            val_label = np.expand_dims(val_label,axis=1).transpose(2, 0, 1, 3, 4)
            predict = predict.cpu().detach().numpy()
            predict = np.expand_dims(predict,axis=1).transpose(2, 0, 1, 3, 4)
            
            evaluation.update(val_label, predict, mask=None)
            
        POD, CSI, FAR = evaluation.calculate_stat()
        
        #将结果写进txt文件
        evaluation.save_txt_readable(index, './results/val_30_600.txt')
        
        #将nan定义为0        
        POD, CSI, FAR = np.nan_to_num(POD), \
                        np.nan_to_num(CSI), \
                        np.nan_to_num(FAR)
            
        CSI_result.append(CSI)
        
    print(CSI_result)
        
    plt.plot(CSI_result,color=(0,0,1),label='val_CSI') 
    max_index = np.argmax(CSI_result)
    max_data = np.max(CSI_result)
    plt.plot(max_index,max_data,"ks")
    show_max = "[{0},{1}]".format(max_index,max_data)
    plt.title("Validation_CSI")
    plt.annotate(show_max,xytext=(max_index,max_data),xy=(max_index,max_data))
    plt.savefig('./log/val_CSI.jpg')
    plt.close() 
        
        
    
    
