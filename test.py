# -*- coding: utf-8 -*
#该程序用于模型测试
import os
import torch
import numpy as np
import torch.nn as nn 
from evaluation import HKOEvaluation
from ium_data.bj_iterator import BJIterator

if __name__ == "__main__":
    
    #最佳的模型
    test_model = torch.load('./checkpoints/trained_model_12000.pkl' )
    test_model.eval()
    
    test_bj_iter = BJIterator(datetime_set="bj_test_set.txt",sample_mode="sequent",
                           seq_len=15,width=600,height=600,
                           begin_idx=None, end_idx=None)
     
    for i in range(10):
    
        frame_data, mask_dat, datetime_batch, _ = test_bj_iter.sample(batch_size=2)

        frame_data = torch.from_numpy(frame_data)
        frame_data = frame_data.permute(1, 2, 0, 3, 4).contiguous()   

        test_input = frame_data[:, :, 0:5, :, :].cuda()
        test_label = frame_data[:, :, 5:15, :, :].cuda()  

        #通过5帧预测之后的10帧，即预测后面一小时
        output1 = test_model(test_input)
        output2 = test_model(output1)
        output = torch.cat((output1,output2),2)

        test_label = test_label * 80
        output = output * 80

        print('testing dataset {}'.format(i))
        
        #计算评价指标        
        evaluation = HKOEvaluation(seq_len=10, use_central=False)
            
        test_label = test_label.cpu().detach().numpy().transpose(2, 0, 1, 3, 4)
        output = output.cpu().detach().numpy().transpose(2, 0, 1, 3, 4)        
    
        evaluation.update(test_label, output, mask=None)
        
    POD, CSI, FAR = evaluation.calculate_stat()
            
    #将结果写进txt文件
    evaluation.print_stat_readable()
    evaluation.save_txt_readable('./results/test_evaluation.txt')    
    

                    
                    
