# -*- coding: utf-8 -*-
import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import numpy as np
from matplotlib.patches import Polygon
import os
from netCDF4 import Dataset as Dataset_nc
import torch
from utils import *
from evaluation import HKOEvaluation




def colormap():

	#      black  darkgreen,   forestgreen, springgreen,mediumseagreen, mediumslateblue, blue, maroon, chocolate,'goldenrod, yellow,   darksalmon, salmon,  snow
	#cdict = ['#000000','#006400', '#228B22','#00FF7F', '#3CB371','#7B68EE', '#0000FF', '#800000',  '#D2691E', '#DAA520', '#FFFF00', '#E9967A', '#FA8072', '#FFFAFA']
	#      black springgreen, darkgreen,   forestgreen, mediumseagreen, mediumslateblue, blue, maroon, chocolate,'goldenrod, yellow,   darksalmon, salmon
	#cdict = ['#FFFFFF','#00FF7F','#006400', '#228B22', '#3CB371','#7B68EE', '#0000FF', '#800000',  '#D2691E', '#DAA520', '#FFFF00', '#E9967A']
	#return colors.ListedColormap(cdict, 'indexed')
	cdict = ['#F5F5F5', '#33A1C9','#00FFFF','#00C957', '#308014','#FFFF00', '#E3CF57', '#FF6100',  '#FF0000', '#C76114', '#5E2612']
	return colors.ListedColormap(cdict, 'indexed')



###########plt pictyre##########################
def save_frame(data,save_path):
        
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(projection='cyl', llcrnrlon=116.00, llcrnrlat=39.00, urcrnrlon=118.00, urcrnrlat=41.00, resolution='l')
    m.readshapefile(r"/media/4T/qxx/ChinaMap/Province_Beijing", 'Province_Beijing', drawbounds=True)
    lons, lats = m.makegrid(600,600) #change here
    x, y = m(lons, lats)
    parallels = np.arange(0., 90, 1)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
    meridians = np.arange(80., 130., 1)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)

    for shp in m.Province_Beijing:
        poly = Polygon(shp, edgecolor='w', fill=False, linewidth=0.8)
        ax.add_patch(poly)
	#m.pcolormesh(x, y, data, vmin=220, vmax=300, cmap=plt.cm.get_cmap('jet'))
    my_cmap = colormap()
    m.pcolormesh(x, y, data,vmin=5,vmax=60,cmap=my_cmap)
    cb = m.colorbar(ticks=[i for i in range(5,60,5)])
    cb.set_label(u'dBZ', size=14)
	#m.pcolormesh(x, y, data, cmap=my_cmap)
	#cb = m.colorbar()
	#cb.set_ticks(np.linspace(-80, 30, 6))
	#cb.set_ticklabels(( '-80', '-30', '-15', '0', '15','30'))
    plt.savefig(save_path, dpi=100)
    plt.clf()



if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0,2"

    path0 = "/home/lianghe/radar_predict/UNet/checkpoints_60_v2/trained_model_45000.pth"    
    test_model = torch.load(path0)
    test_model.eval()
    path = '/home/lianghe/radar_predict/20170911111730'
    file_folder_list = os.listdir(path)
    file_folder_list.sort()

    #select data
    for i in range(15):

        data = Dataset_nc(os.path.join(path,file_folder_list[i]))
        data = data.variables['DBZ'][:]
        data = np.clip(data / 80.0, 0.0, 1.0)
        data = data[0:1,100:700,100:700]
        if i==0:
            data_val = data
        else:
            data_val = np.concatenate((data,data_val),0)

    data_val = data_val[np.newaxis,np.newaxis,:]
    data_val = torch.from_numpy(data_val)

    with torch.no_grad():
        data_val = data_val.cuda()
        data_val = torch.cat([data_val,data_val],dim=0)
        print(data_val.shape)
        final_data = test_model(data_val[:,:,0:5,:,:])
    final_data = final_data.cpu()*80
    
    #\\\\\\\\\\\\\\\\\\\\\\\ calculate CSI /////////////////////////////////  '''

    final_data = torch.unsqueeze(final_data,1)
    print(final_data.shape)
    print(data_val[:,:,9:10].shape)
    true_data = data_val[0:1,:,9:10].cpu().detach().numpy().transpose(2, 0, 1, 3, 4)
    pre_data = final_data[0:1].cpu().detach().numpy().transpose(2, 0, 1, 3, 4)

    evaluation = HKOEvaluation(seq_len=1, use_central=False)

    evaluation.update(true_data*80,pre_data, mask=None)

    POD, CSI, FAR = evaluation.calculate_stat()
    
    evaluation.print_stat_readable()
    evaluation.save_txt_readable(1,'./results/test.txt')
    

    in_data = np.array(final_data)
    save_path = ('/home/lianghe/radar_predict/UNet/20170911111730/predict{}.png'.format(60))
    save_frame(in_data[0,0,0,:],save_path)
