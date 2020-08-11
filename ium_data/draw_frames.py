from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
#from skimage.transform import resize
import matplotlib as mpl
#mpl.use('agg')
from netCDF4 import Dataset
import cv2
import matplotlib.pyplot as plt

from ium_data.config import *
from ium_data.data_process.get_date_list import read_datetime_list
from ium_data.data_process.un_gzip import datetime2ncpath
from ium_data.data_process.un_gzip import get_datetime_boundary_index
from nowcasting.hko_evaluation import pixel_to_dBZ

"""
draw radar echo map
"""

def save_frame(frame_dat,  save_path, normalization=False):
	#fig = plt.figure()
	if normalization:
		frame_dat = pixel_to_dBZ(frame_dat)
	cmap = mpl.colors.ListedColormap(['whitesmoke','dodgerblue','cyan','limegreen','green',\
									  'yellow','goldenrod','orange','red','firebrick', 'darkred'])
	bounds = [-10, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
	plt.imshow(frame_dat, cmap=cmap, norm=norm)
	plt.grid(True, linestyle='solid')
	cb = plt.colorbar(norm = norm, ticks=bounds)
	cb.set_label('Radar reflectivity (dBZ)', size=14, style='italic')
	plt.savefig(save_path, dpi=100)
	plt.close()

def draw_frame(frame_dat, normalization=False, save_path=None, date=None, time=None):
	#fig = plt.figure()
	if normalization:
		norm = mpl.colors.Normalize(vmin=0, vmax=1)
	else:
		norm = mpl.colors.Normalize(vmin=0, vmax=80)

	plt.imshow(frame_dat, cmap='jet', norm=norm)
	plt.grid(True)
	plt.colorbar()
	plt.show()

def save_frames_pic_onebyone(file_names, output_path):
	for i, path in enumerate(file_names):
		fh = Dataset(path, mode = 'r')
		dbz = fh.variables['DBZ'][:].data[0]
		dbz = np.rot90(dbz.transpose(1,0))
		print(dbz.shape)
		dbz = cv2.resize(dbz, (400,400), interpolation=cv2.INTER_AREA)
		#dbz = np.clip((dbz + 10) / 70, 0, 1)
		#dbz[:] = cv2.resize(dbz, (400,400), interpolation=cv2.INTER_LINEAR)
		#dbz = resize(dbz[0], (400,400))
		#dbz = cv2.resize(dbz[0], (400,400), interpolation=cv2.INTER_LINEAR)
		fh.close()
		fig = plt.figure()
		norm = mpl.colors.Normalize(vmin=-10, vmax=60)
		#norm = mpl.colors.Normalize(vmin=0, vmax=1)
		plt.imshow(dbz, cmap='jet', norm = norm)
		plt.grid(True)
		plt.title(path[-18:-5])
		plt.colorbar()

		frame_date = path[-18:-10]
		frame_time = path[-9:-3]

		output_dir = os.path.join(output_path, frame_date)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		#plt.show()
		plt.savefig(os.path.join(output_dir, frame_date+"-"+frame_time+".jpg"), dpi=1000)
		plt.close()
		print("save pic %s-%s Done!" % (frame_date, frame_time))

def draw_all_frames(frame_dats, columns=4, vmin=0, vmax=1):
	frame_len = len(frame_dats)
	print(frame_dats.shape)
	rows = int(np.ceil(frame_len / columns))
	fig = plt.figure()
	for i in range(frame_len):
		ax = fig.add_subplot(rows, columns, i+1)
		im = ax.imshow(frame_dats[i],interpolation='none',cmap='jet', vmin=vmin, vmax=vmax)
		plt.grid(True)
	position=fig.add_axes([0.95,0.1,0.02,0.8])
	plt.colorbar(im, cax=position)
	plt.show()


def get_frames_path(str_datetime="20100602", start_idx=None, end_idx = None):
	datetime_list = read_datetime_list(info_path, "time_info.txt")
	if start_idx is None:
		(start_idx, end_idx) = get_datetime_boundary_index(datetime_list, str_datetime)

	print(start_idx, end_idx)
	return datetime2ncpath(raw_radar_path, datetime_list[start_idx:end_idx])	

'''
def draw_frames(file_names):
	length = len(file_names)
	if length > 5:
		rows = 2
		columns = np.ceil(length / 2)
	else:
		rows = 1
		columns = length

	fig = plt.figure()

	for i, path in enumerate(file_names):
		fh = Dataset(path, mode = 'r')
		dbz = fh.variables['DBZ'][:]
		#dbz[:] = cv2.resize(dbz, (400,400), interpolation=cv2.INTER_LINEAR)
		fh.close()
		ax = fig.add_subplot(rows, columns, i+1)
		im = ax.imshow(dbz[0], cmap='jet')
		plt.grid(True)
		plt.title(path[-18:-5])
		#plt.colorbar(im, orientation='horizontal')
	plt.show()
'''

if __name__ == "__main__":
	#20150820105333
	#time2draw = datetime.strptime("2010060104", '%Y%m%d%H')
	#frames_path = get_frames_path("2010061715")
	#save_frames_pic_onebyone(frames_path, pic_path)
	data = np.load("G:\\radar_ref_mosaic\\compositeRef2_npy_uint8\\20150611\\044734.npy") / 255.0
	print(np.sum(data > 0.6))
	draw_frame(data, True)
