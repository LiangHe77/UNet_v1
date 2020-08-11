from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from netCDF4 import Dataset
from ium_data.config import *
from ium_data.data_process.get_date_list import read_datetime_list

"""
format conversion: nc(float)->npy(uint8)
"""

def read_nc_file(path, normalization=False):
    fh = Dataset(path, mode='r')
    dbz = fh.variables['DBZ'][:].data[0]
    dbz = np.rot90(dbz.transpose(1,0))
    fh.close()

    if normalization:
        dbz = np.clip((dbz+10) / 70, 0, 1)

    return dbz

def nc_read_img_uint8(path, normalization=False):
    fh = Dataset(path, mode='r')
    dbz = fh.variables['DBZ'][:].data[0]	# dbz.shape (800,800)
    dbz = np.rot90(dbz.transpose(1,0))		# transform
    dbz = 255 * (dbz + 10) / 70				# normalizaion: (-10, 60)->(0, 255)
    dbz = np.clip(dbz, 0, 255)
    fh.close()
    return dbz.astype(np.uint8)

def compare_nc_npy(data_nc, data_npy):
	assert(data_nc.shape == data_npy.shape)
	print(data_nc.shape)
	out = np.abs(data_nc - data_npy) > 0.2
	print(out.all())
	print(data_nc[100,3])
	print(data_npy[100,3])

if __name__ == '__main__':
	datetime_list = read_datetime_list(info_path)
	for dt in datetime_list:
		print(dt)
		dbz = nc_read_img_uint8(os.path.join(unzip_radar_path, dt[0:8], dt[8:]+".nc"))
		new_path = os.path.join(npy_radar_path, dt[0:8])
		if not os.path.exists(new_path):
			os.makedirs(new_path)
		np.save(os.path.join(new_path, dt[8:]+".npy"), dbz)
		# remove old nc file
		#os.remove(os.path.join(unzip_radar_path, dt[0:8], dt[8:]+".nc"))

	'''
	datetime_list = read_datetime_list(info_path)
	for dt in datetime_list:
		print(dt)
		data_npy = np.load(os.path.join(npy_radar_path, dt[0:8], dt[8:]+".npy")) / 255
		#data_nc = read_nc_file(os.path.join(radar_path, dt[0:8], dt[8:]+".nc"))
		#compare_nc_npy(data_nc, data_npy)
	'''
