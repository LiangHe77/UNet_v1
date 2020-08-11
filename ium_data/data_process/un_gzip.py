from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip  
import os  

from ium_data.config import *
from ium_data.data_process.get_date_list import read_datetime_list

"""
unzip *.nc.gz file
"""

def get_datetime_boundary_index(datetime_list, str_datetime):
	str_len = len(str_datetime)
	start_idx = None
	end_idx = None
	for idx, datetime in enumerate(datetime_list):
		if str_datetime in datetime[:str_len]:
			start_idx = idx
			break
	if start_idx is None:
		raise ValueError("str_datetime is not found!")

	for idx, datetime in enumerate(datetime_list[start_idx:]):
		if str_datetime not in datetime[:str_len]:
			end_idx = idx + start_idx
			break
	if end_idx is None:
		end_idx = len(datetime_list)

	print("num in boundary: ", end_idx - start_idx)

	return (start_idx, end_idx)

def datetime2ncpath(base_path, datetime_list):
	nc_file_paths = []
	for date in datetime_list:
		nc_file_paths.append(
			os.path.join(base_path, date[0:8], date[8:]+".nc")
		)

	return nc_file_paths

def un_zip(file_path, datetime_list, start_idx = 0, end_idx = None, unzip_path = None):
	nc_file_paths = datetime2ncpath(file_path, datetime_list[start_idx : end_idx])
	if not os.path.exists(unzip_path):
		os.makedirs(unzip_path)

	if unzip_path is None:
		unzip_file_paths = nc_file_paths
	else:
		unzip_file_paths = datetime2ncpath(unzip_path, datetime_list[start_idx : end_idx])
	
	for path in unzip_file_paths:
		if not os.path.exists(path[:-9]):
			os.makedirs(path[:-9])

	for idx, nc_file_path in enumerate(nc_file_paths):
		g = gzip.GzipFile(mode='rb', fileobj=open(nc_file_path+'.gz', 'rb'))
		open(unzip_file_paths[idx], 'wb').write(g.read())
		g.close()
		print("un_zip {}.gz done.".format(nc_file_path))


if __name__ == "__main__":
	datetime_list = read_datetime_list(info_path)
	#for str_time in ['2017','2016','2015','2014','2013','2012','2011','2010']:
	for str_time in ['2011','2010']:
		start_idx, end_idx = get_datetime_boundary_index(datetime_list, str_time)
		un_zip(raw_radar_path, datetime_list, start_idx, end_idx, unzip_radar_path)
