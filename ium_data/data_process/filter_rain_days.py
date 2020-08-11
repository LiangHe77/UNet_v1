from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from netCDF4 import Dataset
import numpy as np
import os
from ium_data.config import *
from ium_data.data_process.get_date_list import read_datetime_list
from ium_data.data_process.save_npy import read_nc_file

"""
radar echo map is collected from 2010 year to 2017 year, 1084 days
find out every day's rainfull infomation

filter_datelist.txt	(sort 1084 days by "the_number_of_points_greater_35_dbz / day_frames")
filter_datalist_by_rain_full.txt (sort 1084 days by "days_rain_full / day_frames")
"""

def dirs_name(file_dir):
	for _, dirs, _ in os.walk(file_dir):
		return dirs[:-1]

def write_datelist_name(file_path, dirs):
	with open(os.path.join(file_path, "datelist.txt"), 'w') as f:
		for time in dirs:
			f.write(time)
			f.write("\n")

def read_datelist_name(file_path):
	date_list = []
	with open(os.path.join(file_path, "datelist.txt"), 'r') as f:
		for line in f:
			date_list.append(line.rstrip('\n'))
	return date_list

def points_greater_specific_dbz(file_path, threshold=35):
	dbz_data = read_nc_file(file_path)
	return np.sum(dbz_data > threshold)

def points_rainfull(file_path, threshold=0.5):
	dbz_data = read_nc_file(file_path)
	rain_full = np.power(10, dbz_data / 14.3) / 386
	rain_full[rain_full < threshold] = 0
	return np.sum(rain_full)

def get_days_dbz_num(dic_date_dbz, dic_date_frame, datetime_list):
	for date in datetime_list:
		points = points_greater_specific_dbz(os.path.join(raw_radar_path, date[0:8], date[8:]+".nc"))
		dic_date_dbz[date[0:8]] += points
		dic_date_frame[date[0:8]] += 1

	for key in dic_date_dbz.keys():
		if dic_date_frame[key] > 0:
			dic_date_dbz[key] /= dic_date_frame[key]

def get_days_rainfull(dic_date_dbz, dic_date_frame, datetime_list, filter_func=points_rainfull):
	for date in datetime_list:
		points = filter_func(os.path.join(raw_radar_path, date[0:8], date[8:]+".nc"))
		print(os.path.join(raw_radar_path, date[0:8], date[8:]+".nc"))
		dic_date_dbz[date[0:8]] += points
		dic_date_frame[date[0:8]] += 1

	for key in dic_date_dbz.keys():
		if dic_date_frame[key] > 0:
			dic_date_dbz[key] /= dic_date_frame[key]

def write_filter_rain_days(path, filter_data, file_name="filter_datelist_by_rainfull.txt"):
	with open(os.path.join(path, file_name), 'w') as f:
		for data in filter_data:
			f.write(data[0])
			f.write("\t")
			f.write(str(data[1]))
			f.write("\n")


if __name__ == "__main__":
	#write_datelist_name(info_path, dirs_name(raw_radar_path))
	date_list = read_datelist_name(info_path)
	datetime_list = read_datetime_list(info_path)
	print("num of days: ", len(date_list))
	print("num of datetime: ", len(datetime_list))
	dic_date_dbz = {date : 0 for date in date_list}
	dic_date_frame = {date : 0 for date in date_list}

	get_days_rainfull(dic_date_dbz, dic_date_frame, datetime_list)
	out = sorted(dic_date_dbz.items(), key=lambda item:item[1], reverse=True)
	write_filter_rain_days(info_path, out)
