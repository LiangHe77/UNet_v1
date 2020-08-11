# -*- coding: utf-8 -*-   
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import sys
sys.path.append("/media/4T/lianghe/radar_predict/TCN/")
from datetime import datetime, timedelta
from ium_data.config import *

'''
Output the datetimelist of BJ-RADAR-DATA

datelist.txt
timeinfo.txt
'''
def dirs_name(file_dir):
	dirs_all = []
	for p in os.listdir(file_dir):
   	    dirs_all.append(p)
	dirs_all.sort()
	return dirs_all

def gzfile_name(file_dir):
	file_all = []
	for p in os.listdir(file_dir):
		file_all.append(p)
	file_all.sort()
	return file_all

def whole_datetime_list(file_path):
	datetime_list = []
	dirs = dirs_name(file_path)
	print(dirs)
	for dir_name in dirs:
        #print(dir_name)
		files = gzfile_name(os.path.join(file_path, dir_name))
		for i in range(len(files)):
			file_time = dir_name + files[i][0:6]
			print(file_time)

def write_datetime_list(file_path, output_path, datetime_list = None, file_name="time_info.txt"):
	if datetime_list is None:
		datetime_list = []
		dirs = dirs_name(file_path)
		for dir_name in dirs:
			files = gzfile_name(os.path.join(file_path, dir_name))
			for i in range(len(files)):
				file_time = dir_name + files[i][0:6]
				datetime_list.append(file_time)

	print("file num: ", len(datetime_list))
	with open(os.path.join(output_path, file_name), 'w') as f:
		for time in datetime_list:
			f.write(time)
			f.write("\n")

def read_datetime_list(file_path, fine_name="time_info.txt"):
	datetime_list = []
	print(os.path.join(file_path, fine_name))
	with open(os.path.join(file_path, "time_info.txt"), 'r') as f:
		for line in f:
			datetime_list.append(line.rstrip('\n').rstrip('\r'))
	print("datetime_list num: ", len(datetime_list))
	return datetime_list

'''
def file_name(file_dir):
	dirs_all = []
	for root, dirs, files in os.walk(file_dir):
		#print(type(dirs))
		#print(root)
		#print(dirs)
		#dirs_all.append(root) #当前目录路径
		#return dirs_all
		#print(dirs) #当前路径下所有子目录
		print(files) #当前路径下所有非目录子文件
		time.sleep(2)
	return dirs_all
'''
def test_datetime_delta(datetime_list):
	num = 0
	for i in range(len(datetime_list) - 1):
		time1 = datetime.strptime(datetime_list[i], '%Y%m%d%H%M%S')
		time2 = datetime.strptime(datetime_list[i+1], '%Y%m%d%H%M%S')
		
		time_delta = (time2 - time1).seconds / 60.0
		if time_delta < 1:
			num += 1
			print("time: ", datetime_list[i])

	print("num: ", num)

def filter_datetime_delta(datetime_list, low_min = 1):
	filtered_datetime_list = []
	num = 0
	flag_filter = False
	for i in range(len(datetime_list) - 1):
		if flag_filter:
			flag_filter = False
			continue

		time1 = datetime.strptime(datetime_list[i], '%Y%m%d%H%M%S')
		time2 = datetime.strptime(datetime_list[i+1], '%Y%m%d%H%M%S')
		
		time_delta = (time2 - time1).total_seconds() / 60.0

		if time_delta < low_min:
			num += 1; flag_filter = True
		filtered_datetime_list.append(datetime_list[i])

	filtered_datetime_list.append(datetime_list[-1])
	print("get rid of datetime num: ", num)
	return filtered_datetime_list

if __name__ == '__main__':
	write_datetime_list("/media/4T/yjt62/compositeRef2_nc/", info_path)
	#datetime_list = read_datetime_list(info_path, "time_info.txt")
	#date_list = read_datetime_list(info_path, "datelist.txt")
	#filtered_datetime_list = filter_datetime_delta(datetime_list)
	#write_datetime_list(radar_path, info_path, filtered_datetime_list, "time_info_weed_1min_delta.txt")
