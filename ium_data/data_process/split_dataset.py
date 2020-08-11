from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os

import sys
sys.path.append("/media/4T/lianghe/radar_predict/TCN/")

from ium_data.config import *
from ium_data.data_process.get_date_list import read_datetime_list
from ium_data.data_process.mask_date import read_valid_datetime
from ium_data.data_process.mask_date import write_time_mask

"""
split datetime list into three part: train,valid,test
sample proportion = 11(train):1(valid):2(test)
bj_train_set.txt
bj_valid_set.txt
bj_test_set.txt
"""

#np.random.seed(92120)
np.random.seed(92123)

def valid_days_vs_whole_days(valid_days):
	whole_days = read_datetime_list(info_path, "datelist.txt")
	whole_days = set(whole_days)
	return whole_days - valid_days

def get_valid_days(valid_datetime):
	valid_datetime_bool = (valid_datetime[:,1] == 1)
	valid_datetime = valid_datetime[valid_datetime_bool][:,0]
	date = set()
	for dt in valid_datetime.astype(str):
		date.add(dt[:8])
	return date

def split_dataset(valid_datetime):
	""" (train / valid / test) are all using data from 2010-2017
	"""
	print("all_frames: ", valid_datetime.shape[0])
	print("valid_frames: ", np.sum(valid_datetime[:,1]))
	print("no_valid_frames: ", np.sum(1-valid_datetime[:,1]))

	valid_datetime_bool = (valid_datetime[:,1] == 1)
	valid_datetime_idx = np.where(valid_datetime_bool)[0]
	valid_datetime = valid_datetime[valid_datetime_bool][:,0]

	train_set = []
	valid_set = []
	test_set = []
	
	date = set()
	for dt in valid_datetime.astype(str):
		date.add(dt[:8])
	date = list(date)
	date.sort()

	for i in date:
		flag = np.random.randint(0,14)
		if flag < 11:
			train_set.append(i)
		elif flag == 11:
			valid_set.append(i)
		else:
			test_set.append(i)

	print("valid days: ", len(date))
	print("train set days: ", len(train_set))
	print("valid set days: ", len(valid_set))
	print("test set days: ", len(test_set))

	train_set_data = []
	valid_set_data = []
	test_set_data = []
	for dt,idx in zip(valid_datetime.astype(str), valid_datetime_idx):
		t_date = dt[:8]
		if t_date in train_set:
			train_set_data.append([dt,idx])
		elif t_date in valid_set:
			valid_set_data.append([dt,idx])
		else:
			test_set_data.append([dt,idx])

	print("train set samples: ", len(train_set_data))
	print("valid set samples: ", len(valid_set_data))
	print("test set samples: ", len(test_set_data))	

	#assert(np.sum(valid_datetime[:,1]) == valid_datetime.shape[0])
	return [train_set_data, valid_set_data, test_set_data]

def split_dataset2(valid_datetime):
	''' (train / valid) are using data from 2010-2016
		(test) are using data 2017	
	'''
	print("all_frames: ", valid_datetime.shape[0])
	print("valid_frames: ", np.sum(valid_datetime[:,1]))
	print("no_valid_frames: ", np.sum(1-valid_datetime[:,1]))

	valid_datetime_bool = (valid_datetime[:,1] == 1)
	valid_datetime_idx = np.where(valid_datetime_bool)[0]
	valid_datetime = valid_datetime[valid_datetime_bool][:,0]

	train_set = []
	valid_set = []
	test_set = []
	
	date = set()
	for dt in valid_datetime.astype(str):
		date.add(dt[:8])
	date = list(date)
	date.sort()

	for i in date:
		if i[:4] == "2017":
			test_set.append(i)
			continue
		flag = np.random.randint(0,14)
		if flag < 13:
			train_set.append(i)
		else:
			valid_set.append(i)

	print("valid days: ", len(date))
	print("train set days: ", len(train_set))
	print("valid set days: ", len(valid_set))
	print("test set days: ", len(test_set))

	train_set_data = []
	valid_set_data = []
	test_set_data = []
	for dt,idx in zip(valid_datetime.astype(str), valid_datetime_idx):
		t_date = dt[:8]
		if t_date in train_set:
			train_set_data.append([dt,idx])
		elif t_date in valid_set:
			valid_set_data.append([dt,idx])
		else:
			test_set_data.append([dt,idx])

	print("train set samples: ", len(train_set_data))
	print("valid set samples: ", len(valid_set_data))
	print("test set samples: ", len(test_set_data))	

	#assert(np.sum(valid_datetime[:,1]) == valid_datetime.shape[0])
	return [train_set_data, valid_set_data, test_set_data]

def write_date_set(dataset, filename):
	print("write set to path: ", filename)
	with open(filename, 'w') as f:
		for line in dataset:
			f.write(line[0])
			f.write("\t")
			f.write(str(line[1]))
			f.write("\n")

def read_datetime_set(filename, seq_len):
	"""
	read train/valid/test datetime set
	"""
	date_set = []
	with open(os.path.join(info_path, "squence_len_{}".format(seq_len), filename), 'r') as f:
		for line in f:
			new_line = line.rstrip('\n').split('\t')
			date_set.append([int(new_line[0]), int(new_line[1])])
	return np.array(date_set)

def generate_datetime_set(seq_len, low_min=4, high_min=9):
	dateset_path = os.path.join(info_path, "squence_len_{}".format(seq_len))
	train_dateset_path = os.path.join(dateset_path, "bj_train_set.txt")
	valid_dateset_path = os.path.join(dateset_path, "bj_valid_set.txt")
	test_dateset_path = os.path.join(dateset_path, "bj_test_set.txt")

	flag1 = os.path.exists(train_dateset_path)
	flag2 = os.path.exists(valid_dateset_path)
	flag3 = os.path.exists(test_dateset_path)
	if not (flag1 and flag2 and flag3):
		print("generate squence len {} dataset!".format(seq_len))
		write_time_mask(seq_len=seq_len, low_min=low_min, high_min=high_min)
		valid_datetime = read_valid_datetime(dateset_path,\
						"time_mask_"+str(seq_len)+"_timedelta_"+str(low_min)+str(high_min)+".txt")

		train_set, valid_set, test_set = split_dataset2(valid_datetime)
		write_date_set(train_set, train_dateset_path)
		write_date_set(valid_set, valid_dateset_path)
		write_date_set(test_set, test_dateset_path)
	else:
		print("dateset already exists!")

if __name__ == "__main__":
	generate_datetime_set(15)
	'''
	valid_datetime = read_valid_datetime(info_path, "time_mask.txt")
	train_set, valid_set, test_set = split_dataset(valid_datetime)
	write_date_set(train_set, "bj_train_set.txt")
	write_date_set(test_set, "bj_test_set.txt")
	write_date_set(valid_set, "bj_valid_set.txt")
	'''
	#train_date = read_datetime_set("bj_test_set.txt", 25)
	#print(len(train_date))


	'''
	seq_len = 25
	dateset_path = os.path.join(info_path, "squence_len_{}".format(seq_len))
	valid_datetime = read_valid_datetime(dateset_path, "time_mask_"+str(seq_len)+".txt")
	invalid_days = valid_days_vs_whole_days(get_valid_days(valid_datetime))
	print(invalid_days)
	'''

