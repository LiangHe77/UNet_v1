# -*- coding: utf-8 -*-   
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.append("/media/4T/lianghe/radar_predict/TCN/")

from ium_data.config import *
from ium_data.data_process.get_date_list import read_datetime_list

"""
output datetime mask:
timedelta = next_frame_datetime - current_frame_datetime
if all $(seq_len) frames timedelta in specific range(i.g: 5min-7min),
then the start frame corresponding "1",
which means those $(seq_len) frames can make up a available sample

time_mask_$(seq_len)_timedelta_${lowmin}${highmin}.txt
"""

def mask_datetime_delta(datetime_list, low_min=4, high_min=9):
	print("datetime_list number: ", len(datetime_list))
	mask_data = []
	num_true = 0
	num_false = 0
	for i in range(len(datetime_list) - 1):
		time1 = datetime.strptime(datetime_list[i], '%Y%m%d%H%M%S')
		time2 = datetime.strptime(datetime_list[i+1], '%Y%m%d%H%M%S')
		min_delta = (time2 - time1).total_seconds() / 60.0

		if low_min < min_delta and min_delta < high_min:
			num_true += 1
			mask_data.append(True)
		else:
			num_false += 1
			mask_data.append(False)

	mask_data.append(False); num_false += 1 # the last one
	assert len(mask_data) == len(datetime_list), 'len1: {}, len2: {}'.format(len(mask_data), len(datetime_list))
	print("num_false: ", num_false)
	print("num_true: ", num_true)
	return mask_data

def find_valid_datetime(mask_data, seq_len=15, invalid_frames=0):
	assert invalid_frames == 0, "no implement invalid frames threshold!"
	valid_data = []
	valid_len = len(mask_data) - seq_len + 1
	for i in range(valid_len):
		valid_frames = sum(mask_data[i:i+seq_len-1])  # sum (seq_len-1) frames
		if (valid_frames >= seq_len - 1 - invalid_frames):
			valid_data.append(True)
		else:
			valid_data.append(False)

	for i in range(seq_len - 1):
		valid_data.append(False)

	assert(len(mask_data) == len(valid_data))
	return valid_data

def write_valid_datetime(file_path, datetime_list, valid_data, seq_len, low_min=4, high_min=9):
	assert(len(datetime_list) == len(valid_data))
	file_path = os.path.join(file_path, "squence_len_{}".format(seq_len))
	if not os.path.exists(file_path):
		os.makedirs(file_path)
	with open(os.path.join(file_path,\
		"time_mask_"+str(seq_len)+"_timedelta_"+str(low_min)+str(high_min)+".txt"), 'w') as f:
		for i in range(len(valid_data)):
			f.write(datetime_list[i])
			f.write("\t")
			f.write(str(int(valid_data[i])))
			f.write("\n")

def read_valid_datetime(file_path, file_name = "time_mask.txt"):
	valid_datetime = []
	with open(os.path.join(file_path, file_name), 'r') as f:
		for line in f:
			new_line = line.rstrip('\n').split('\t')
			valid_datetime.append([int(new_line[0]), int(new_line[1])])

	return np.array(valid_datetime)

def test_valid_datetime(valid_datetime, start_year=15, end_year=None):
	print("datetime len: ", len(valid_datetime))
	print("valid datetime num: ", sum(valid_datetime[:,1]))
	print("%d year start: %s" % (start_year, valid_datetime[113588][0]))
	print("%d-now valid datetime num: %d" % (start_year, sum(valid_datetime[113588:,1])))

def write_time_mask(seq_len=15, low_min=4, high_min=9):
	datetime_list = read_datetime_list(info_path, "time_info.txt")
	mask_data = mask_datetime_delta(datetime_list, low_min, high_min)
	valid_data = find_valid_datetime(mask_data, seq_len)
	write_valid_datetime(info_path, datetime_list, valid_data, seq_len, low_min, high_min)

if __name__ == '__main__':
	write_time_mask(seq_len=15)

	#valid_datetime = read_valid_datetime(info_path, "time_mask_raw.txt")
	#test_valid_datetime(valid_datetime)
