# Python plugin that supports loading batch of images in parallel
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import threading
import os
from netCDF4 import Dataset
import struct
import concurrent.futures
#from concurrent.futures import ThreadPoolExecutor, wait

#_imread_executor_pool = ThreadPoolExecutor(max_workers=8)

'''
def nc_read_img(path, read_storage, grayscale=True, normalization=True):
    fh = Dataset(path, mode='r')
    dbz = fh.variables['DBZ'][:].data[0]
    if normalization:
        dbz = np.clip((dbz+10) / 70, 0, 1)
    read_storage[:] = dbz
    fh.close()
'''

def read_npy_frame(path, read_storage, grayscale=True):
    #path = os.path.join(npy_radar_path, path[0:8], path[8:]+".npy")
    read_storage[:] = np.load(path)

def read_nc_frame(path, read_storage):
    assert(os.path.exists(path))
    fh = Dataset(path, mode='r')
    dbz = fh.variables['DBZ'][:].data[0]
    dbz = np.rot90(dbz.transpose(1, 0))
    dbz = np.clip(dbz / 80.0, 0.0, 1.0)
    read_storage[:] = dbz
    fh.close()

def quick_read_frames(path_list, im_w=800, im_h=800, resize=False, frame_size=None, grayscale=True, normalization=True):
    """
    Multi-thread Frame Loader
    """
    frame_num = len(path_list)
    for path in path_list:
        if not os.path.exists(path):
            print(path)
            raise IOError

    if grayscale:
        read_storage = np.empty((frame_num, im_h, im_w), dtype=np.float32)
    else:
        read_storage = np.empty((frame_num, im_h, im_w, 3), dtype=np.float32)

    if resize:
        if grayscale:
            resize_storage = np.empty((frame_num, frame_size[0], frame_size[1]), dtype=np.uint8)
        else:
            resize_storage = np.empty((frame_num, frame_size[0], frame_size[1], 3), dtype=np.uint8)
        if frame_num == 1:
            cv2_read_img_resize(path=path_list[0], read_storage=read_storage[0],
                                resize_storage=resize_storage[0],
                                frame_size=frame_size, grayscale=grayscale)
        else:
            future_objs = []
            for i in range(frame_num):
                obj = _imread_executor_pool.submit(cv2_read_img_resize,
                                                   path_list[i],
                                                   read_storage[i],
                                                   resize_storage[i], frame_size, grayscale)
                future_objs.append(obj)
            wait(future_objs)
        if grayscale:
            resize_storage = resize_storage.reshape((frame_num, 1, frame_size[0], frame_size[1]))
        else:
            resize_storage = resize_storage.transpose((0, 3, 1, 2))
        return resize_storage[:, ::-1, ...]
    else:
        '''
        if frame_num == 1:
            read_npy_frame(path=path_list[0], read_storage=read_storage[0], grayscale=grayscale)
        else:
            future_objs = []
            for i in range(frame_num):
                obj = _imread_executor_pool.submit(read_npy_frame, path_list[i], read_storage[i], grayscale)
                future_objs.append(obj)
            wait(future_objs)
        if grayscale:
            read_storage = read_storage.reshape((frame_num, 1, im_h, im_w))
        else:
            read_storage = read_storage.transpose((0, 3, 1, 2))
        return read_storage[:, ::-1, ...]
        '''
        for i in range(frame_num):
            read_nc_frame(path=path_list[i], read_storage=read_storage[i])
        read_storage = read_storage.reshape((frame_num, 1, im_h, im_w))
        return read_storage
