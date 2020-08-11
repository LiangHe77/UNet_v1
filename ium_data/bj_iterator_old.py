from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os

from ium_data.config import *
#from ium_data.draw_frames import draw_all_frames, draw_frame
from ium_data.read_frames import quick_read_frames
from ium_data.data_process.get_date_list import read_datetime_list
from ium_data.data_process.split_dataset import read_datetime_set
from ium_data.data_process.split_dataset import generate_datetime_set


class BJIterator(object):
    """
    Beijing dataset iterator warpper
    """
    def __init__(self, datetime_set, sample_mode, begin_idx=None, end_idx=None,
                 width=480, height=480, raw_width=800, raw_height=800,
                 seq_len=25, datetime_list_file="time_info.txt", stride=1):
        assert width <= raw_width and height <= raw_height
        self._raw_width = raw_width
        self._raw_height = raw_height
        self._width = width
        self._height = height
        self._stride = stride
        generate_datetime_set(seq_len)
        self._datetime_set = read_datetime_set(datetime_set, seq_len)
        self._length = len(self._datetime_set)
        self.set_begin_end(begin_idx, end_idx)
        self._seq_len = seq_len
        self._datetime_list = read_datetime_list(info_path, datetime_list_file)
        assert sample_mode in ["random", "sequent"]
        self._sample_mode = sample_mode
        if sample_mode == "sequent":
            self._current_idx = self._begin_idx

        # used for last_sample
        self.frame_dat = None
        self.mask_dat = None
        self.datetime_batch_real = None

    def set_begin_end(self, begin_idx=None, end_idx=None):
        self._begin_idx = 0 if begin_idx is None else begin_idx
        self._end_idx = self._length - 1 if end_idx is None else end_idx
        assert self._begin_idx >= 0 and self._end_idx < self._length
        assert self._begin_idx <= self._end_idx

    def reset(self, begin_idx=None, end_idx=None):
        assert self._sample_mode == "sequent"
        self.set_begin_end(begin_idx=begin_idx, end_idx=end_idx)
        self._current_idx = self._begin_idx

    def random_reset(self):
        assert self.sample_mode == "sequent"
        self.set_begin_end(begin_idx=np.random.randint(self._begin_idx, self._end_idx + 1))
        self._current_idx = self._begin_idx

    def check_new_start(self):
        '''should use the "time_mask_${seq_len}_timedelta_57.txt" to check new start
        '''
        assert self.sample_mode == "sequent"
        raise NotImplementedError


    @property
    def total_sample_num(self):
        return self._length

    @property
    def begin_time(self):
        return self._datetime_set[self._begin_idx]

    @property
    def end_time(self):
        return self._datetime_set[self._end_idx]

    @property
    def use_up(self):
        if self._sample_mode == "random":
            return False
        else:
            return self._current_idx > self._end_idx

    def get_frame_paths(self, datetime_batch):
        frame_paths = []
        for dt in datetime_batch:
            batch = self._datetime_list[dt : dt + self._seq_len]
            batch = [os.path.join(npy_radar_path, dt[0:8], dt[8:]+".npy") for dt in batch]
            frame_paths.append(batch)
        return np.array(frame_paths)

    def get_real_datetime_batch(self, datetime_batch):
        datetime_batch_real = []
        for dt in datetime_batch:
            batch = self._datetime_list[dt: dt + self._seq_len]
            datetime_batch_real.append(batch)
        return datetime_batch_real

    def load_frames(self, datetime_batch, offset_height, offset_width, normalization):
        assert isinstance(datetime_batch, list)
        frame_paths = self.get_frame_paths(datetime_batch)
        for dt in frame_paths:
            assert len(dt) == self._seq_len
        batch_size = len(frame_paths)
        frame_dat = np.zeros((self._seq_len, batch_size, 1, self._height, self._width),
                                dtype=np.uint8)
        mask_dat = np.ones((self._seq_len, batch_size, 1, self._height, self._width),
                                dtype=np.bool)
        if self._sample_mode == "random":
            paths = []
            hit_inds = []
            for i in range(self._seq_len):
                for j in range(batch_size):
                    paths.append(frame_paths[j,i])
                    hit_inds.append([i, j])
            hit_inds = np.array(hit_inds, dtype=np.int)
            all_frame_dat = quick_read_frames(path_list=paths,
                                              im_h=self._raw_height,
                                              im_w=self._raw_width)
            #print(all_frame_dat.shape)
            frame_dat[hit_inds[:,0], hit_inds[:,1], :, :, :] =\
            all_frame_dat[:,:,offset_height:(offset_height+self._height), offset_width:(offset_width+self._width)]
        else:
            # np.unique(frame_paths)
            uniq_paths = set()
            for i in range(self._seq_len):
                for j in range(batch_size):
                    uniq_paths.add(frame_paths[j,i])
            uniq_paths = list(uniq_paths)
            uniq_paths.sort()
            all_frame_dat = quick_read_frames(path_list=uniq_paths,
                                              im_h=self._raw_height,
                                              im_w=self._raw_width)
            #print(all_frame_dat.shape)
            for i in range(self._seq_len):
                for j in range(batch_size):
                    idx = uniq_paths.index(frame_paths[j,i])
                    frame_dat[i,j,:,:,:] =\
                    all_frame_dat[idx,:,offset_height:(offset_height+self._height), offset_width:(offset_width+self._width)]

        return frame_dat, mask_dat

    def sample(self, batch_size, only_return_datetime=False, normalization=True):
        if self._sample_mode == "sequent":
            if self.use_up:
                raise ValueError("The BJIterator has been used up!")
            datetime_batch = []
            #new_start = False
            offset_width = (self._raw_width - self._width) // 2
            offset_height = (self._raw_height - self._height) // 2
            for i in range(batch_size):
                if not self.use_up:
                    frame_idx = self._datetime_set[self._current_idx, 1]
                    #new_start = new_start or (self._current_idx == self._begin_idx)
                    datetime_batch.append(frame_idx)
                    self._current_idx += self._stride
            #new_start = None if batch_size != 1 else new_start


        if self._sample_mode == "random":
            datetime_batch = []
            #new_start = None
            offset_width = np.random.randint(0, self._raw_width - self._width + 1, 1)[0]
            offset_height = np.random.randint(0, self._raw_height - self._height + 1, 1)[0]
            for i in range(batch_size):
                rand_idx = np.random.randint(self._begin_idx, self._end_idx + 1, 1)[0]
                #frame_datetime = self._datetime_set[rand_idx, 0]
                frame_idx = self._datetime_set[rand_idx, 1]
                datetime_batch.append(frame_idx)
            '''
            rand_idx = np.random.randint(self._begin_idx, self._end_idx + 1 - self._seq_len * batch_size, 1)[0]
            for i in range(batch_size):
                frame_idx = self._datetime_set[rand_idx, 1]
                datetime_batch.append(frame_idx)
                rand_idx += self._seq_len
            '''
        # because "datetime_batch" only contain the start datetime of batch
        datetime_batch_real = self.get_real_datetime_batch(datetime_batch)
        if only_return_datetime:
            return datetime_batch_real, None

        frame_dat, mask_dat = self.load_frames(datetime_batch, offset_height, offset_width, normalization)

        # used for last_sample
        self.frame_dat = frame_dat
        self.mask_dat = mask_dat
        self.datetime_batch_real = datetime_batch_real

        return frame_dat, mask_dat, datetime_batch_real, None

    def last_sample(self):
        """ save last sample output, call sample first!
        """
        assert(self.frame_dat is not None)
        return self.frame_dat, self.mask_dat, self.datetime_batch_real, None



if __name__ == "__main__":
    np.random.seed(123344)
    import time
    import cProfile, pstats


    train_bj_iter = BJIterator(datetime_set="bj_train_set.txt",
                               sample_mode="random", seq_len=15,
                               width=480, height=480)
    '''
    mini_batch = 4
    valid_bj_iter = BJIterator(datetime_set="bj_cv_set.txt",
                               sample_mode="sequent",
                               end_idx=None)
    test_bj_iter = BJIterator(datetime_set="bj_test_set.txt",
                              sample_mode="sequent",
                              end_idx=None)
    repeat_time=3
    pr = cProfile.Profile()
    pr.enable()
    begin = time.time()
    for i in range(repeat_time):
        sample_sequence, sample_mask, sample_datetimes =\
            train_bj_iter.sample(batch_size=mini_batch)
    end = time.time()
    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.print_stats(20)
    print("Train Data Sample FPS: %f" % (mini_batch * 20
                                    * repeat_time / float(end - begin)))  

    '''
    begin = time.time()
    for i in range(100):
        frame_dat, mask_dat, datetime_batch, _ = train_bj_iter.sample(batch_size=4)
        print(datetime_batch[0][0])
        print(frame_dat.shape)

        #draw_all_frames(frame_dat[:24,0,...].squeeze() / 255, 8)
        #for frame in frame_dat:
        #    draw_frame(frame[0,0] / 255, True)
    end = time.time()   
    print(end - begin)
