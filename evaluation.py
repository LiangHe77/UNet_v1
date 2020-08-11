# -*- coding: utf-8 -*
import numpy as np
import os
import logging


def get_hit_miss_counts(prediction, truth, thresholds, mask=None, sum_batch=False):
    """This function calculates the overall hits and misses for the prediction, which could be used
    to get the skill scores and threat scores:


    This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
    and all inputs should be between 0~1

    Parameters
    ----------
    prediction : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    truth : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    mask : np.ndarray or None
        Shape: (seq_len, batch_size, 1, height, width)
        0 --> not use
        1 --> use
    thresholds : list or tuple

    Returns
    -------
    hits : np.ndarray
        (seq_len, len(thresholds)) or (seq_len, batch_size, len(thresholds))
        TP
    misses : np.ndarray
        (seq_len, len(thresholds)) or (seq_len, batch_size, len(thresholds))
        FN
    false_alarms : np.ndarray
        (seq_len, len(thresholds)) or (seq_len, batch_size, len(thresholds))
        FP
    correct_negatives : np.ndarray
        (seq_len, len(thresholds)) or (seq_len, batch_size, len(thresholds))
        TN
    """
    assert prediction.ndim == 5
    assert truth.ndim == 5
    assert prediction.shape == truth.shape
    assert prediction.shape[2] == 1
    
    thresholds = np.array(thresholds,dtype=np.float32)\
                          .reshape((1, 1, len(thresholds), 1, 1))
    bpred = (prediction >= thresholds)
    btruth = (truth >= thresholds)
    bpred_n = np.logical_not(bpred)
    btruth_n = np.logical_not(btruth)
    
    if sum_batch:
        summation_axis = (1,3,4)
    else:
        summation_axis = (3,4)
        
    if mask is None:
    
        hits = np.logical_and(bpred, btruth).sum(axis=summation_axis)
        misses = np.logical_and(bpred_n, btruth).sum(axis=summation_axis)
        false_alarms = np.logical_and(bpred, btruth_n).sum(axis=summation_axis)
        correct_negatives = np.logical_and(bpred_n, btruth_n).sum(axis=summation_axis)
    
    else:
        
        hits = np.logical_and(np.logical_and(bpred, btruth), mask)\
            .sum(axis=summation_axis)
        misses = np.logical_and(np.logical_and(bpred_n, btruth), mask)\
            .sum(axis=summation_axis)
        false_alarms = np.logical_and(np.logical_and(bpred, btruth_n), mask)\
            .sum(axis=summation_axis)
        correct_negatives = np.logical_and(np.logical_and(bpred_n, btruth_n), mask)\
            .sum(axis=summation_axis)  

    return hits, misses, false_alarms, correct_negatives   

def get_correlation(prediction, truth):
    """

    Parameters
    ----------
    prediction : np.ndarray
    truth : np.ndarray

    Returns
    -------

    """
    assert truth.shape == prediction.shape
    assert 5 == prediction.ndim
    assert prediction.shape[2] == 1
    eps = 1E-12
    ret = (prediction * truth).sum(axis=(3, 4)) / (
        np.sqrt(np.square(prediction).sum(axis=(3, 4))) * np.sqrt(np.square(truth).sum(axis=(3, 4))) + eps)
    ret = ret.sum(axis=(1, 2))
    return ret    
    
'''    
def get_balancing_weights(data, mask, base_balancing_weights=[1,2,5,10,30], thresholds=[10,20,30,40]):

    weights = np.ones_like(data) * base_balancing_weights[0]
    threshold_mask = np.expand_dims(data, axis=5) >= thresholds
    base_weights = np.diff(np.array(base_balancing_weights, dtype=np.float32))\
        .reshape((1, 1, 1, 1, 1, len(base_balancing_weights) - 1))
    weights += (threshold_mask * base_weights).sum(axis=-1)
    weights *= mask
    return weights
'''    
class HKOEvaluation(object):
    def __init__(self, seq_len, use_central, thresholds=[10,20,30,40], central_region=[120,120,360,360]):
    
        self._seq_len = seq_len
        self._use_central = use_central
        self._thresholds = thresholds
        self._central_region = central_region
        self.begin()
        
    def begin(self):
        
        self._total_hits = np.zeros((self._seq_len, len(self._thresholds)), \
                                     dtype = np.int)
        self._total_misses = np.zeros((self._seq_len, len(self._thresholds)),\
                                     dtype = np.int)
        self._total_false_alarms = np.zeros((self._seq_len, len(self._thresholds)),\
                                     dtype = np.int)
        self._total_correct_negatives = np.zeros((self._seq_len, len(self._thresholds)),\
                                     dtype = np.int)
        self._datetime_dict = {}
        self._total_batch_num = 0  

    def clear_all(self):
        
        self._total_hits[:] = 0
        self._total_misses[:] = 0
        self._total_false_alarms[:] = 0
        self._total_correct_negatives[:] = 0    
        self._total_batch_num = 0        
                                     
    def update(self, gt, pred, mask, start_datetimes=None):
        """

        Parameters
        ----------
        gt(ground_truth) : np.ndarray
        pred : np.ndarray
        mask : np.ndarray
            0 indicates not use and 1 indicates that the location will be taken into account
        start_datetimes : list
            The starting datetimes of all the testing instances

        Returns
        -------

        """
        if start_datetimes is not None:
        
            batch_size = len(start_datetimes)
            assert gt.shape[1] == batch_size
            
        else:
        
            batch_size = gt.shape[1]
            
        assert gt.shape[0] == self._seq_len
        assert gt.shape == pred.shape, "gt.shape{}, pred.shape{}".format(gt.shape, pred.shape)
        #assert gt.shape == mask.shape

        if self._use_central:
            # Crop the central regions for evaluation
            pred = pred[:, :, :,
                        self._central_region[1]:self._central_region[3],
                        self._central_region[0]:self._central_region[2]]
            gt = gt[:, :, :,
                    self._central_region[1]:self._central_region[3],
                    self._central_region[0]:self._central_region[2]]
            mask = mask[:, :, :,
                        self._central_region[1]:self._central_region[3],
                        self._central_region[0]:self._central_region[2]]
                        
        self._total_batch_num += batch_size
        
        #TODO Save all the hits, misses, false_alarms and correct_negatives
        
        hits, misses, false_alarms, correct_negatives = \
            get_hit_miss_counts(prediction=pred, truth=gt, mask=mask,
                                      thresholds=self._thresholds)
                                      
        self._total_hits += hits.sum(axis=1)
        self._total_misses += misses.sum(axis=1)
        self._total_false_alarms += false_alarms.sum(axis=1)
        self._total_correct_negatives += correct_negatives.sum(axis=1)
    
    def calculate_stat(self):
        """The following measurements will be used to measure the score of the forecaster
        See Also
        [Weather and Forecasting 2010] Equitability Revisited: Why the "Equitable Threat Score" Is Not Equitable
        http://www.wxonline.info/topics/verif2.html

        We will denote
        (a b    (hits       false alarms
         c d) =  misses   correct negatives)

        We will report the
        POD = a / (a + c)
        FAR = b / (a + b)
        CSI = a / (a + b + c)
        Heidke Skill Score (HSS) = 2(ad - bc) / ((a+c) (c+d) + (a+b)(b+d))
        Gilbert Skill Score (GSS) = HSS / (2 - HSS), also known as the Equitable Threat Score
            HSS = 2 * GSS / (GSS + 1)
        MSE = mask * (pred - gt) **2
        MAE = mask * abs(pred - gt)
        GDL = valid_mask_h * abs(gd_h(pred) - gd_h(gt)) + valid_mask_w * abs(gd_w(pred) - gd_w(gt))
        Returns
        -------
        """
        np.seterr(divide='ignore',invalid='ignore')   #忽视分母为0的情况        

        a = self._total_hits.astype(np.float64)
        b = self._total_false_alarms.astype(np.float64)
        c = self._total_misses.astype(np.float64)
        d = self._total_correct_negatives.astype(np.float64)
        
        POD = a / (a + c)
        FAR = b / (a + b)
        CSI = a / (a + b + c)
        
        return POD, CSI, FAR

    def print_stat_readable(self, prefix=""):
        
        logging.info("%sTotal Sequence Number: %d, Use Central: %d"
                     %(prefix, self._total_batch_num, self._use_central))
                     
        pod, csi, far = self.calculate_stat() 
        
        logging.info("   POD: " + ', '.join([">%g:%g/%g" % (threshold, pod[:, i].mean(), pod[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   FAR: " + ', '.join([">%g:%g/%g" % (threshold, far[:, i].mean(), far[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   CSI: " + ', '.join([">%g:%g/%g" % (threshold, csi[:, i].mean(), csi[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))        
    
    def save_txt_readable(self, index, path):
    
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        pod, csi, far = self.calculate_stat()

        pod, csi, far = np.nan_to_num(pod), \
                        np.nan_to_num(csi), \
                        np.nan_to_num(far)

        # pod, far, csi, hss, gss, mse, mae, gdl = self.calculate_stat()
        f = open(path, 'a')
        logging.info("Saving readable txt of HKOEvaluation to %s" % path)
        f.write("Total Sequence Num: %d, Out Seq Len: %d, Use Central: %d\n"
                %(self._total_batch_num,
                  self._seq_len,
                  self._use_central))

        mid_idx = (self._seq_len - 1) // 2 

        f.write("This is {} model\n".format(index))
        
        for (i, threshold) in enumerate(self._thresholds):
            
            f.write("Threshold = %g:\n" %threshold)
            f.write("   POD: %s\n" % str(list(pod[:, i])))
            f.write("   FAR: %s\n" % str(list(far[:, i])))
            f.write("   CSI: %s\n" % str(list(csi[:, i])))

            f.write("   POD stat: avg %g/mid %g/final %g\n" %(pod[:, i].mean(),pod[mid_idx, i],pod[-1, i]))
            f.write("   FAR stat: avg %g/mid %g/final %g\n" %(far[:, i].mean(),far[mid_idx, i],far[-1, i]))
            f.write("   CSI stat: avg %g/mid %g/final %g\n" %(csi[:, i].mean(),csi[mid_idx, i],csi[-1, i]))
            f.write(" ******************** \n")

        f.close()

        return csi.mean(axis=0)
    
    
