# coding:utf-8
import tensorflow as tf 
import numpy as np
import time
from datamanager import gen_dataset, padding
from tf_dtw import tf_dtw
from np_dtw import np_dtw
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt 


def test():
    '''
        50 samples, length from 100 to 200, compute DTW distance matrix.
    '''
    dataset, lens = gen_dataset(N=50, minlen=100, maxlen=200, datadim=3)
    dataset = padding(dataset, maxlen=200)

    start = time.time()
    dtw_mat_tf = tf_dtw(dataset, lens, batch_size=50*(50-1))
    print "tf version cost {} sec.".format(time.time()-start) # 19.5 sec
    
    start = time.time()
    dtw_mat_np = np_dtw(dataset, lens)
    print "np version cost {} sec.".format(time.time()-start) # 188.3 sec

    dtw_mat_tf = np.around(dtw_mat_tf, decimals=3)
    dtw_mat_np = np.around(dtw_mat_np, decimals=3)

    print "loss = {}".format(np.mean(np.abs(dtw_mat_tf - dtw_mat_np)))

if __name__ == "__main__":
    test()