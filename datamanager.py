# coding:utf-8
import numpy as np

def padding(minibatch, maxlen=None):
    '''
        pad sequences [T, d] to [maxlen, d]
    '''
    lens = map(len, minibatch)
    dim = minibatch[0].shape[-1]
    maxlen = maxlen or max(lens)
    res = []
    for i in range(len(minibatch)):
		if len(minibatch[i]) > maxlen:
			res.append(minibatch[i][:maxlen, :])
		else:
			res.append(np.concatenate([minibatch[i], np.zeros([maxlen-lens[i], dim])], axis=0))
    # [batch_size, maxlen, d]
    return np.asarray(res)

def gen_dataset(N=100, minlen=150, maxlen=200, datadim=3):
    '''
        generate random sequences, each sample is [T, datadim],  minlen<=T<=maxlen
    '''
    np.random.seed(0)
    rand_lens = np.random.randint(minlen, maxlen+1, size=(N,))
    np.random.seed(0)
    dataset = [np.random.normal(size=(int(l), datadim)) for l in rand_lens]
    
    return np.array(dataset), rand_lens