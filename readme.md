# Dynamic Temporal Warping(DTW) in Tensorflow

Implementation of DTW computation in Tensorflow, a GPU-accelerated version of DTW computation.

## Introduction

Prepared for a dataset with `N` temporal sequences, each in shape of `[T, d]`.

You will obtain a DTW distance matrix (`[N,N]`) by:
```python
    dtw_dist_mat = tf_dtw(dataset, lens)
```

## Demo

Run `test.py` to compare DTW matrix computation within `50 sequences` between `fastdtw` implementation and this version.
```python
    python demo.py
```

## Requirement

1. tensorflow>=1.4
2. fastdtw

## Reference

[TensorFlow rnn.py](https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/rnn.py)
